import datetime
import logging
import random
import time
from enum import IntEnum
import uuid

import panel as pn
import polars as pl
import upath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  #! doesn't take effect

# pn.config.theme = 'dark'
pn.config.notifications = True

CACHE_VERSION = "v0.0.261"
SCRATCH_DIR = upath.UPath("s3://aind-scratch-data/dynamic-routing")
RASTER_DIR = SCRATCH_DIR / f"unit-rasters/{CACHE_VERSION}/fig3c"

TABLE_NAME = "from_app"
DB_PATH = "//allen/programs/mindscope/workgroups/dynamicrouting/ben/unit_drift.parquet"

df_path = "s3://aind-scratch-data/dynamic-routing/cache/nwb_components/{}/consolidated/{}.parquet"
epochs_df = pl.read_parquet(df_path.format(CACHE_VERSION, "epochs"))
units_df = (
    pl.scan_parquet(df_path.format(CACHE_VERSION, "units"))
    .select(
        [
            "session_id",
            "unit_id",
            "structure",
            "default_qc",
            "presence_ratio",
            "drift_ptp",
            "obs_intervals",
        ]
    )
    .collect()
    # filter out units that are only partially-observed within the task
    .join(
        other=(
            epochs_df
            .filter(
                pl.col('script_name') == 'DynamicRouting1'
            )
            .select('session_id', 'stop_time', 'start_time')
        ),
        on='session_id',
        how='left',
    ).filter(
        pl.col('obs_intervals').list.get(0).list.get(0).le(pl.col('start_time')),
        pl.col('obs_intervals').list.get(0).list.get(1).ge(pl.col('stop_time')),
    )
    .drop("obs_intervals")
)

class UnitDriftRating(IntEnum):
    NO = 0
    YES = 1
    UNSURE = 5
    
class Lock:
    def __init__(self, path=DB_PATH + ".lock"):
        self.path = upath.UPath(path)
        
    def acquire(self):
        while self.path.exists():
            time.sleep(0.01)
        self.path.touch()
                
    def release(self):
        upath.UPath(self.path).unlink()
        
    def __enter__(self):
        self.acquire()
        logger.info(f"Acquired lock at {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        logger.info(f"Released lock at {self.path}")


def create_db(
    save_path: str = DB_PATH, overwrite: bool = False, with_paths=True
) -> None:
    if upath.UPath(save_path).exists() and not overwrite:
        raise FileExistsError(
            f"{save_path} already exists: set overwrite=True to overwrite"
        )
    df = units_df.select("unit_id").with_columns(
        drift_rating=pl.lit(None),
        checked_timestamp=pl.lit(None),
    )
    if with_paths:
        logger.warning("getting paths from S3 for all available units")
        paths = tuple(RASTER_DIR.rglob("*"))
        paths_df = pl.DataFrame(
            {
                "unit_id": [
                    p.stem.split("fig3c_")[1] for p in paths if "fig3c_" in str(p)
                ],
                "path": [str(p) for p in paths if "fig3c_" in str(p)],
            }
        )
        df = df.join(paths_df, on="unit_id", how="left")
    else:
        df = df.with_columns(path=pl.lit(None))
    df.write_parquet(save_path)
    return

def reset_db(src_path: str, dest_path: str) -> None:
    pl.read_parquet(src_path).with_columns(
        drift_rating=pl.lit(None), checked_timestamp=pl.lit(None)
    ).write_parquet(dest_path)


def get_df(
    already_checked: bool | None = None,
    unit_id_filter: str | None = None,
    session_id_filter: str | None = None,
    drift_rating_filter: int | None = None,
    with_paths: bool | None = True,
    db_path=DB_PATH,
) -> pl.DataFrame:
    filter_exprs = []
    if with_paths is True:
        filter_exprs.append(pl.col("path").is_not_null())
    elif with_paths is False:
        filter_exprs.append(pl.col("path").is_null())
    if already_checked is True and drift_rating_filter is None:
        filter_exprs.append(pl.col("drift_rating").is_not_null())
    elif already_checked is False and drift_rating_filter is None:
        filter_exprs.append(pl.col("drift_rating").is_null())
    if session_id_filter:
        filter_exprs.append(pl.col("session_id").str.starts_with(session_id_filter))
    if unit_id_filter:
        filter_exprs.append(pl.col("unit_id") == unit_id_filter)
    if drift_rating_filter is not None:
        filter_exprs.append(pl.col("drift_rating") == int(drift_rating_filter))
    if filter_exprs:
        logger.info(
            f"Filtering units df with {' & '.join([str(f) for f in filter_exprs])}"
        )
    return pl.read_parquet(db_path).filter(*filter_exprs)


def unit_generator(
    already_checked: bool | None = False,
    unit_id_filter: str | None = None,
    session_id_filter: str | None = None,
    drift_rating_filter: int | None = None,
    with_paths: bool | None = True,
    db_path=DB_PATH,
):
    while True:
        df = get_df(
            already_checked=already_checked,
            unit_id_filter=unit_id_filter,
            session_id_filter=session_id_filter,
            drift_rating_filter=drift_rating_filter,
            with_paths=with_paths,
            db_path=db_path,
        )
        session_ids = df["session_id"].unique().to_list()
        if not session_ids:
            raise StopIteration("No more sessions to check")
        random.shuffle(session_ids)
        for session_id in session_ids:
            sub_df = df.filter(pl.col("session_id") == session_id)
            if sub_df.is_empty():
                logger.info(f"No more units to check for {session_id}")
                continue
            yield sub_df.sample(1)["unit_id"].first()


def update_db(unit_id: str, drift_rating: int, db_path=DB_PATH) -> None:
    timestamp = int(time.time())
    session_id = "_".join(unit_id.split("_")[:2])
    original_df = get_df()
    unit_id_filter = pl.col("unit_id") == unit_id
    logger.info(f"Updating row for {unit_id} with drift_rating={drift_rating}")
    df = original_df.with_columns(
        drift_rating=pl.when(unit_id_filter)
        .then(pl.lit(drift_rating))
        .otherwise(pl.col("drift_rating")),
        checked_timestamp=pl.when(unit_id_filter)
        .then(pl.lit(timestamp))
        .otherwise(pl.col("checked_timestamp")),
    )
    assert len(df) == len(
        original_df
    ), f"Row count changed: {len(original_df)} -> {len(df)}"
    with Lock():
        df.write_parquet(db_path)
    logger.info(f"Overwrote {db_path}")


def test_db(with_paths=False):
    db_path = "test.parquet"
    create_db(overwrite=True, save_path=db_path, with_paths=with_paths)
    i = next(unit_generator(db_path=db_path))
    update_db(i, 2, db_path=db_path)
    df = get_df(unit_id_filter=i, already_checked=True, db_path=db_path)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    upath.UPath(db_path).unlink()


def get_raster(unit_id: str) -> bytes:
    filtered_df = get_df(unit_id_filter=unit_id, with_paths=True)
    if len(filtered_df) > 1:
        import pdb

        pdb.set_trace()
    assert len(filtered_df) == 1, "df filtering likely incorrect"
    path: str = filtered_df["path"].first()
    assert path is not None, f"Path not stored for {unit_id}"
    logger.info(f"Getting raster image data from {path}")
    t0 = time.time()
    b = upath.UPath(path).read_bytes()
    logger.info(f"Got raster image data in {time.time() - t0:.2f}s")
    return b


def get_metrics(unit_id: str):
    return units_df.filter(pl.col("unit_id") == unit_id).to_dicts()[0]


def display_metrics(
    unit_id=str,
) -> pn.pane.Markdown:
    metrics = get_metrics(unit_id)
    stats = f"""

### `{unit_id}`
"""
    for k, v in metrics.items():
        if k not in ("structure", "presence_ratio", "default_qc", "drift_ptp"):
            continue
        if isinstance(v, float):
            v = f"{v:.2f}"
        stats += f"\n{k.replace('_', ' ')}:\n`{v if v else '-'}`\n"
    return pn.pane.Markdown(stats)


def display_rating(
    unit_id=str,
) -> pn.pane.Markdown:
    df = get_df(unit_id_filter=unit_id)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    row = df.to_dicts()[0]
    rating: int | None = row["drift_rating"]
    if rating is None:
        return pn.pane.Markdown("not yet rated")
    else:
        return pn.pane.Markdown(
            f"**{UnitDriftRating(rating).name.title()}** ({datetime.datetime.fromtimestamp(row['checked_timestamp']).strftime('%Y-%m-%d %H:%M:%S')})"
        )


def display_image(unit_id: str):
    return pn.pane.PNG(
        get_raster(unit_id),
        sizing_mode="stretch_height",
    )

if not upath.UPath(DB_PATH).exists():
    create_db()
    
unit_generator_params = dict(
    already_checked=False,
    with_paths=True,
    unit_id_filter=None,
    drift_rating_filter=None,
)
unit_id_generator = unit_generator(**unit_generator_params)
current_unit_id = next(unit_id_generator)
previous_unit_id: str | None = None
unit_metrics_pane = display_metrics(current_unit_id)
unit_rating_pane = display_rating(current_unit_id)
raster_image_pane = display_image(current_unit_id)


def update_and_next(unit_id: str, drift_rating: int) -> None:
    update_db(unit_id=unit_id, drift_rating=drift_rating)
    next_unit()


def next_unit(override_next_unit_id: str | None = None) -> None:
    global current_unit_id, previous_unit_id
    if override_next_unit_id:
        current_unit_id = override_next_unit_id
        undo_button.disabled = True
    else:
        previous_unit_id = current_unit_id
        undo_button.disabled = False
        try:
            current_unit_id = next(unit_id_generator)
        except StopIteration:
            pn.state.notifications.warning("No matching units")
    raster_image_pane.object = display_image(current_unit_id).object
    unit_metrics_pane.object = display_metrics(current_unit_id).object
    unit_rating_pane.object = display_rating(current_unit_id).object


SIDEBAR_WIDTH = 230
BUTTON_WIDTH = int(SIDEBAR_WIDTH * 0.8)

# make three buttons for rating drift, which all call update_unit_id
# 0: no, 1: yes, 2: unsure
no_button_text = f"{UnitDriftRating.NO.name.title()} [{UnitDriftRating.NO.value}]"
no_button = pn.widgets.Button(name=no_button_text, width=BUTTON_WIDTH)
no_button.on_click(
    lambda event: update_and_next(
        unit_id=current_unit_id, drift_rating=UnitDriftRating.NO
    )
)
yes_button_text = f"{UnitDriftRating.YES.name.title()} [{UnitDriftRating.YES.value}]"
yes_button = pn.widgets.Button(name=yes_button_text, width=BUTTON_WIDTH)
yes_button.on_click(
    lambda event: update_and_next(
        unit_id=current_unit_id, drift_rating=UnitDriftRating.YES
    )
)
unsure_button_text = (
    f"{UnitDriftRating.UNSURE.name.title()} [{UnitDriftRating.UNSURE.value}]"
)
unsure_button = pn.widgets.Button(name=unsure_button_text, width=BUTTON_WIDTH)
unsure_button.on_click(
    lambda event: update_and_next(
        unit_id=current_unit_id, drift_rating=UnitDriftRating.UNSURE
    )
)
skip_button = pn.widgets.Button(name="Skip [s]", width=BUTTON_WIDTH)
skip_button.on_click(lambda event: next_unit())
undo_button = pn.widgets.Button(name="Previous [p]", width=BUTTON_WIDTH)
undo_button.disabled = True
undo_button.on_click(lambda event: next_unit(override_next_unit_id=previous_unit_id))

# - ---------------------------------------------------------------- #
# from https://github.com/holoviz/panel/issues/3193#issuecomment-2357189979
from typing import TypedDict, NotRequired

# Note: this uses TypedDict instead of Pydantic or dataclass because Bokeh/Panel doesn't seem to
# like serializing custom classes to the frontend (and I can't figure out how to customize that).
class KeyboardShortcut(TypedDict):
    name: str
    key: str
    altKey: NotRequired[bool]
    ctrlKey: NotRequired[bool]
    metaKey: NotRequired[bool]
    shiftKey: NotRequired[bool]

from panel.custom import ReactComponent
import param

class KeyboardShortcuts(ReactComponent):
    """
    Class to install global keyboard shortcuts into a Panel app.

    Pass in shortcuts as a list of KeyboardShortcut dictionaries, and then handle shortcut events in Python
    by calling `on_msg` on this component. The `name` field of the matching KeyboardShortcut will be sent as the `data`
    field in the `DataEvent`.

    Example:
    >>> shortcuts = [
        KeyboardShortcut(name="save", key="s", ctrlKey=True),
        KeyboardShortcut(name="print", key="p", ctrlKey=True),
    ]
    >>> shortcuts_component = KeyboardShortcuts(shortcuts=shortcuts)
    >>> def handle_shortcut(event: DataEvent):
            if event.data == "save":
                print("Save shortcut pressed!")
            elif event.data == "print":
                print("Print shortcut pressed!")
    >>> shortcuts_component.on_msg(handle_shortcut)
    """

    shortcuts = param.List(class_=dict)

    _esm = """
    // Hash a shortcut into a string for use in a dictionary key (booleans / null / undefined are coerced into 1 or 0)
    function hashShortcut({ key, altKey, ctrlKey, metaKey, shiftKey }) {
      return `${key}.${+!!altKey}.${+!!ctrlKey}.${+!!metaKey}.${+!!shiftKey}`;
    }

    export function render({ model }) {
      const [shortcuts] = model.useState("shortcuts");

      const keyedShortcuts = {};
      for (const shortcut of shortcuts) {
        keyedShortcuts[hashShortcut(shortcut)] = shortcut.name;
      }

      function onKeyDown(e) {
        const name = keyedShortcuts[hashShortcut(e)];
        if (name) {
          e.preventDefault();
          e.stopPropagation();
          model.send_msg(name);
          return;
        }
      }

      React.useEffect(() => {
        window.addEventListener('keydown', onKeyDown);
        return () => {
          window.removeEventListener('keydown', onKeyDown);
        };
      });

      return <></>;
    }
    """
shortcuts = [
    KeyboardShortcut(name="skip", key="s", ctrlKey=False),
    KeyboardShortcut(name="previous", key="p", ctrlKey=False),
] + [
    KeyboardShortcut(name=k, key=str(v), ctrlKey=False) for k, v in UnitDriftRating.__members__.items()
]
shortcuts_component = KeyboardShortcuts(shortcuts=shortcuts)
def handle_shortcut(event):
        if event.data == "skip":
            next_unit()
        elif event.data == "previous":
            next_unit(override_next_unit_id=previous_unit_id)
        else:
            update_and_next(
                unit_id=current_unit_id, drift_rating=UnitDriftRating[event.data]
            )
shortcuts_component.on_msg(handle_shortcut)

# - ---------------------------------------------------------------- #

unit_generator_params = dict(
    already_checked=False,
    with_paths=True,
    unit_id_filter=None,
    session_id_filter=None,
    drift_rating_filter=None,
)


def update_unit_generator(event) -> None:
    global unit_id_generator
    unit_generator_params = dict(
        unit_id_filter=unit_id_filter_text.value or None,
        session_id_filter=session_id_filter_text.value or None,
        with_paths=True,
    )
    if drift_rating_filter_radio.value == "unrated":
        unit_generator_params["already_checked"] = False
    else:
        unit_generator_params["drift_rating_filter"] = UnitDriftRating[
            drift_rating_filter_radio.value.upper()
        ].value
    logger.info(f"Updating unit generator with {unit_generator_params}")
    unit_id_generator = unit_generator(**unit_generator_params)
    next_unit()


unit_id_filter_text = pn.widgets.TextInput(
    name="Unit ID filter", value="", placeholder="Matches exact ID", width=BUTTON_WIDTH
)
unit_id_filter_text.param.watch(update_unit_generator, "value")
session_id_filter_text = pn.widgets.TextInput(
    name="Session ID filter",
    value="",
    placeholder="Matches using 'startswith'",
    width=BUTTON_WIDTH,
)
session_id_filter_text.param.watch(update_unit_generator, "value")
drift_rating_filter_radio = pn.widgets.RadioBoxGroup(
    name="Show rated units",
    options=["unrated"] + [k.lower() for k in UnitDriftRating.__members__],
    inline=False,
)
drift_rating_filter_radio.param.watch(update_unit_generator, "value")


def app():
    sidebar = pn.Column(
        unit_metrics_pane,
        pn.layout.Divider(margin=(20, 0, 15, 0)),
        pn.pane.Markdown("""**Does the unit's activity drift in or out?**"""),
        no_button,
        yes_button,
        unsure_button,
        unit_rating_pane,
        undo_button,
        skip_button,
        shortcuts_component,
        pn.layout.Divider(margin=(20, 0, 15, 0)),
        pn.pane.Markdown("### Filter units"),
        drift_rating_filter_radio,
        unit_id_filter_text,
        session_id_filter_text,
    )

    return pn.template.MaterialTemplate(
        site="DR dashboard",
        title="unit drift qc",
        sidebar=[sidebar],
        main=[raster_image_pane],
        sidebar_width=SIDEBAR_WIDTH,
    )


app().servable()
