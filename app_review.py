import logging
import random
import time
from typing import Generator, NotRequired, TypedDict

import numpy as np
import panel as pn
import param
import polars as pl
import upath
from panel.custom import ReactComponent

import db_utils

try:
    pass
except ImportError:
    pass  # filenames are different on vm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  #! doesn't take effect

# pn.config.theme = 'dark'
pn.config.notifications = True


SIDEBAR_WIDTH = 230
BUTTON_WIDTH = int(SIDEBAR_WIDTH * 0.8)

if not upath.UPath(db_utils.DB_PATH).exists():
    db_utils.create_db()

units_df: None | pl.DataFrame = (
    db_utils.get_units_df()
    .join(
        pl.concat(
            [
                pl.scan_parquet(
                    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda_all.parquet"
                )
                .select("unit_id", "lda")
                .collect(),
                pl.scan_parquet(
                    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda_ks4.parquet"
                )
                .select("unit_id", "lda")
                .collect(),
            ]
        ),
        on="unit_id",
        how="left",
    )
    .join(
        pl.scan_parquet(db_utils.DB_PATH)
        .select("unit_id", "drift_rating", "path")
        .collect(),
        on="unit_id",
        how="left",
    )
    .filter(pl.col("lda").fill_nan(None).is_not_null())
)


def get_metrics(unit_id: str):
    return units_df.filter(pl.col("unit_id") == unit_id).to_dicts()[0]


def get_df(
    already_checked: bool | None = None,
    unit_id_filter: str | None = None,
    session_id_filter: str | None = None,
    drift_rating_filter: int | None = None,
    with_paths: bool | None = True,
    ks4_filter: bool | None = None,
    presence_ratio_filter: tuple[float, float] | None = None,
    sort_by_column: str | None = None,
    descending: bool = False,
    classification_filter: str | None = None,
    lda_threshold: float | None = None,
    db_path=db_utils.DB_PATH,
) -> pl.DataFrame:
    filter_exprs = []
    if with_paths is True:
        filter_exprs.append(pl.col("path").is_not_null())
    elif with_paths is False:
        filter_exprs.append(pl.col("path").is_null())
    if (
        already_checked is True
        and drift_rating_filter is None
        and classification_filter.lower() not in ("tp", "fp", "tn", "fn")
    ):
        filter_exprs.append(pl.col("drift_rating").is_not_null())
    elif (
        already_checked is False
        and drift_rating_filter is None
        and classification_filter.lower() not in ("tp", "fp", "tn", "fn")
    ):
        filter_exprs.append(pl.col("drift_rating").is_null())
    if session_id_filter:
        filter_exprs.append(pl.col("session_id").str.starts_with(session_id_filter))
    if unit_id_filter:
        filter_exprs.append(pl.col("unit_id") == unit_id_filter)
    if drift_rating_filter is not None:
        filter_exprs.append(pl.col("drift_rating") == int(drift_rating_filter))
    if ks4_filter is True:
        filter_exprs.append(pl.col("unit_id").str.ends_with("_ks4"))
    elif ks4_filter is False:
        filter_exprs.append(pl.col("unit_id").str.ends_with("_ks4").not_())
    if filter_exprs:
        logger.info(
            f"Filtering units df with {' & '.join([str(f) for f in filter_exprs])}"
        )
    if presence_ratio_filter:
        filter_exprs.append(pl.col("presence_ratio").ge(presence_ratio_filter[0]))
        filter_exprs.append(pl.col("presence_ratio").le(presence_ratio_filter[1]))

    if classification_filter and classification_filter not in ("all", "none"):
        lda_threshold = fpr_to_classifier_info(fpr_slider.value)["lda_threshold"]
        if classification_filter == "yes":
            filter_exprs.append(pl.col("lda") >= lda_threshold)
        elif classification_filter == "no":
            filter_exprs.append(pl.col("lda") < lda_threshold)

    if sort_by_column:
        return units_df.filter(*filter_exprs).sort(
            sort_by_column, descending=descending
        )
    else:
        return units_df.filter(*filter_exprs)


def unit_generator(
    df: pl.DataFrame,
    shuffle: bool = True,
) -> Generator[str, None, None]:
    while True:
        if shuffle:
            session_ids = df["session_id"].unique().to_list()
            if not session_ids:
                raise StopIteration("No more sessions to check")
            random.shuffle(session_ids)  # shuffle in place
            for session_id in session_ids:
                sub_df = df.filter(pl.col("session_id") == session_id)
                if sub_df.is_empty():
                    logger.info(f"No more units to check for {session_id}")
                    continue
                yield str(sub_df.sample(1)["unit_id"].first())  # cast to str for mypy
        else:
            for unit_id in df["unit_id"]:
                yield str(unit_id)


def display_metrics(
    unit_id=str,
) -> pn.pane.Markdown:
    metrics = get_metrics(unit_id)
    stats = f"""

### `{unit_id}`
"""
    for k, v in metrics.items():
        if k not in (
            "structure",
            "presence_ratio",
            "default_qc",
            "drift_ptp",
            "lda",
            "drift_rating",
        ):
            continue
        if isinstance(v, float):
            v = f"{v:.2f}"
        stats += f"\n{k.replace('_', ' ')}:\n`{v if v is not None else '-'}`\n"
    return pn.pane.Markdown(stats)


def get_raster(unit_id: str) -> bytes:
    filtered_df = get_df(unit_id_filter=unit_id, with_paths=True)
    assert len(filtered_df) == 1, "df filtering likely incorrect"
    path: str = filtered_df["path"].first()
    assert path is not None, f"Path not stored for {unit_id}"
    logger.info(f"Getting raster image data from {path}")
    t0 = time.time()
    image_bytes = upath.UPath(path).read_bytes()
    logger.info(f"Got raster image data in {time.time() - t0:.2f}s")
    return image_bytes


def display_image(unit_id: str):
    return pn.pane.PNG(
        get_raster(unit_id),
        sizing_mode="stretch_height",
    )


# initialize for startup, then update as a global variable
unit_id_generator = unit_generator(get_df())
try:
    current_unit_id = next(unit_id_generator)
except StopIteration:
    pn.state.notifications.warning("No matching units")
previous_unit_id: str | None = None
unit_metrics_pane = display_metrics(current_unit_id)
raster_image_pane = display_image(current_unit_id)


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


def update_and_next(unit_id: str, drift_rating: int) -> None:
    next_unit()


skip_button = pn.widgets.Button(name="Skip [s]", width=BUTTON_WIDTH)
skip_button.on_click(lambda event: next_unit())
undo_button = pn.widgets.Button(name="Previous [p]", width=BUTTON_WIDTH)
undo_button.disabled = True
undo_button.on_click(lambda event: next_unit(override_next_unit_id=previous_unit_id))

# - ---------------------------------------------------------------- #
# from https://github.com/holoviz/panel/issues/3193#issuecomment-2357189979


# Note: this uses TypedDict instead of Pydantic or dataclass because Bokeh/Panel doesn't seem to
# like serializing custom classes to the frontend (and I can't figure out how to customize that).
class KeyboardShortcut(TypedDict):
    name: str
    key: str
    altKey: NotRequired[bool]
    ctrlKey: NotRequired[bool]
    metaKey: NotRequired[bool]
    shiftKey: NotRequired[bool]


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
        // Ignore key presses in input, textarea, or content-editable elements
        const target = e.target;
        const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;
        if (isInput) {
          return;
        }
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
]
shortcuts_component = KeyboardShortcuts(shortcuts=shortcuts)


def handle_shortcut(event):
    if event.data == "skip":
        next_unit()
    elif event.data == "previous":
        next_unit(override_next_unit_id=previous_unit_id)
    else:
        raise ValueError(f"Unknown shortcut: {event.data}")


shortcuts_component.on_msg(handle_shortcut)

# - ---------------------------------------------------------------- #


def update_unit_generator(event) -> None:
    global unit_id_generator
    unit_generator_params = dict(
        unit_id_filter=unit_id_filter_text.value or None,
        session_id_filter=session_id_filter_text.value or None,
        with_paths=True,
        presence_ratio_filter=presence_ratio_range_slider.value,
        ks4_filter=ks4_checkbox.value,
        sort_by_column=sort_by_column_select.value,
        descending=descending_checkbox.value,
        classification_filter=classification_filter_radio.value,
        lda_threshold=fpr_slider.value,
    )
    if drift_rating_filter_radio.value == "unrated":
        unit_generator_params["already_checked"] = False
    elif drift_rating_filter_radio.value == "all":
        unit_generator_params["already_checked"] = None
    else:
        unit_generator_params["drift_rating_filter"] = db_utils.UnitDriftRating[
            drift_rating_filter_radio.value.upper()
        ].value

    logger.info(f"Updating unit generator with {unit_generator_params}")
    unit_id_generator = unit_generator(
        get_df(**unit_generator_params), shuffle=shuffle_checkbox.value
    )
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
    name="Select annotated units",
    options=["all", "unrated"]
    + [k.lower() for k in db_utils.UnitDriftRating.__members__],
    inline=False,
    value="all",
)
drift_rating_filter_radio.param.watch(update_unit_generator, "value")

classification_filter_radio = pn.widgets.RadioBoxGroup(
    name="Select classification",
    options=[
        "all",
        "yes",
        "no",
    ],
    inline=False,
    value="all",
)
classification_filter_radio.param.watch(update_unit_generator, "value")

presence_ratio_range_slider = pn.widgets.RangeSlider(
    name="Presence ratio",
    start=0,
    end=1,
    value=(0.95, 1),
    step=0.05,
    width=BUTTON_WIDTH,
)
presence_ratio_range_slider.param.watch(update_unit_generator, "value")

ks4_checkbox = pn.widgets.Checkbox(name="Kilosort 4", value=False)
ks4_checkbox.param.watch(update_unit_generator, "value")

fpr_slider = pn.widgets.FloatSlider(
    name="FPR",
    start=0,
    end=1,
    value=0.2,
    step=0.01,
    width=BUTTON_WIDTH,
)


def get_roc_curve_df(
    metric: str, n_points=50, logspace=False, below_threshold=False
) -> pl.DataFrame:
    df = units_df
    YES = db_utils.UnitDriftRating.YES.value
    NO = db_utils.UnitDriftRating.NO.value
    # only use the subset of the data where the metric overlaps between the two drift ratings
    metric_min = max(
        [float(df.filter(pl.col("drift_rating") == x)[metric].min()) for x in (YES, NO)]  # type: ignore[arg-type]
    )
    metric_max = min(
        [float(df.filter(pl.col("drift_rating") == x)[metric].max()) for x in (YES, NO)]  # type: ignore[arg-type]
    )
    if logspace:
        values = np.logspace(
            np.log10(metric_min), np.log10(metric_max), n_points, endpoint=True
        )
    else:
        values = np.linspace(metric_min, metric_max, n_points, endpoint=True)
    return pl.DataFrame(
        {
            "value": values,
            "tp": [
                df.filter(
                    (pl.col("drift_rating") == YES)
                    & (
                        pl.col(metric).le(v)
                        if below_threshold
                        else pl.col(metric).ge(v)
                    )
                ).height
                for v in values
            ],
            "fp": [
                df.filter(
                    (pl.col("drift_rating") == NO)
                    & (
                        pl.col(metric).le(v)
                        if below_threshold
                        else pl.col(metric).ge(v)
                    )
                ).height
                for v in values
            ],
            "tn": [
                df.filter(
                    (pl.col("drift_rating") == NO)
                    & (
                        pl.col(metric).gt(v)
                        if below_threshold
                        else pl.col(metric).lt(v)
                    )
                ).height
                for v in values
            ],
            "fn": [
                df.filter(
                    (pl.col("drift_rating") == YES)
                    & (
                        pl.col(metric).gt(v)
                        if below_threshold
                        else pl.col(metric).lt(v)
                    )
                ).height
                for v in values
            ],
        }
    ).with_columns(
        fpr=pl.col("fp") / (pl.col("fp") + pl.col("tn")),
        tpr=pl.col("tp") / (pl.col("tp") + pl.col("fn")),
        metric=pl.lit(metric),
    )


def fpr_to_classifier_info(fpr_threshold: float) -> dict:
    return (
        get_roc_curve_df(metric="lda")
        .select(pl.col("value").sort_by((pl.col("fpr") - fpr_threshold).abs()))
        .rename({"value": "lda_threshold"})
        .to_dicts()[0]
    )


sort_by_column_select = pn.widgets.Select(
    name="Sort by",
    options=["lda", "unit_id", "presence_ratio", "lda"],
    value="lda",
    width=BUTTON_WIDTH,
)
sort_by_column_select.param.watch(update_unit_generator, "value")
descending_checkbox = pn.widgets.Checkbox(name="Descending", value=False)
descending_checkbox.param.watch(update_unit_generator, "value")
shuffle_checkbox = pn.widgets.Checkbox(name="Shuffle", value=False)
shuffle_checkbox.param.watch(update_unit_generator, "value")


def app():
    sidebar = pn.Column(
        unit_metrics_pane,
        pn.layout.Divider(margin=(20, 0, 15, 0)),
        undo_button,
        skip_button,
        shortcuts_component,
        pn.layout.Divider(margin=(20, 0, 15, 0)),
        pn.pane.Markdown("### Filter units"),
        ks4_checkbox,
        pn.pane.Markdown("#### Classification"),
        classification_filter_radio,
        fpr_slider,
        pn.pane.Markdown("#### Annotation"),
        drift_rating_filter_radio,
        presence_ratio_range_slider,
        sort_by_column_select,
        descending_checkbox,
        shuffle_checkbox,
        pn.pane.Markdown("""### Paste text only"""),
        pn.pane.Markdown("Keyboard shortcuts active."),
        pn.pane.Markdown("Don't type in filter box!"),
        unit_id_filter_text,
        session_id_filter_text,
    )

    return pn.template.MaterialTemplate(
        site="DR dashboard",
        title="unit drift classification",
        sidebar=[sidebar],
        main=[raster_image_pane],
        sidebar_width=SIDEBAR_WIDTH,
    )


app().servable()
