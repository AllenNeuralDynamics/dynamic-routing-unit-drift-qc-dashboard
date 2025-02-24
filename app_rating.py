import datetime
import logging
import time
from typing import NotRequired, TypedDict

import panel as pn
import param
import polars as pl
import upath
from panel.custom import ReactComponent

import db_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  #! doesn't take effect

# pn.config.theme = 'dark'
pn.config.notifications = True


SIDEBAR_WIDTH = 230
BUTTON_WIDTH = int(SIDEBAR_WIDTH * 0.8)

if not upath.UPath(db_utils.DB_PATH).exists():
    db_utils.create_db()

units_df: None | pl.DataFrame = db_utils.get_units_df()


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


def get_raster(unit_id: str) -> bytes:
    filtered_df = db_utils.get_df(unit_id_filter=unit_id, with_paths=True)
    assert len(filtered_df) == 1, "df filtering likely incorrect"
    path: str = filtered_df["path"].first()
    assert path is not None, f"Path not stored for {unit_id}"
    logger.info(f"Getting raster image data from {path}")
    t0 = time.time()
    image_bytes = upath.UPath(path).read_bytes()
    logger.info(f"Got raster image data in {time.time() - t0:.2f}s")
    return image_bytes


def display_rating(
    unit_id=str,
) -> pn.pane.Markdown:
    df = db_utils.get_df(unit_id_filter=unit_id)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    row = df.to_dicts()[0]
    rating: int | None = row["drift_rating"]
    if rating is None:
        return pn.pane.Markdown("not yet rated")
    else:
        return pn.pane.Markdown(
            f"**{db_utils.UnitDriftRating(rating).name.title()}** ({datetime.datetime.fromtimestamp(row['checked_timestamp']).strftime('%Y-%m-%d %H:%M:%S')})"
        )


def display_image(unit_id: str):
    return pn.pane.PNG(
        get_raster(unit_id),
        sizing_mode="stretch_height",
    )

# initialize for startup, then update as a global variable
unit_id_generator = db_utils.unit_generator(
    already_checked=False,
    with_paths=True,
)
current_unit_id = next(unit_id_generator)
previous_unit_id: str | None = None
unit_metrics_pane = display_metrics(current_unit_id)
unit_rating_pane = display_rating(current_unit_id)
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
    unit_rating_pane.object = display_rating(current_unit_id).object


def update_and_next(unit_id: str, drift_rating: int) -> None:
    db_utils.set_drift_rating_for_unit(unit_id=unit_id, drift_rating=drift_rating)
    next_unit()


# make three buttons for rating drift, which all call update_unit_id
# 0: no, 1: yes, 2: unsure
no_button_text = (
    f"{db_utils.UnitDriftRating.NO.name.title()} [{db_utils.UnitDriftRating.NO.value}]"
)
no_button = pn.widgets.Button(name=no_button_text, width=BUTTON_WIDTH)
no_button.on_click(
    lambda event: update_and_next(
        unit_id=current_unit_id, drift_rating=db_utils.UnitDriftRating.NO
    )
)
yes_button_text = f"{db_utils.UnitDriftRating.YES.name.title()} [{db_utils.UnitDriftRating.YES.value}]"
yes_button = pn.widgets.Button(name=yes_button_text, width=BUTTON_WIDTH)
yes_button.on_click(
    lambda event: update_and_next(
        unit_id=current_unit_id, drift_rating=db_utils.UnitDriftRating.YES
    )
)
unsure_button_text = f"{db_utils.UnitDriftRating.UNSURE.name.title()} [{db_utils.UnitDriftRating.UNSURE.value}]"
unsure_button = pn.widgets.Button(name=unsure_button_text, width=BUTTON_WIDTH)
unsure_button.on_click(
    lambda event: update_and_next(
        unit_id=current_unit_id, drift_rating=db_utils.UnitDriftRating.UNSURE
    )
)
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
] + [
    KeyboardShortcut(name=k, key=str(v), ctrlKey=False)
    for k, v in db_utils.UnitDriftRating.__members__.items()
]
shortcuts_component = KeyboardShortcuts(shortcuts=shortcuts)


def handle_shortcut(event):
    if event.data == "skip":
        next_unit()
    elif event.data == "previous":
        next_unit(override_next_unit_id=previous_unit_id)
    else:
        update_and_next(
            unit_id=current_unit_id, drift_rating=db_utils.UnitDriftRating[event.data]
        )


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
    unit_id_generator = db_utils.unit_generator(**unit_generator_params)
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
    options=["all", "unrated"] + [k.lower() for k in db_utils.UnitDriftRating.__members__],
    inline=False,
    value="unrated",
)
drift_rating_filter_radio.param.watch(update_unit_generator, "value")
presence_ratio_range_slider = pn.widgets.RangeSlider(
    name="Presence ratio",
    start=0,
    end=1,
    value=(0.7, 1),
    step=0.05,
    width=BUTTON_WIDTH,
)
presence_ratio_range_slider.param.watch(update_unit_generator, "value")

ks4_checkbox = pn.widgets.Checkbox(name="Kilosort 4", value=False)
ks4_checkbox.param.watch(update_unit_generator, "value")

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
        ks4_checkbox,
        drift_rating_filter_radio,
        presence_ratio_range_slider,
        pn.pane.Markdown("""### Paste text only"""),
        pn.pane.Markdown("Keyboard shortcuts active."),
        pn.pane.Markdown("Don't type in filter box!"),
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
