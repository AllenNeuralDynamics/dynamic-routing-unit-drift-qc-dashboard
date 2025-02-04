import logging
import random
import time
import panel as pn
import polars as pl
import itertools
import upath
import sqlite3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CACHE_VERSION = 'v0.0.261'
SCRATCH_DIR = upath.UPath(f's3://aind-scratch-data/dynamic-routing')
RASTER_DIR = SCRATCH_DIR / f'unit-rasters/{CACHE_VERSION}/fig3c'

TABLE_NAME = 'from_app'
DB_PATH = 'unit_drift.parquet'

units_df = (
    pl.scan_parquet(f'{SCRATCH_DIR}/cache/nwb_components/{CACHE_VERSION}/consolidated/units.parquet')
    .select([
        'unit_id',
        'presence_ratio',
        'default_qc',
        'drift_ptp',
    ])
).collect()

def create_db(save_path: str = DB_PATH, overwrite: bool = False, with_paths=True) -> None:
    if upath.UPath(save_path).exists() and not overwrite:
        raise FileExistsError(f'{save_path} already exists: set overwrite=True to overwrite')
    df = (
        units_df
        .select('unit_id')
        .with_columns(
            session_id=pl.col('unit_id').str.split('_').list.slice(0, 2).list.join('_'),
            drift_rating=pl.lit(None),
            checked_timestamp=pl.lit(None),
        )
    )
    if with_paths:
        logger.warning('getting paths from S3 for all available units')
        paths = tuple(RASTER_DIR.rglob('*'))
        paths_df = pl.DataFrame({
            'unit_id': [p.stem.split('fig3c_')[1] for p in paths if 'fig3c_' in str(p)],
            'path': [str(p) for p in paths if 'fig3c_' in str(p)],
        })
        df = df.join(paths_df, on='unit_id', how='left')
    else:
        df = df.with_columns(path=pl.lit(None))
    df.write_parquet(save_path)
    return
    conn = sqlite3.connect(save_path)
    df.write_database(
        table_name=TABLE_NAME, 
        connection=f'sqlite:///{save_path}', 
        if_table_exists='replace' if overwrite else 'fail',
        engine='adbc',
    )
    conn.close()
    
def get_df(
    already_checked: bool | None = False,
    unit_id_filter: str | None = None,
    drift_rating_filter: int | None = None,
    with_paths: bool | None = True,
    db_path=DB_PATH,    
) -> pl.DataFrame:
    filter_exprs = []
    if with_paths is True:
        filter_exprs.append(pl.col('path').is_not_null())
    else:
        filter_exprs.append(pl.col('path').is_null())
    if already_checked:
        filter_exprs.append(pl.col('drift_rating').is_not_null())
    elif already_checked is False:
        filter_exprs.append(pl.col('drift_rating').is_null())
    if unit_id_filter:
        filter_exprs.append(pl.col('unit_id').str.starts_with(unit_id_filter))
    if drift_rating_filter is not None:
        filter_exprs.append(pl.col('drift_rating') == drift_rating_filter)
    if filter_exprs:
        logger.debug(f"Filtering units df with {' & '.join([str(f) for f in filter_exprs])}")
    return pl.read_parquet(db_path).filter(*filter_exprs)
    return pl.read_database(
            query=f'SELECT * FROM {TABLE_NAME}', 
            connection=sqlite3.connect(db_path),
        ).filter(*filter_exprs)

def unit_generator(
    **get_df_kwargs
):
    while True:
        df = get_df(**get_df_kwargs)
        session_ids = df['session_id'].unique().to_list()
        if not session_ids:
            raise StopIteration('No more sessions to check')
        random.shuffle(session_ids)
        for session_id in session_ids:
            sub_df = df.filter(pl.col('session_id') == session_id)
            if sub_df.is_empty():
                logger.info(f'No more units to check for {session_id}')
                continue
            yield sub_df.sample(1)['unit_id'].first()

def update_row(unit_id: str, drift_rating: int, db_path=DB_PATH) -> None:
    timestamp = int(time.time())
    session_id = '_'.join(unit_id.split('_')[:2])
    original_df = get_df()
    unit_id_filter = pl.col('unit_id') == unit_id
    logger.info(f'Updating row for {unit_id} with drift_rating={drift_rating}')
    df = (
        original_df
        .with_columns(
            drift_rating=pl.when(unit_id_filter).then(pl.lit(drift_rating)).otherwise(pl.col('drift_rating')),
            checked_timestamp=pl.when(unit_id_filter).then(pl.lit(timestamp)).otherwise(pl.col('checked_timestamp')),
            session_id=pl.when(unit_id_filter).then(pl.lit(session_id)).otherwise(pl.col('session_id')),
        )
    )
    assert len(df) == len(original_df), f'Row count changed: {len(original_df)} -> {len(df)}'
    df.write_parquet(db_path)
    logger.info(f'Overwrote {db_path}')

def test_db(with_paths=False):
    db_path = 'test.parquet'
    create_db(overwrite=True, save_path=db_path, with_paths=with_paths)
    i = next(unit_generator(db_path=db_path))
    update_row(i, 2, db_path=db_path)
    df = get_df(unit_id_filter=i, already_checked=True, db_path=db_path)
    assert len(df) == 1, f'Expected 1 row, got {len(df)}'
    upath.UPath(db_path).unlink()
    
def get_raster(unit_id: str) -> bytes:
    path: str = get_df(unit_id_filter=unit_id)['path'].first()
    logger.debug(f'Getting raster image data from {path}')
    t0 = time.time()
    b = upath.UPath(path).read_bytes()
    logger.debug(f'Got raster image data in {time.time() - t0:.2f}s')
    return b 

def get_metrics(unit_id: str):
    return units_df.filter(pl.col('unit_id') == unit_id).to_dicts()[0]

def display_metrics(
    unit_id=str,
) -> pn.pane.Markdown:
    metrics = get_metrics(unit_id)
    stats = f"""

### `{unit_id}`
"""
    for k, v in metrics.items():
        if k not in ('presence_ratio', 'default_qc', 'drift_ptp'):
            continue
        stats += f"\n{k.replace('_', ' ')}:\n`{v if v else '-'}`\n"
    return pn.pane.Markdown(stats)

def display_image(unit_id: str):
    return pn.pane.PNG(get_raster(unit_id), sizing_mode="stretch_height",)

unit_id_generator = unit_generator()
current_unit_id = next(unit_id_generator)
unit_id_pane = display_metrics(current_unit_id)
raster_image_pane = display_image(current_unit_id)

def update_unit_id(event):
    global current_unit_id
    current_unit_id = next(unit_id_generator)
    raster_image_pane.object = display_image(current_unit_id).object
    unit_id_pane.object = display_metrics(current_unit_id).object
    
button = pn.widgets.Button(name='Click to get new unit', width=190)
button.on_click(update_unit_id)


# # JavaScript to trigger button click on spacebar press
# def trigger_spacebar_click():
#     js_code = """
#     document.addEventListener('keydown', function(event) {
#         if (event.code === 'Space') {
#             document.querySelector('button').click();
#         }
#     });
#     """
#     pn.state.execute(lambda: pn.state.curdoc().add_root(pn.pane.HTML(f'<script>{js_code}</script>')))

# pn.state.onload(trigger_spacebar_click)

# pn.extension()
# pn.Column(button, unit_id_pane, raster_image_pane).servable()

def app():
    width = 150
    # unit_id = next(unit_generator())
#     subject_id = pn.widgets.TextInput(name="Subject ID(s)", value="", placeholder="comma separated", width=width)
#     specific_date = pn.widgets.TextInput(name="Specific date", value="", width=width)
#     start_date = pn.widgets.TextInput(name="Start date", value="", width=width)
#     end_date = pn.widgets.TextInput(name="End date", value="", width=width)
#     usage_info = pn.pane.Alert(
#         """
#         Press spacebar to get a new unit to check.
#         Hit 0, 1 or 2 to rate if the unit drifts: 0: no, 1: yes, 2: unsure.
#         """,
#         alert_type='info',
#         sizing_mode="stretch_width",
#     )
#     refresh_table_button = pn.widgets.Button(name="Refresh table", button_type="primary")
    sidebar = pn.Column(
        button, unit_id_pane,
#         subject_id,
#         specific_date,
#         start_date,
#         end_date,
#         refresh_table_button,
    )
#     # add on click event to refresh table
#     def refresh_table(event):
#         v = subject_id.value
#         subject_id.value = ''
#         subject_id.value = v
#     refresh_table_button.on_click(refresh_table)
        
#     bound_get_session_df = pn.bind(get_sessions_table, subject_id, specific_date, start_date, end_date)
    return pn.template.MaterialTemplate(
        site="DR dashboard",
        title='unit-raster-qc',
        sidebar=[sidebar],
        main=[raster_image_pane],
        sidebar_width=220,
    )

app().servable()