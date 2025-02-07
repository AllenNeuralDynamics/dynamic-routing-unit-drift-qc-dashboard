import datetime
import logging
import time
from typing import NotRequired, TypedDict

import altair as alt
import panel as pn
import param
import polars as pl
import upath
from panel.custom import ReactComponent
import numpy as np
import polars.selectors as cs 

import db_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  #! doesn't take effect

all_units_df = pl.read_parquet(db_utils.CACHED_DF_PATH.format(db_utils.CACHE_VERSION, "units"))
unfiltered_df = (
    db_utils.get_df()
    .drop_nulls('drift_rating')
    # replace drift rating numbers with names 
    .filter(pl.col('drift_rating') != db_utils.UnitDriftRating.UNSURE)
    .with_columns(
        drift_rating=pl.when(pl.col('drift_rating') == db_utils.UnitDriftRating.YES)
        .then(pl.lit(db_utils.UnitDriftRating.YES.name))
        .otherwise(pl.lit(db_utils.UnitDriftRating.NO.name))
    )
    .join(
        other=all_units_df,
        on='unit_id',
        how='inner',
    )
)
QUANTILE = 0.99
def filter_outlier_exprs() -> list[pl.Expr]:
    exprs = []
    for column in (
        'drift_ptp', 
        'amplitude_cutoff',
    ):
        q: float | None = unfiltered_df[column].quantile(QUANTILE, interpolation='lower')
        exprs.append(
            (pl.col(column) <= q) | (pl.col(column).is_null()) | (pl.col(column).is_nan()))
    return exprs

filtered_df = unfiltered_df.filter(*filter_outlier_exprs())

metric_columns = sorted(
    set(unfiltered_df.select(cs.float()).columns) - set([
        'unit_id', 'session_id', 'drift_rating',
        'ccf_ap', 'ccf_ml', 'ccf_dv', 
        'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
    ])
)
metric_dropdown = alt.binding_select(
    options=metric_columns,
    name='metric'
)
metric_param = alt.param(
    value=metric_columns[0],
    bind=metric_dropdown
)

brush = alt.selection_interval(
    encodings=['x'],
    bind='scales',
)


def get_histogram_chart(df: pl.DataFrame, title: str | None = None) -> alt.Chart:
    # alt.Chart(df.to_pandas())
    return (
        alt.Chart(
            data=df.to_pandas(),
            title=alt.Title(
                title or "",
                anchor='start',
                orient='bottom',
            )
        )
        .transform_calculate(
            metric=f'datum[{metric_param.name}]'
        )
        # .transform_filter(
        #     alt.FieldLTEPredicate('metric', alt.expr.quantileUniform(0.9))
        # )
        .mark_bar(
            opacity=0.5,
            binSpacing=0,
        )
        .encode(       
            alt.X('metric:Q', bin=alt.Bin(maxbins=100), title=''),
            alt.Y('count()', title='count'),
            alt.Color('drift_rating:N'),
        ) 
        .add_params(
            metric_param,
            brush,
        )
    ).interactive()

def get_cum_dist_chart(df: pl.DataFrame) -> alt.Chart:
    return (
        alt.Chart(
            data=df.to_pandas(),
            title=alt.Title(
                '(nan values may affect starting point)',
                anchor='start',
                orient='bottom',
            )
        )
        .transform_calculate(
            metric=f'datum[{metric_param.name}]'
        )
        .transform_window(
            cdf="cume_dist()",
            frame=[-5, 5],
            sort=[{"field": "metric"}],
        )
        .encode(
            alt.X('metric:Q', title=''),
            alt.Y("cdf:Q"),
            alt.Color('drift_rating:N'),
            tooltip=['drift_rating:N', 'metric:Q', 'cdf:Q'],
        )
        .mark_line(
            interpolate="step-after"
        )
        .add_params(
            metric_param,
            brush,
        )
    ).interactive()

outliers_df = unfiltered_df.join(filtered_df, on='unit_id', how='anti')  
metric_charts =  pn.pane.Vega((
    (get_histogram_chart(filtered_df, title="excluding outliers") | get_cum_dist_chart(filtered_df)) )
    & (get_histogram_chart(outliers_df, title=f'outliers (>{100*QUANTILE}th percentile of any metric)') | get_cum_dist_chart(outliers_df))                               
)

df = db_utils.get_df()
stats = pn.pane.DataFrame(
    pl.DataFrame(
        {
            'unannotated': [df.filter(pl.col('drift_rating').is_null()).height],
            'unsure': [df.filter(pl.col('drift_rating') == db_utils.UnitDriftRating.UNSURE).height],
            'no': [df.filter(pl.col('drift_rating') == db_utils.UnitDriftRating.NO).height],
            'yes': [df.filter(pl.col('drift_rating') == db_utils.UnitDriftRating.YES).height],
            'outliers': [len(unfiltered_df) - len(filtered_df)],
        }
    ).to_pandas(),
    index=False,
)

timeline = pn.pane.Vega(
    alt.Chart(
        data=(
            db_utils.get_df()
            .drop_nulls('drift_rating')
            .with_columns(
                pl.from_epoch('checked_timestamp').alias('timestamp'),
            )
            .drop('session_id', 'drift_rating', 'checked_timestamp')
            .sort('timestamp')
            .group_by_dynamic(
                'timestamp',
                every="1h",
            )
            .agg(pl.col('unit_id').count().alias('annotations'))
        ).to_pandas(),
    )
    .mark_rect()
    .encode(
        alt.X('hoursminutes(timestamp):O').title('hour'),
        alt.Y('monthdate(timestamp):O').title('date'),
        alt.Color('annotations:Q', legend=None).scale(scheme="greens"),
        tooltip=['annotations:Q'],
    )
    .properties(
        title='annotations per hour',
    )
    .interactive()
)

def app():
    return pn.template.MaterialTemplate(
        site="DR dashboard",
        title="unit drift charts",
        main=[
            pn.layout.Divider(margin=(20, 0, 15, 0)),
            metric_charts, 
            pn.layout.Divider(margin=(20, 0, 15, 0)),
            pn.Row(stats, timeline),
        ],
    )
    
app().servable()
