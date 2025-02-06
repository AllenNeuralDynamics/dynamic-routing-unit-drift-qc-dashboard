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
def filter_outlier_exprs() -> list[pl.Expr]:
    exprs = []
    for column in (
        'drift_ptp', 
        'amplitude_cutoff',
    ):
        q: float | None = unfiltered_df[column].quantile(0.99, interpolation='lower')
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
    value='drift_ptp',
    bind=metric_dropdown
)

brush = alt.selection_interval(
    encodings=['x'],
    bind='scales',
)


def get_histogram_chart(df: pl.DataFrame, title: str | None = None) -> alt.Chart:
    # alt.Chart(df.to_pandas())
    return (
        alt.Chart(df.to_pandas())
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
        .properties(
            title=title or "",
        )
    ).interactive()

def get_cum_dist_chart(df) -> alt.Chart:
    return (
        alt.Chart(df.to_pandas())
        .transform_calculate(
            metric=f'datum[{metric_param.name}]'
        )
        .transform_window(
            cdf="cume_dist()",
            frame=[-5, 5],
            sort=[{"field": "metric"}],
        )
        .mark_line(
            interpolate="step-after"
        )
        .encode(
            alt.X('metric:Q', title=''),
            alt.Y("cdf:Q"),
            alt.Color('drift_rating:N'),
        )
        .add_params(
            metric_param,
            brush,
        )
    ).interactive()

outliers_df = unfiltered_df.join(filtered_df, on='unit_id', how='anti')  
metric_charts =  pn.pane.Vega((
    (get_histogram_chart(filtered_df) | get_cum_dist_chart(filtered_df)) )
    & (get_histogram_chart(outliers_df, title='outliers (in any metric)') | get_cum_dist_chart(outliers_df))
                                
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

def app():
    return pn.template.MaterialTemplate(
        site="DR dashboard",
        title="unit drift charts",
        main=[
            stats,
            pn.layout.Divider(margin=(20, 0, 15, 0)),
            metric_charts, 
        ],
    )
    
app().servable()
