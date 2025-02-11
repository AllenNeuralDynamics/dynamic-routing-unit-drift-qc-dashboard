import datetime
import logging
from os import name
import time
from typing import Iterable, NotRequired, TypedDict

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

ROC_DF_PATH = '//allen/programs/mindscope/workgroups/dynamicrouting/ben/roc_df.parquet'

all_units_df = pl.read_parquet(db_utils.CACHED_DF_PATH.format(db_utils.CACHE_VERSION, "units"))
drop_score_df = pl.read_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/df_qc_drop_session_time.parquet")
corr_values_df = pl.read_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/corr_values.parquet")
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
    .join(
        other=drop_score_df.select('unit_id', 'time_weight', 'drop_score'),
        on='unit_id',
        how='left',
    )
    .join(
        other=(
            corr_values_df
            .rename(
                {
                    'vis_baseline_r2': 'r2_vis_baseline',
                    'vis_response_r2': 'r2_vis_response',
                    'aud_baseline_r2': 'r2_aud_baseline',
                    'aud_response_r2': 'r2_aud_response',
                }
            )
        ),
        on='unit_id',
        how='left',
    )
    .with_columns(
        time_weight=pl.col('time_weight').abs(),
    )
)

QUANTILE = 0.99
def filter_outlier_exprs() -> list[pl.Expr]:
    exprs = []
    for column in (
        'drift_ptp', 
        'amplitude_cutoff',
        'time_weight',
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
    value='r2_aud_response',
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
            opacity=0.8,
            binSpacing=0,
        )
        .encode(       
            alt.X('metric:Q', bin=alt.Bin(maxbins=50), title='metric value'),
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
        )
        .transform_calculate(
            metric=f'datum[{metric_param.name}]'
        )
        .transform_filter('isValid(datum.metric)') 
        .transform_window(
            cdf="cume_dist()",
            sort=[{"field": "metric"}],
            groupby=["drift_rating"],
        )
        .encode(
            alt.X('metric:Q', title='metric value'),
            alt.Y("cdf:Q"),
            alt.Color('drift_rating:N'),
            tooltip=[
                alt.Tooltip('metric:Q', title='metric value', format='.2f'),
                alt.Tooltip('cdf:Q', title='cdf', format='.2f'),
                alt.Tooltip('drift_rating:N', title='annotation'),
            ]
        )
        .mark_line(
            interpolate="step-after"
        )
        .add_params(
            metric_param,
            brush,
        )
    ).interactive()


def get_roc_curve(metric: str, n_points=50) -> pl.DataFrame:
    df = unfiltered_df
    YES = db_utils.UnitDriftRating.YES.name
    NO = db_utils.UnitDriftRating.NO.name
    # only use the subset of the data where the metric overlaps between the two drift ratings
    metric_min = max([df.filter(pl.col('drift_rating') == x)[metric].min() for x in (YES, NO)])
    metric_max = min([df.filter(pl.col('drift_rating') == x)[metric].max() for x in (YES, NO)])
    values = np.linspace(metric_min, metric_max, n_points, endpoint=True)
    return (
        pl.DataFrame(
            {
                'value': values,
                'tp': [df.filter((pl.col('drift_rating') == YES) & (pl.col(metric) >= v)).height for v in values],
                'fp': [df.filter((pl.col('drift_rating') == NO) & (pl.col(metric) >= v)).height for v in values],
                'tn': [df.filter((pl.col('drift_rating') == NO) & (pl.col(metric) < v)).height for v in values],
                'fn': [df.filter((pl.col('drift_rating') == YES) & (pl.col(metric) < v)).height for v in values],
            }
        )
        .with_columns(
            fpr=pl.col('fp') / (pl.col('fp') + pl.col('tn')),
            tpr=pl.col('tp') / (pl.col('tp') + pl.col('fn')),
            metric=pl.lit(metric),
        )
    )

def create_roc_df(roc_df_path: str = ROC_DF_PATH) -> pl.DataFrame:
    t0 = time.time()
    roc_df = pl.concat(
        get_roc_curve(metric) for metric in metric_columns
    )
    logger.info(f"roc_df created in {time.time() - t0:.2f}s")
    roc_df.write_parquet(roc_df_path)
    logger.info(f"roc_df written to {roc_df_path}")
    return roc_df

# create_roc_df()
roc_df = pl.read_parquet(ROC_DF_PATH)

def get_roc_chart() -> alt.Chart:
    return (
        alt.Chart(  
            data=roc_df.to_pandas(),
        )
        .transform_filter(
            alt.FieldEqualPredicate(metric_param, 'metric')
        )
        .encode(
            alt.X('fpr:Q', title='false positive rate'),
            alt.Y('tpr:Q', title='true positive rate'),
            tooltip=[
                alt.Tooltip(
                    'metric:N',
                ),
                alt.Tooltip(
                    'value:Q',
                    title='metric threshold',
                    format='.2f',
                ),
                alt.Tooltip(
                    'tpr:Q',
                    title='true positive rate',
                    format='.2f',
                ),
                alt.Tooltip(
                    'fpr:Q',
                    title='false positive rate',
                    format='.2f',
                ),
            ]
        )
        .mark_line(point={'fill': '#2ca02c'}, color='#2ca02c', tooltip=True)
        .add_params(
            metric_param,
        )
    )
  
roc_chart = get_roc_chart()  

outliers_df = unfiltered_df.join(filtered_df, on='unit_id', how='anti')  
metric_charts =  pn.pane.Vega(
    (
        (get_histogram_chart(filtered_df, title="excluding outliers") | get_cum_dist_chart(filtered_df))  | roc_chart
    ).configure_legend(
        title=None,
    )
    # & (get_histogram_chart(outliers_df, title=f'outliers (>{100*QUANTILE}th percentile of any metric)') | get_cum_dist_chart(outliers_df))                               
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
