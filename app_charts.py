import functools
import logging
import time

import altair as alt
import numpy as np
import panel as pn
import polars as pl
import polars.selectors as cs

import db_utils

# alt.data_transformers.enable("vegafusion")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  #! doesn't take effect

ROC_DF_PATH = "//allen/programs/mindscope/workgroups/dynamicrouting/ben/roc_df.parquet"

all_units_df = pl.read_parquet(
    db_utils.CACHED_DF_PATH.format(db_utils.CACHE_VERSION, "units")
)
drop_score_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/df_qc_drop_session_time.parquet"
)
corr_values_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/corr_values.parquet"
)
lda_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda.parquet"
)
lda_all_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda_all.parquet"
)
lda_all_ks4_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda_ks4.parquet"
)
anova_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/anova.parquet"
)
ancova_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/ancova.parquet"
)
ks_test_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/ks_test.parquet"
)
kw_test_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/kw_test.parquet"
)
med_test_df = pl.read_parquet(
    "//allen/programs/mindscope/workgroups/dynamicrouting/ben/med_test.parquet"
)

unfiltered_df = (
    db_utils.get_df(ks4_filter=False)
    .drop_nulls("drift_rating")
    # replace drift rating numbers with names
    .filter(pl.col("drift_rating") != db_utils.UnitDriftRating.UNSURE)
    .with_columns(
        drift_rating=pl.when(pl.col("drift_rating") == db_utils.UnitDriftRating.YES)
        .then(pl.lit(db_utils.UnitDriftRating.YES.name))
        .otherwise(pl.lit(db_utils.UnitDriftRating.NO.name))
    )
    .join(
        other=all_units_df,
        on="unit_id",
        how="inner",
    )
    .join(
        other=drop_score_df.select("unit_id", "time_weight", "drop_score"),
        on="unit_id",
        how="left",
    )
    .with_columns(
        time_weight=pl.col("time_weight").abs(),
    )
    .join(
        other=(
            corr_values_df.rename(
                {
                    "vis_baseline_r2": "r2_vis_baseline",
                    "vis_response_r2": "r2_vis_response",
                    "aud_baseline_r2": "r2_aud_baseline",
                    "aud_response_r2": "r2_aud_response",
                }
            )
        ),
        on="unit_id",
        how="left",
    )
    .join(
        other=lda_df.drop_nulls().drop_nans(),
        on="unit_id",
        how="left",
    )
    .join(
        other=anova_df.drop_nulls().drop_nans(),
        on="unit_id",
        how="left",
    )
    .join(
        other=ancova_df.drop_nulls().drop_nans(),
        on="unit_id",
        how="left",
    )
    .join(
        other=ks_test_df.drop_nulls().drop_nans(),
        on="unit_id",
        how="left",
    )
    .join(
        other=kw_test_df.drop_nulls().drop_nans(),
        on="unit_id",
        how="left",
    )
    .join(
        other=med_test_df.drop_nulls().drop_nans(),
        on="unit_id",
        how="left",
    )
)

QUANTILE = 0.99


def filter_outlier_exprs() -> list[pl.Expr]:
    exprs = []
    for column in (
        "drift_ptp",
        "amplitude_cutoff",
        "time_weight",
    ):
        q: float | None = unfiltered_df[column].quantile(
            QUANTILE, interpolation="lower"
        )
        exprs.append(
            (pl.col(column) <= q)
            | (pl.col(column).is_null())
            | (pl.col(column).is_nan())
        )
    return exprs


filtered_df = unfiltered_df.filter(*filter_outlier_exprs())

    
metric_columns = sorted(
    set(unfiltered_df.select(cs.float()).columns)
    - set(
        [
            "unit_id",
            "session_id",
            "drift_rating",
            "ccf_ap",
            "ccf_ml",
            "ccf_dv",
            "sync_spike_2",
            "sync_spike_4",
            "sync_spike_8",
        ]
    )
)
metric_dropdown = alt.binding_select(options=metric_columns, name="metric")
metric_param = alt.param(value="r2_aud_response", bind=metric_dropdown)

brush = alt.selection_interval(
    encodings=["x"],
    bind="scales",
)


def get_histogram_chart(
    df: pl.DataFrame, metric: str, title: str | None = None
) -> alt.Chart:
    # alt.Chart(df.to_pandas())
    return (
        alt.Chart(
            data=(
                df
                .select(metric, "drift_rating")
                .sample(min(5000, len(df) - 1)) # vega limit
                .to_pandas()
            ),
            title=alt.Title(
                title or "",
                anchor="start",
                orient="bottom",
            ),
        )
        # .transform_calculate(
        #     metric=f'datum[{metric_param.name}]'
        # )
        # .transform_filter(
        #     alt.FieldLTEPredicate('metric', alt.expr.quantileUniform(0.9))
        # )
        .mark_bar(
            opacity=0.8,
            binSpacing=0,
        )
        .encode(
            alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=50), title=f"{metric} value"),
            alt.Y("count()", title="count"),
            alt.Color("drift_rating:N", title="annotation"),
        )
        .add_params(
            # metric_param,
            brush,
        )
    ).interactive()


def get_cum_dist_chart(df: pl.DataFrame, metric: str) -> alt.Chart:
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[metric], empty=False)
    source = (
        df.select(metric, "drift_rating")
        .drop_nulls(metric)
        .sample(min(5000, len(df.drop_nulls(metric)))) # vega limit
        .sort(metric)
        .with_columns(
            cdf=(pl.col(metric).cum_count() / pl.col(metric).count()).over('drift_rating'),
        )
    )
    chart = (
        alt.Chart(
            data=source.to_pandas(),
        )
        .encode(
            alt.X(f"{metric}:Q", title=f"{metric} value"),
            alt.Y("cdf:Q"),
            alt.Color("drift_rating:N", title="annotation", legend=None),
            tooltip=[
                alt.Tooltip("drift_rating:N"),
                alt.Tooltip(f"{metric}:Q", title=f"{metric} value", format=".2f"),
                alt.Tooltip("cdf:Q", title="cdf", format=".2f"),
            ],
        )
        .mark_line(interpolate="step-after")
        .add_params(
            # metric_param,
            brush,
        )
    ).interactive()
    when_near = alt.when(nearest)
    points = chart.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0))
    )
    # Draw a rule at the location of the selection
    rules = (
        alt.Chart(source.to_pandas())
        .transform_pivot(
            'drift_rating',
            value="cdf",
            groupby=[metric],
        )
        .mark_rule(color="gray")
        .encode(
            x=f"{metric}:Q",
            opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
            tooltip=[alt.Tooltip(c, type="quantitative", format=".2f") for c in ['NO', 'YES']],
        )
        .add_params(nearest)
    )
    return chart #+ points  + rules


def get_roc_curve_df(metric: str, n_points=50) -> pl.DataFrame:
    df = unfiltered_df
    YES = db_utils.UnitDriftRating.YES.name
    NO = db_utils.UnitDriftRating.NO.name
    # only use the subset of the data where the metric overlaps between the two drift ratings
    metric_min = max(
        [float(df.filter(pl.col("drift_rating") == x)[metric].min()) for x in (YES, NO)] # type: ignore[arg-type]
    )
    metric_max = min(
        [float(df.filter(pl.col("drift_rating") == x)[metric].max()) for x in (YES, NO)] # type: ignore[arg-type]
    )
    values = np.linspace(metric_min, metric_max, n_points, endpoint=True)
    return pl.DataFrame(
        {
            "value": values,
            "tp": [
                df.filter(
                    (pl.col("drift_rating") == YES) & (pl.col(metric) >= v)
                ).height
                for v in values
            ],
            "fp": [
                df.filter((pl.col("drift_rating") == NO) & (pl.col(metric) >= v)).height
                for v in values
            ],
            "tn": [
                df.filter((pl.col("drift_rating") == NO) & (pl.col(metric) < v)).height
                for v in values
            ],
            "fn": [
                df.filter((pl.col("drift_rating") == YES) & (pl.col(metric) < v)).height
                for v in values
            ],
        }
    ).with_columns(
        fpr=pl.col("fp") / (pl.col("fp") + pl.col("tn")),
        tpr=pl.col("tp") / (pl.col("tp") + pl.col("fn")),
        metric=pl.lit(metric),
    )

def get_area_under_roc_curve(roc_df: pl.DataFrame) -> float:
    roc_df = roc_df.sort("fpr")
    fpr = roc_df["fpr"].to_numpy()
    tpr = roc_df["tpr"].to_numpy()

    # Add (0, 0) if not present
    if fpr[0] != 0.0:
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)

    # Add (1, 1) if not present
    if fpr[-1] != 1.0:
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    auc = float(np.trapz(y=tpr, x=fpr)) # explicit casting for mypy

    return auc

def get_roc_chart(roc_df: pl.DataFrame) -> alt.Chart:
    return (
        alt.Chart(
            data=roc_df.to_pandas(),
            title=f'AUC: {get_area_under_roc_curve(roc_df):.3f}',
        )
        # .transform_filter(
        #     alt.FieldEqualPredicate(metric_param, 'metric')
        # )
        .encode(
            alt.X("fpr:Q", title="false positive rate").scale(alt.Scale(domain=(0, 1))),
            alt.Y("tpr:Q", title="true positive rate").scale(alt.Scale(domain=(0, 1))),
            tooltip=[
                alt.Tooltip(
                    "metric:N",
                ),
                alt.Tooltip(
                    "value:Q",
                    title="metric threshold",
                    format=".2f",
                ),
                alt.Tooltip(
                    "tpr:Q",
                    title="true positive rate",
                    format=".2f",
                ),
                alt.Tooltip(
                    "fpr:Q",
                    title="false positive rate",
                    format=".2f",
                ),
            ],
        ).mark_line(point={"fill": "#2ca02c"}, color="#2ca02c", tooltip=True)
        # .add_params(
        #     metric_param,
        # )
    )


metrics_dropdown_pn = pn.widgets.Select(
    name="metric", options=metric_columns, value="lda"
)

outliers_df = unfiltered_df.join(filtered_df, on="unit_id", how="anti")

stats = pn.pane.DataFrame(
    db_utils.get_df()
    .with_columns(sorter=pl.when(pl.col('unit_id').str.ends_with('_ks4')).then(pl.lit('ks4')).otherwise(pl.lit('ks2.5')))
    .group_by('sorter')
    .agg(
        unannotated=pl.col('drift_rating').is_null().sum(),
        unsure=(pl.col('drift_rating') == db_utils.UnitDriftRating.UNSURE).sum(),
        no=(pl.col('drift_rating') == db_utils.UnitDriftRating.NO).sum(),
        yes=(pl.col('drift_rating') == db_utils.UnitDriftRating.YES).sum(),
    ).to_pandas(),
    index=False,
)

timeline = pn.pane.Vega(
    alt.Chart(
        data=(
            db_utils.get_df()
            .drop_nulls("drift_rating")
            .with_columns(
                pl.from_epoch("checked_timestamp").alias("timestamp"),
            )
            .drop("session_id", "drift_rating", "checked_timestamp")
            .sort("timestamp")
            .group_by_dynamic(
                "timestamp",
                every="1h",
            )
            .agg(pl.col("unit_id").count().alias("annotations"))
        ).to_pandas(),
    )
    .mark_rect()
    .encode(
        alt.X("hoursminutes(timestamp):O").title("hour"),
        alt.Y("monthdate(timestamp):O").title("date"),
        alt.Color("annotations:Q", legend=None).scale(scheme="greens"),
        tooltip=["annotations:Q"],
    )
    .properties(
        title="annotations per hour",
    )
    .interactive()
)


# roc_chart = get_roc_chart(pl.read_parquet(ROC_DF_PATH))
def get_roc_chart_pn(metric: str) -> alt.Chart:
    roc_df = get_roc_curve_df(metric)
    return get_roc_chart(roc_df)


def app():
    return pn.template.MaterialTemplate(
        site="DR dashboard",
        title="unit drift charts",
        main=[
            pn.layout.Divider(margin=(20, 0, 15, 0)),
            pn.Row(
                pn.bind(
                    functools.partial(get_histogram_chart, filtered_df),
                    metric=metrics_dropdown_pn,
                ),
                pn.bind(
                    functools.partial(get_cum_dist_chart, filtered_df),
                    metric=metrics_dropdown_pn,
                ),
                pn.bind(get_roc_chart_pn, metric=metrics_dropdown_pn),
            ),
            # metric_charts,
            metrics_dropdown_pn,
            pn.layout.Divider(margin=(20, 0, 15, 0)),
            pn.Row(stats, timeline),
        ],
    )


app().servable()
