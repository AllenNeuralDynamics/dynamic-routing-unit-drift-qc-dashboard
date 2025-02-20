import logging
import time
import polars as pl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


logger.info("Loading data from parquet files")
t0 = time.time()
df = (
    pl.scan_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/unit_drift.parquet")
    .select('unit_id', 'drift_rating')
    .with_columns(
        session_id=pl.col('unit_id').str.split('_').list.slice(0, 2).list.join('_'),
    )
    # get spike-count / time correlation values ------------------------ #
    .join(
        other=pl.scan_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/corr_values.parquet"),
        on='unit_id',
        how='inner',
    )
    # get spike-counts ------------------------------------------------- #
    # .join(
    #     other=(
    #         pl.scan_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/spike_counts.parquet")
    #         # .tail(1_000)
    #     ),
    #     on='unit_id',
    #     how='left',
    # )
    # get unit metrics from sorting ------------------------------------ #
    .join(
        other=(
            pl.scan_parquet("s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.261/consolidated/units.parquet")
            .select('unit_id', 'presence_ratio')
        ),
        on='unit_id',
        how='left',
    )
    # get anova metrics ------------------------------------------------ #
    # .join(
    #     other=(
    #         pl.scan_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/anova.parquet")
    #         .select('unit_id', 'anova_baseline_F_block', 'anova_baseline_F_context', 'anova_response_F_context')
    #     ),
    #     on='unit_id',
    #     how='left',
    # )
    # get ancova metrics ------------------------------------------------ #
    # .join(
    #     other=(
    #         pl.scan_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/ancova.parquet")
    #     ),
    #     on='unit_id',
    #     how='left',
    # )
    # get trial metadata ----------------------------------------------- #
    # .join(
    #     other=(
    #         pl.scan_parquet("s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.261/consolidated/trials.parquet")
    #         .select('session_id', 'trial_index', 'block_index', 'context_name', 'start_time')
    #     ),
    #     on=['session_id', 'trial_index',],
    #     how='inner',
    # )
    # aggregate spike counts by block ---------------------------------- #
    # .group_by(gb_cols := ['unit_id', 'block_index', 'context_name'])
    # .agg(
    #     pl.all().exclude('baseline', 'response'),
    #     pl.col('baseline').median(),
    #     pl.col('response').median(),
    # )
    # .explode(pl.all().exclude(*gb_cols, 'baseline', 'response'))
    # discard trial info ----------------------------------------------- #
    # .drop('trial_index')
    # .group_by('unit_id', 'block_index').agg(pl.all()).explode(pl.all().exclude('unit_id', 'block_index'))
    # ------------------------------------------------------------------ #
    .collect(streaming=False)
)
print(df.describe())
logger.info(f"Loaded data from parquet files in {time.time() - t0:.2f} seconds")

logger.info(f"Fitting LDA with leave-one-out cross-validation")
t0 = time.time()
METRICS_TO_KEEP = {'presence_ratio', 'vis_response_r2', 'aud_response_r2'} # , 'ancova_t_time', 'ancova_coef_time'
COLUMNS_TO_DROP = set(df.columns) - METRICS_TO_KEEP - {'unit_id', 'drift_rating'}

annotated = (
    df
    .unique('unit_id')
    .filter(
        pl.col('drift_rating') != 5,
    )
    .fill_nan(None)
    .drop_nulls(pl.all().exclude('drift_rating'))
    .filter(
        pl.col('drift_rating').is_not_null()
    )   
    .drop(COLUMNS_TO_DROP)
)

def get_x_y(df: pl.DataFrame) -> tuple:
    # z-score remaining metrics
    df = (
        df
        .with_columns(
            *[(col - col.mean()) / col.std() for col in (pl.col(name) for name in METRICS_TO_KEEP)]
        )
    )
    x = df.drop('unit_id', 'drift_rating', strict=False).to_numpy()
    y = df['drift_rating'].to_numpy()
    assert x.shape[0] == y.shape[0]
    return x, y

lda = LinearDiscriminantAnalysis()
unit_id_to_value = {}
for unit_id in annotated['unit_id']:
    train = annotated.filter(pl.col('unit_id') != unit_id)
    test = annotated.filter(pl.col('unit_id') == unit_id)
    lda.fit(*get_x_y(train))
    unit_id_to_value[unit_id] = lda.transform(get_x_y(test)[0]).item()

logger.info(f"Fitted LDA in {time.time() - t0:.2f} seconds")
lda_df = (
    annotated
    .join(pl.DataFrame({'unit_id': unit_id_to_value.keys(), 'lda': unit_id_to_value.values()}), on='unit_id', how='inner')  
)
logger.info(f"Writing LDA results for annotated units to parquet")
lda_df.write_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda.parquet")
print(lda_df.describe())

logger.info(f"Applying LDA to all units")
lda.fit(*get_x_y(annotated)) # re-fit on all annotated units
all_units = (
    df
    .unique('unit_id')
    .drop(COLUMNS_TO_DROP)
    .fill_nan(None)
    .drop_nulls(pl.all().exclude('drift_rating'))
)
scores = lda.transform(get_x_y(all_units)[0]).squeeze()
assert scores.ndim == 1
assert len(scores) == len(all_units)
lda_df = (
    all_units
    .with_columns(lda=pl.Series(scores))
)
lda_df.write_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda_all.parquet")

print(lda_df.describe())
print(dict(zip(train.drop('unit_id', 'drift_rating').columns, [float(x) for x in lda.coef_.squeeze()])))
