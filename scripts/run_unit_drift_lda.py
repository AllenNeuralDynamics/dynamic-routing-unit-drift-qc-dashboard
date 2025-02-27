import functools
import logging
import pathlib
import time
from typing import Iterable

import numpy.typing as npt
import polars as pl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tqdm

logger = logging.getLogger(pathlib.Path(__file__).stem)
logging.basicConfig(level=logging.INFO)

@functools.cache
def load_annotation_and_metrics_df() -> pl.DataFrame:
    logger.info("Loading data from parquet files")
    t0 = time.time()
    df = (
        pl.scan_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/unit_drift.parquet")
        .select('unit_id', 'drift_rating')
        .filter(pl.col('unit_id').str.ends_with('_ks4').not_())
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
    return df

def z_score(df: pl.DataFrame, columns: Iterable[str] | None = None) -> pl.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    return df.with_columns(
        *[(col - col.mean()) / col.std() for col in (pl.col(name) for name in (columns or set(df.columns) - {'unit_id', 'drift_rating'}))]
    )
    
def get_x(df: pl.DataFrame) -> npt.NDArray:
    return df.drop('unit_id', 'drift_rating', strict=False).to_numpy()

def get_x_y(df: pl.DataFrame) -> tuple[npt.NDArray, npt.NDArray]:
    # z-score metrics
    df = df.pipe(z_score).drop_nans()
    x = get_x(df)
    y = df['drift_rating'].to_numpy()
    assert x.shape[0] == y.shape[0]
    return x, y

def get_annotations_df() -> pl.DataFrame:
    return (
        load_annotation_and_metrics_df()
        .unique('unit_id')
        .filter(
            pl.col('drift_rating') != 5,
        )
        .fill_nan(None)
        .drop_nulls(pl.all().exclude('drift_rating'))
        .filter(
            pl.col('drift_rating').is_not_null()
        )   
    )
    
def main():
    df = load_annotation_and_metrics_df()
    t0 = time.time()

    METRICS_TO_KEEP = {'presence_ratio', 'vis_response_r2', 'aud_response_r2'} #, 'vis_baseline_r2', 'aud_baseline_r2'} # , 'ancova_t_time', 'ancova_coef_time'
    COLUMNS_TO_DROP = set(df.columns) - METRICS_TO_KEEP - {'unit_id', 'drift_rating'}

    annotated = (
        get_annotations_df()
        .drop(COLUMNS_TO_DROP)
    )

    lda = LinearDiscriminantAnalysis()
    unit_id_to_value = {}

    for unit_id in tqdm.tqdm(annotated['unit_id'], total=len(annotated), unit='units', ncols=100, desc="Fitting LDA with leave-one-out cross-validation"):
        train = annotated.filter(pl.col('unit_id') != unit_id)
        test = annotated.filter(pl.col('unit_id') == unit_id)
        lda.fit(*get_x_y(train))
        unit_id_to_value[unit_id] = lda.transform(get_x(test)).item()

    logger.info(f"Fitted LDA in {time.time() - t0:.2f} seconds")
    lda_df = (
        annotated
        .join(pl.DataFrame({'unit_id': unit_id_to_value.keys(), 'lda': unit_id_to_value.values()}), on='unit_id', how='inner')  
    )
    logger.info(f"Writing LDA results for annotated units to parquet")
    lda_df.write_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda.parquet")
    print(lda_df.describe())

    logger.info("Applying LDA score to un-annotated units")
    lda.fit(*get_x_y(annotated)) # re-fit on all annotated units
    print(dict(zip(train.drop('unit_id', 'drift_rating').columns, [float(x) for x in lda.coef_.squeeze()])))
    all_units = (
        df
        .unique('unit_id')
        .drop(COLUMNS_TO_DROP)
        .fill_nan(None)
        .drop_nulls(pl.all().exclude('drift_rating'))
    )
    ks4_units = (
        pl.read_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/corr_values_ks4.parquet")
        .with_columns(
            unit_id=pl.col('unit_id').str.replace('_ks4_', '_'),
        )
        .join(
            other=(
                pl.scan_parquet("s3://aind-scratch-data/dynamic-routing/unit-rasters-ks4/units.parquet")
                .select('unit_id', 'presence_ratio') 
                .collect()
            ),
            on='unit_id',
            how='left',
        )
        .fill_nan(None)
        .drop(COLUMNS_TO_DROP, strict=False)
        .drop_nulls()
    )
    for filename_suffix, units_df in {'all': all_units, 'ks4': ks4_units}.items():
        if len(units_df) == 0:
            logger.warning(f"No {filename_suffix} units found")
            continue
        if METRICS_TO_KEEP - set(units_df.columns):
            logger.warning(f"Missing metrics for {filename_suffix} units: {METRICS_TO_KEEP - set(units_df.columns)}")
            continue
        logger.info(f"Applying LDA score to {filename_suffix} units")
        scores = lda.transform(get_x(units_df)).squeeze()
        assert scores.ndim == 1
        assert len(scores) == len(units_df)
        lda_df = (
            units_df
            .with_columns(lda=pl.Series(scores))
        )
        lda_df.write_parquet(f"//allen/programs/mindscope/workgroups/dynamicrouting/ben/lda_{filename_suffix}.parquet")

        print(lda_df.describe())

if __name__ == '__main__':
    main()