import concurrent.futures as cf
import math

import polars as pl
import scipy.stats
import tqdm
import numpy as np


def get_annotations_df(session_id: str | None = None):
    return (
        pl.scan_parquet(
            "//allen/programs/mindscope/workgroups/dynamicrouting/ben/unit_drift.parquet"
        )
        .select("unit_id", "drift_rating")
        .filter(pl.col("unit_id").str.ends_with("_ks4").not_())
        .with_columns(
            session_id=pl.col("unit_id").str.split("_").list.slice(0, 2).list.join("_"),
        )
        .filter((pl.col("session_id") == session_id) if session_id else pl.lit(True))
        .drop_nulls("drift_rating")
        # get spike-counts ------------------------------------------------- #
        # .join(
        #     other=(
        #         pl.scan_parquet("//allen/programs/mindscope/workgroups/dynamicrouting/ben/spike_counts.parquet")
        #     ),
        #     on='unit_id',
        #     how='left',
        # )
        .join(
            other=(
                pl.scan_parquet(
                    "s3://aind-scratch-data/dynamic-routing/cache/nwb_components/v0.0.261/consolidated/trials.parquet"
                ).select(
                    "session_id",
                    "trial_index",
                    "block_index",
                    "context_name",
                    "start_time",
                    "stim_name",
                )
            ),
            on=["session_id"],
            how="left",
        )
    ).collect()


def get_samples(unit_df: pl.DataFrame, interval: str) -> list[pl.Series] | None:
    if interval == "trial" and "trial" not in unit_df.columns:
        unit_df = unit_df.with_columns(trial=pl.col("baseline") + pl.col("response"))
    if unit_df[interval].sum() == 0:
        # cannot perform test if all samples are the same (zero spikes is a common case)
        return None
    if unit_df.n_unique("block_index") < 2:
        # for Templeton we chunk spike counts into 3 segments of time
        return list(
            s[interval]
            for s in unit_df.sort("trial_index").iter_slices(
                math.ceil(len(unit_df) / 3)
            )
        )
    return unit_df.group_by("block_index").agg(pl.col(interval)).get_column(interval)


def helper(session_id):

    class NullResult:
        statistic = None
        pvalue = None

    null_result = NullResult()

    pvalue_method = None
    pvalue_method = scipy.stats.PermutationMethod(n_resamples=10_000)
    midrank = False

    results = []
    intervals_with_counts_df = get_annotations_df(session_id).join(
        other=(
            pl.scan_parquet(
                "C:/Users/ben.hardcastle/Downloads/spike_counts (2).parquet"
            )
            .filter(pl.col("unit_id").str.starts_with(session_id))
            .collect()
        ),
        on=["trial_index", "unit_id"],
        how="left",
    )
    iterable = tuple(
        intervals_with_counts_df.drop_nulls(["baseline", "response"])
        .sort("unit_id")
        .group_by("unit_id", "context_name", maintain_order=True)
    )
    with np.errstate(over='ignore'):
        for (unit_id, context_name, *_), unit_df in iterable:
            result = dict(
                unit_id=unit_id,
                context_name=context_name,
            )
            for interval in ("baseline", "response", "trial"):
                samples = get_samples(unit_df, interval)
                if samples is None:
                    stats = null_result
                else:
                    try:
                        stats = scipy.stats.anderson_ksamp(
                            samples, midrank=midrank, method=pvalue_method
                        )
                    except RuntimeWarning as e:
                        print(f"Warning for {unit_id}, {interval}: {e!r}")
                        stats = null_result
                result[f"ad_stat_{interval}"] = stats.statistic
                result[f"ad_p_{interval}"] = stats.pvalue
            results.append(result)
    return results


def main():
    annotations_df = get_annotations_df()
    results = []
    parallel = True
    if not parallel:
        for session_id in annotations_df["session_id"].unique().sort():
            results.extend(helper(session_id))
    else:
        with cf.ProcessPoolExecutor() as executor:
            future_to_session_id = {}
            for session_id in annotations_df["session_id"].unique().sort():
                future = executor.submit(helper, session_id)
                future_to_session_id[future] = session_id

            for future in tqdm.tqdm(
                cf.as_completed(future_to_session_id),
                total=annotations_df.n_unique("session_id"),
                unit="session",
            ):
                results.extend(future.result())

    result_df = pl.DataFrame(results)
    max_min_df = result_df.select(
        "unit_id",
        ad_stat_max_baseline=pl.col("ad_stat_baseline").max().over("unit_id"),
        ad_stat_max_response=pl.col("ad_stat_response").max().over("unit_id"),
        ad_stat_max_trial=pl.col("ad_stat_trial").max().over("unit_id"),
        ad_p_min_baseline=pl.col("ad_p_baseline").min().over("unit_id"),
        ad_p_min_response=pl.col("ad_p_response").min().over("unit_id"),
        ad_p_min_trial=pl.col("ad_p_trial").min().over("unit_id"),
    ).unique("unit_id")
    max_min_df.write_parquet(
        "//allen/programs/mindscope/workgroups/dynamicrouting/ben/ad_test.parquet"
    )
    return max_min_df


if __name__ == "__main__":
    print(main().describe())
