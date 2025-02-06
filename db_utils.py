import datetime
import logging
import random
import time
from enum import IntEnum

import polars as pl
import upath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CACHE_VERSION = "v0.0.261"
SCRATCH_DIR = upath.UPath("s3://aind-scratch-data/dynamic-routing")
RASTER_DIR = SCRATCH_DIR / f"unit-rasters/{CACHE_VERSION}/fig3c/sessions"

TABLE_NAME = "from_app"
DB_PATH = "//allen/programs/mindscope/workgroups/dynamicrouting/ben/unit_drift.parquet"

CACHED_DF_PATH = "s3://aind-scratch-data/dynamic-routing/cache/nwb_components/{}/consolidated/{}.parquet"


def get_units_df() -> pl.DataFrame:
    return (
        pl.scan_parquet(CACHED_DF_PATH.format(CACHE_VERSION, "units"))
        .select(
            [
                "session_id",
                "unit_id",
                "structure",
                "default_qc",
                "presence_ratio",
                "drift_ptp",
                "obs_intervals",
            ]
        )
        .collect()
        .join(
            other=(
                pl.read_parquet(CACHED_DF_PATH.format(CACHE_VERSION, "session"))
                .filter(
                    pl.col("keywords").list.contains("production"),
                )
            ),
            on="session_id",
            how="semi",
        )
        # filter out units that are only partially-observed within the task
        .join(
            other=(
                pl.read_parquet(CACHED_DF_PATH.format(CACHE_VERSION, "epochs"))
                .filter(pl.col("script_name") == "DynamicRouting1").select(
                    "session_id", "stop_time", "start_time"
                )
            ),
            on="session_id",
            how="left",
        )
        .filter(
            pl.col("obs_intervals").list.get(0).list.get(0).le(pl.col("start_time")),
            pl.col("obs_intervals").list.get(0).list.get(1).ge(pl.col("stop_time")),
        )
        .drop("obs_intervals")
    )


class UnitDriftRating(IntEnum):
    NO = 0
    YES = 1
    UNSURE = 5


class Lock:
    def __init__(self, path=DB_PATH + ".lock"):
        self.path = upath.UPath(path)

    def acquire(self):
        while self.path.exists():
            time.sleep(0.01)
        self.path.touch()

    def release(self):
        upath.UPath(self.path).unlink()

    def __enter__(self):
        self.acquire()
        logger.info(f"Acquired lock at {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        logger.info(f"Released lock at {self.path}")


def create_db(
    save_path: str = DB_PATH,
    overwrite: bool = False,
    with_paths=True,
) -> pl.DataFrame:
    if upath.UPath(save_path).exists() and not overwrite:
        raise FileExistsError(
            f"{save_path} already exists: set overwrite=True to overwrite"
        )
    df = (
        get_units_df()
        .select(
            pl.col("unit_id"),
            pl.col("session_id"),
        )
        .with_columns(
            drift_rating=pl.lit(None),
            checked_timestamp=pl.lit(None),
        )
    )
    if with_paths:
        logger.warning("getting paths from S3 for all available units")
        paths = set(p for p in RASTER_DIR.rglob("*") if p.suffix == ".png")
        # remove duplicate unit_ids:
        id_to_path = {p.stem.removeprefix("fig3c_"): str(p) for p in paths}
        paths_df = pl.DataFrame(
            {
                "unit_id": id_to_path.keys(),
                "path": id_to_path.values(),
            }
        )
        df = df.join(paths_df, on="unit_id", how="left")
        assert df.n_unique("unit_id") == len(df), "Duplicate unit_ids found"
    else:
        df = df.with_columns(path=pl.lit(None))
    df.write_parquet(save_path)
    return df


def update_db(db_path: str = DB_PATH) -> None:
    upath.UPath(
        db_path + f".backup.{datetime.datetime.now():%Y%m%d_%H%M%S}"
    ).write_bytes(upath.UPath(db_path).read_bytes())
    temp_path = db_path + ".temp"
    # get a brand new df with up-to-date paths
    new_df = create_db(save_path=temp_path, overwrite=True)
    original_df = pl.read_parquet(db_path)
    (
        new_df
        .select("session_id", "unit_id", "path")
        .join(
            other=(
                original_df
                .select("unit_id", "drift_rating", "checked_timestamp")
            ),
            on="unit_id",
            how="left",
            # coalesce=True,
        )
    ).write_parquet(db_path)
    assert (df := pl.read_parquet(db_path)).n_unique("unit_id") == len(
        df
    ), "Duplicate unit_ids found"
    logger.info(f"Updated {db_path}: {len(original_df)} -> {len(df)} rows")
    upath.UPath(temp_path).unlink()


def reset_db(src_path: str, dest_path: str) -> None:
    pl.read_parquet(src_path).with_columns(
        drift_rating=pl.lit(None), checked_timestamp=pl.lit(None)
    ).write_parquet(dest_path)


def get_df(
    already_checked: bool | None = None,
    unit_id_filter: str | None = None,
    session_id_filter: str | None = None,
    drift_rating_filter: int | None = None,
    with_paths: bool | None = True,
    db_path=DB_PATH,
) -> pl.DataFrame:
    filter_exprs = []
    if with_paths is True:
        filter_exprs.append(pl.col("path").is_not_null())
    elif with_paths is False:
        filter_exprs.append(pl.col("path").is_null())
    if already_checked is True and drift_rating_filter is None:
        filter_exprs.append(pl.col("drift_rating").is_not_null())
    elif already_checked is False and drift_rating_filter is None:
        filter_exprs.append(pl.col("drift_rating").is_null())
    if session_id_filter:
        filter_exprs.append(pl.col("session_id").str.starts_with(session_id_filter))
    if unit_id_filter:
        filter_exprs.append(pl.col("unit_id") == unit_id_filter)
    if drift_rating_filter is not None:
        filter_exprs.append(pl.col("drift_rating") == int(drift_rating_filter))
    if filter_exprs:
        logger.info(
            f"Filtering units df with {' & '.join([str(f) for f in filter_exprs])}"
        )
    return pl.read_parquet(db_path).filter(*filter_exprs)


def unit_generator(
    already_checked: bool | None = False,
    unit_id_filter: str | None = None,
    session_id_filter: str | None = None,
    drift_rating_filter: int | None = None,
    with_paths: bool | None = True,
    db_path=DB_PATH,
):
    while True:
        df = get_df(
            already_checked=already_checked,
            unit_id_filter=unit_id_filter,
            session_id_filter=session_id_filter,
            drift_rating_filter=drift_rating_filter,
            with_paths=with_paths,
            db_path=db_path,
        )
        session_ids = df["session_id"].unique().to_list()
        if not session_ids:
            raise StopIteration("No more sessions to check")
        random.shuffle(session_ids)
        for session_id in session_ids:
            sub_df = df.filter(pl.col("session_id") == session_id)
            if sub_df.is_empty():
                logger.info(f"No more units to check for {session_id}")
                continue
            yield sub_df.sample(1)["unit_id"].first()


def set_drift_rating_for_unit(unit_id: str, drift_rating: int, db_path=DB_PATH) -> None:
    timestamp = int(time.time())
    original_df = get_df()
    unit_id_filter = pl.col("unit_id") == unit_id
    logger.info(f"Updating row for {unit_id} with drift_rating={drift_rating}")
    df = original_df.with_columns(
        drift_rating=pl.when(unit_id_filter)
        .then(pl.lit(drift_rating))
        .otherwise(pl.col("drift_rating")),
        checked_timestamp=pl.when(unit_id_filter)
        .then(pl.lit(timestamp))
        .otherwise(pl.col("checked_timestamp")),
    )
    assert len(df) == len(
        original_df
    ), f"Row count changed: {len(original_df)} -> {len(df)}"
    with Lock():
        df.write_parquet(db_path)
    logger.info(f"Overwrote {db_path}")


def test_db(with_paths=False):
    db_path = "test.parquet"
    create_db(overwrite=True, save_path=db_path, with_paths=with_paths)
    i = next(unit_generator(db_path=db_path))
    set_drift_rating_for_unit(i, 2, db_path=db_path)
    df = get_df(unit_id_filter=i, already_checked=True, db_path=db_path)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    upath.UPath(db_path).unlink()
