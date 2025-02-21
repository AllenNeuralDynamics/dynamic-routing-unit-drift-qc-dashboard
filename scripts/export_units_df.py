# used to correct spike times from KS4 sorted NWBs in https://codeocean.allenneuraldynamics.org/capsule/2210357/tree

import json
import pathlib
import time

import lazynwb
import polars as pl
import npc_lims
import matplotlib.pyplot as plt
import zarr
import numpy as np

import paths


session_dfs = []
for path in (paths.KS4_21, paths.KS4_22, ): # paths.KS4_17, 
    dfs = []

    session_id = '_'.join(path.split('/')[-1].split('_')[1:3])
    print(session_id)
    timing_data = json.loads(pathlib.Path(f'{session_id}_timing.json').read_text())
    
    for (device_name, ), df in pl.DataFrame(
            lazynwb.get_df(path, 'units')
            .pipe(lazynwb.merge_array_column, 'spike_times')
            .pipe(lazynwb.merge_array_column, 'obs_intervals')
        ).group_by('device_name'):
        
        dfs.append(
            df.with_columns(
                spike_times=(pl.col('spike_times') / 30_000) * timing_data[device_name]['sampling_rate'] + timing_data[device_name]['start_time'],
                session_id=pl.lit(session_id),
                unit_id=pl.concat_str([
                    pl.lit(f'{session_id}_{device_name.removeprefix("Probe")}-'), 
                    pl.col("ks_unit_id"), 
                    pl.lit('_ks4'),
                ]),
            )
        )
    session_df = pl.concat(dfs, how='vertical_relaxed')
    session_df.write_parquet(f'/results/{session_id}_ks4_units.parquet', compression_level=22)
    session_dfs.append(session_df.drop('spike_times'))
big_df = pl.concat(dfs, how='diagonal_relaxed')
big_df.write_parquet('s3://aind-scratch-data/dynamic-routing/unit-rasters-ks4/units.parquet')

print(big_df)
