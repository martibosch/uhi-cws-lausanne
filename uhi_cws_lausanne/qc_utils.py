"""Quality checks (QC) utils."""

import geopandas as gpd
import numpy as np
import pandas as pd
from meteora import qc


def per_heatwave_qc(
    ts_df: pd.DataFrame,
    *,
    station_gdf: gpd.GeoDataFrame | None = None,
    unreliable_threshold: float | None = None,
    lower_alpha: float | None = None,
    upper_alpha: float | None = None,
    radiative_error_max_prop_threshold: float | None = None,
    station_indoor_corr_threshold: float | None = None,
    adjust_elevation: bool | None = None,
    station_elevation: pd.Series | str | None = None,
    atmospheric_lapse_rate: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Per-heatwave QC."""
    # systematic radiative error stations kwargs
    radiative_error_stations_kwargs = dict(
        lower_alpha=lower_alpha,
        upper_alpha=upper_alpha,
        max_prop_threshold=radiative_error_max_prop_threshold,
    )

    # daily z peak station kwargs
    daily_z_peak_stations_kwargs = dict(
        lower_alpha=lower_alpha,
        upper_alpha=upper_alpha,
    )

    qc_keys = ["unreliable", "radiative_error", "daily_z_peak", "indoor"]
    qc_dict = {qc_key: {} for qc_key in qc_keys}
    heatwave_ts_dfs = []
    for heatwave, heatwave_ts_df in ts_df.groupby(level="heatwave"):
        _heatwave_ts_df, heatwave_qc_dict = qc.full_qc(
            heatwave_ts_df.droplevel("heatwave"),
            unreliable_threshold=unreliable_threshold,
            radiative_error_stations_kwargs=radiative_error_stations_kwargs,
            daily_z_peak_stations_kwargs=daily_z_peak_stations_kwargs,
            station_indoor_corr_threshold=station_indoor_corr_threshold,
            replace_outliers=True,
            replacement_value=np.nan,
            adjust_elevation=adjust_elevation,
            station_elevation=station_elevation,
            atmospheric_lapse_rate=atmospheric_lapse_rate,
        )
        heatwave_ts_dfs.append(
            _heatwave_ts_df.assign(heatwave=heatwave)
            .reset_index()
            .set_index(["heatwave", "time"])
        )
        for qc_key in heatwave_qc_dict:
            qc_dict[qc_key][heatwave] = heatwave_qc_dict[qc_key]

    return pd.concat(heatwave_ts_dfs), qc_dict


def qc_aspect_df(qc_aspect_dict):
    """Get a dictionary of QC data frames."""
    qc_aspect_df = pd.DataFrame(
        index=qc_aspect_dict.keys(), columns=list(set().union(*qc_aspect_dict.values()))
    )
    for heatwave in qc_aspect_dict:
        qc_aspect_df.loc[heatwave] = qc_aspect_df.columns.isin(qc_aspect_dict[heatwave])
    return qc_aspect_df
