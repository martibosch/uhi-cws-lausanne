"""Quality checks (QC) utils."""

import geopandas as gpd
import numpy as np
import pandas as pd
from meteora import qc


def sequential_qc(
    ts_df: pd.DataFrame,
    *,
    station_gdf: gpd.GeoDataFrame | None = None,
    unreliable_threshold: float | None = None,
    low_alpha: float | None = None,
    high_alpha: float | None = None,
    systematic_outlier_station_threshold: float | None = None,
    outlier_values: str | None = None,
    direct_radiation_outlier_threshold: float | None = None,
    station_indoor_corr_threshold: float | None = None,
) -> dict:
    """Sequential QC."""
    qc_dict = {}

    # unreliable stations
    unreliable_stations = qc.get_unreliable_stations(
        ts_df, unreliable_threshold=unreliable_threshold
    )
    ts_df = ts_df.drop(columns=unreliable_stations, errors="ignore")
    qc_dict["unreliable"] = unreliable_stations

    # systematic outlier stations
    systematic_outlier_stations = qc.get_systematic_outlier_stations(
        ts_df,
        low_alpha=low_alpha,
        high_alpha=high_alpha,
        station_outlier_threshold=systematic_outlier_station_threshold,
    )
    ts_df = ts_df.drop(columns=systematic_outlier_stations, errors="ignore")
    qc_dict["systematic_outlier"] = systematic_outlier_stations

    # direct radiation outlier stations
    # TODO:
    # if outlier_threshold is None:
    #     outlier_threshold = "three-sigma"
    outlier_ts_df = qc.get_outlier_ts_df(
        ts_df,
        direction="upper",  # outlier_threshold=outlier_threshold
    )
    if direct_radiation_outlier_threshold is None:
        direct_radiation_outlier_threshold = 0.05
    prop_outlier_ts_ser = outlier_ts_df.sum() / len(outlier_ts_df.index)
    qc_dict["direct_radiation_outlier"] = prop_outlier_ts_ser.index[
        prop_outlier_ts_ser.gt(direct_radiation_outlier_threshold)
    ]

    if outlier_values is None:
        # TODO: get from settings
        outlier_values = "replace"
    if outlier_values == "replace":
        outlier_ts_df = qc.get_outlier_ts_df(
            ts_df,  # direction=outlier_direction, threshold=outlier_threshold
        )
    # elif outlier_values == "remove_station":
    #     # TODO: get from settings
    #     if max_outlier_values_threshold is None:
    #         max_outlier_values_threshold = 0.05
    #     prop_outlier_ts_ser = outlier_ts_df.sum() / len(outlier_ts_df.index)

    #     qc_dict["outlier_values"] = prop_outlier_ts_ser.index[
    #         prop_outlier_ts_ser.gt(max_outlier_values_threshold)
    #     ]

    # indoor stations
    indoor_stations = qc.get_indoor_stations(
        ts_df, station_indoor_corr_threshold=station_indoor_corr_threshold
    )
    ts_df = ts_df.drop(columns=indoor_stations, errors="ignore")
    qc_dict["indoor"] = indoor_stations

    return qc_dict


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
    # elevation adjustment (optional), only once (heatwave independent)
    if adjust_elevation:
        if isinstance(station_elevation, str):
            # `station_elevation` is a column of `station_gdf`
            station_elevation = station_gdf[station_elevation]
        # at this point `station_elevation` must be a series indexed by the station ids
        ts_df = qc.elevation_adjustment(
            ts_df, station_elevation, atmospheric_lapse_rate=atmospheric_lapse_rate
        )

    # systematic radiative error stations kwargs
    radiative_error_stations_kwargs = dict(
        lower_alpha=lower_alpha,
        upper_alpha=upper_alpha,
        max_prop_threshold=radiative_error_max_prop_threshold,
    )

    qc_keys = ["unreliable", "radiative_error", "daily_peak_overheating", "indoor"]
    qc_dict = {qc_key: {} for qc_key in qc_keys}
    heatwave_ts_dfs = []
    for heatwave, heatwave_ts_df in ts_df.groupby(level="heatwave"):
        _heatwave_ts_df, heatwave_qc_dict = qc.full_qc(
            heatwave_ts_df.droplevel("heatwave"),
            unreliable_threshold=unreliable_threshold,
            radiative_error_stations_kwargs=radiative_error_stations_kwargs,
            station_indoor_corr_threshold=station_indoor_corr_threshold,
            adjust_elevation=False,
            replace_outliers=True,
            replacement_value=np.nan,
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
