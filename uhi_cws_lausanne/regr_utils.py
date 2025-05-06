"""Regression utils."""

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, utils
from tqdm.auto import tqdm


def get_long_ts_df(wide_ts_df, station_id_col, target_var_label):
    """Get long time series data frame."""
    return wide_ts_df.reset_index().melt(
        id_vars=["heatwave", "time"],
        var_name=station_id_col,
        value_name=target_var_label,
    )


def r2(model, X, y):
    """R-squared score for the given model, input features, and target variable."""
    return np.corrcoef(y, model.predict(X))[0, 1] ** 2


def get_hourly_regr(
    ts_df, station_features_gdf, feature_cols, regr_model, *, target_var_label="UHI"
):
    """Get hourly regression models."""
    station_id_col = station_features_gdf.index.name
    long_ts_df = get_long_ts_df(ts_df, station_id_col, target_var_label)
    # ts_df = ts_df.reset_index()
    regr_dict = {}
    for hour, hour_df in tqdm(long_ts_df.groupby(long_ts_df["time"].dt.hour)):
        regr_df = (
            hour_df.merge(station_features_gdf, on=station_id_col)
            .fillna(0)
            .set_index(station_id_col)
        )
        regr_dict[hour] = regr_model().fit(
            preprocessing.StandardScaler().fit_transform(regr_df[feature_cols]),
            regr_df[target_var_label],
        )

    return regr_dict


def get_hourly_obs_pred_df(
    ts_df,
    station_features_gdf,
    feature_cols,
    regr_model,
    *,
    target_var_label="UHI",
    hour_range=None,
):
    """Get hourly observations and predictions data frame."""
    # preprocess args
    if hour_range is None:
        hour_range = ts_df.index.get_level_values("time").hour.unique()
    # get the station id column to use below
    station_id_col = station_features_gdf.index.name
    # get the feature group column to use below
    # group_col = ts_df.index.names[0]

    # dropna first and keep the index to filter out the observations respective response
    X = station_features_gdf[feature_cols].dropna()
    station_ids = X.index
    # rescale features
    X = preprocessing.StandardScaler().fit_transform(X)

    # filter out observations respective response
    # ts_df = ts_df.loc[slice(None), station_ids, slice(None)]
    ts_df = ts_df[station_ids]

    # get an hourly regression dictionary
    regr_dict = get_hourly_regr(
        ts_df,
        station_features_gdf,
        feature_cols,
        regr_model,
        target_var_label=target_var_label,
    )

    # return a long obs pred dataframe
    # # for long time series data frames
    # hour_ts_dfs = []
    # for hour in hour_range:
    #     hour_obs_ser = (
    #         ts_df.loc[
    #             slice(None),
    #             slice(None),
    #             ts_df.index.get_level_values("time").hour == hour,
    #         ]
    #         .groupby(["heatwave", "station_id"])
    #         .mean()
    #     )
    #     hour_ts_dfs.append(
    #         pd.DataFrame(
    #             {
    #                 f"{target_var_label}_obs": hour_obs_ser,
    #                 f"{target_var_label}_pred": hour_obs_ser.index.get_level_values(
    #                     "station_id"
    #                 ).map(pd.Series(regr_dict[hour].predict(X), index=station_ids)),
    #                 "heatwave": hour_obs_ser.index.get_level_values("heatwave"),
    #                 "hour": hour,
    #             }
    #         ).dropna()
    #     )
    # return pd.concat(hour_ts_dfs)
    return pd.concat(
        [
            ts_df.loc[
                (slice(None), ts_df.index.get_level_values("time").hour == hour), :
            ]
            .groupby("heatwave")
            .mean()
            .reset_index()
            .melt(
                "heatwave",
                var_name=station_id_col,
                value_name=f"{target_var_label}_obs",
            )
            .set_index(station_id_col)
            .assign(
                **{
                    f"{target_var_label}_pred": pd.Series(
                        regr_dict[hour].predict(X), index=ts_df.columns
                    ),
                    "hour": hour,
                }
            )
            for hour in hour_range
        ]
    )


def get_cross_val_df(
    X, y, model_dict, num_repetitions, num_folds, scoring, n_jobs=4, show_progress=False
):
    """Get cross-validation data frame."""
    cross_val_records = []
    if show_progress:
        pbar = tqdm(total=num_repetitions * len(model_dict))
    for i in range(num_repetitions):
        X, y = utils.shuffle(X, y)
        for model_label in model_dict:
            # cross_val_df.loc[(i, model_label)] = model_selection.cross_val_score(
            #     model_dict[model_label](), X, y, cv=num_folds, scoring=scoring
            # ).mean()
            cross_val_records.append(
                [
                    model_selection.cross_val_score(
                        model_dict[model_label](),
                        X,
                        y,
                        cv=num_folds,
                        scoring=scoring,
                        n_jobs=n_jobs,
                    ).mean(),
                    i,
                    model_label,
                ]
            )
            if show_progress:
                pbar.update(1)
    # return cross_val_df
    return pd.DataFrame(cross_val_records, columns=["score", "repetition", "model"])


def get_hourly_cross_val_df(
    uhi_df,
    station_features_gdf,
    feature_cols,
    model_dict,
    num_repetitions,
    num_folds,
    scoring=r2,
    target_var_label="UHI",
):
    """Get hourly cross-validation data frame."""
    station_id_col = station_features_gdf.index.name
    long_ts_df = get_long_ts_df(uhi_df, station_id_col, target_var_label)
    cross_val_dfs = []
    for hour, hour_df in tqdm(long_ts_df.groupby(long_ts_df["time"].dt.hour)):
        regr_df = (
            hour_df.merge(station_features_gdf, on=station_id_col)
            .fillna(0)
            .set_index(station_id_col)
        )
        cross_val_dfs.append(
            get_cross_val_df(
                # regr_df[feature_cols],
                preprocessing.StandardScaler().fit_transform(regr_df[feature_cols]),
                regr_df[target_var_label],
                model_dict,
                num_repetitions,
                num_folds,
                scoring,
            ).assign(hour=hour)
        )
    return pd.concat(cross_val_dfs, axis="rows")  # .groupby("hour").mean()
