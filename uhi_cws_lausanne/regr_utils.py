"""Regression utils."""

import geopandas as gpd
import multilandpy
import numpy as np
import pandas as pd
import seaborn as sns
import spreg
from pysal.lib import weights
from sklearn import feature_selection, linear_model, preprocessing


def weights_from_gser(sample_gser):
    """Get distance band weights for a set of sample (point) locations."""
    # station_gser = station_features_gdf.loc[y_ser.index]["geometry"]
    dist_threshold = multilandpy.fully_connected_threshold(sample_gser)

    # distance-based weight matrix
    w = weights.DistanceBand.from_dataframe(
        sample_gser.to_frame().reset_index(drop=True), threshold=dist_threshold
    )
    # row-standardize the weights
    w.transform = "r"
    return w


def coeff_df(model, *, stat_attr="t_stat", constant_col="CONSTANT"):
    """Get model coefficients as a data frame."""
    return (
        pd.DataFrame(
            {
                "beta": model.betas.flatten(),
                "p": np.array(getattr(model, stat_attr))[:, 1],
            },
            index=model.name_x,
        )
        .drop(constant_col)
        .reset_index(names="feature")
    )


class Regressor:
    """Regression helper.

    Parameters
    ----------
    station_features_gdf : geopandas.GeoDataFrame
        Station locations and features as a geo-data frame, indexed by station ids.
    y_ser : pandas.Series
        Target variable as a pandas Series, indexed by station ids.
    """

    def __init__(self, station_features_gdf, y_ser):
        """Initialize the Regressor with station features and target variable."""
        # preprocess the data
        X_df = station_features_gdf.loc[y_ser.index].drop(columns="geometry")
        scaler = preprocessing.StandardScaler().fit(X_df)
        X_df = pd.DataFrame(
            scaler.transform(X_df),
            index=X_df.index,
            columns=X_df.columns,
        )
        self.scaler = scaler
        self.X_df = X_df
        # TODO: filter also y_ser based on X_df indices?
        self.y_ser = y_ser

        station_gser = station_features_gdf.loc[y_ser.index]["geometry"]
        self.w = weights_from_gser(station_gser)

    def sequential_feature_selector(self, estimator=None, **sfs_kwargs):
        """Perform sequential feature selection.

        Parameters
        ----------
        estimator : sklearn estimator, optional
            Estimator to use for feature selection. Defaults to LinearRegression.
        sfs_kwargs : dict, optional
            Additional keyword arguments for the SequentialFeatureSelector.

        Returns
        -------
        sklearn.feature_selection.SequentialFeatureSelector
        """
        if estimator is None:
            estimator = linear_model.LinearRegression()
        if sfs_kwargs is None:
            _sfs_kwargs = {}
        else:
            _sfs_kwargs = sfs_kwargs.copy()
        _sfs_kwargs = {
            **{
                "n_features_to_select": "auto",
                "tol": 0.001,
                "direction": "forward",
                "scoring": "r2",
                "cv": 5,
            },
            **_sfs_kwargs,
        }

        sfs = feature_selection.SequentialFeatureSelector(estimator, **_sfs_kwargs)
        sfs.fit(self.X_df, self.y_ser)

        return sfs

    def scale_of_effect_selector(self, *, criteria="pearsonr", **scale_eval_df_kwargs):
        """Select the optimal feature scale based on the specified criteria.

        Parameters
        ----------
        criteria : str, optional
            Criteria to use for selecting the optimal feature scale. Must be a method of
            `scipy.stats`. Defaults to "pearsonr".
        scale_eval_df_kwargs : dict, optional
            Additional keyword arguments for the `multilandpy.scale_eval_df` function.

        Returns
        -------
        feature_columns : list
            List of feature columns with the optimal scale for each feature.
        """
        # get multi-level columns of the form (feature_group, buffer_dist)
        # feature_df.columns =
        X_df = self.X_df.copy()
        X_df.columns = X_df.columns.str.rsplit("_", n=1, expand=True).set_names(
            ["feature", "scale"]
        )
        scale_level = X_df.columns.get_level_values("scale")
        multiscale_sel = scale_level.str.isnumeric() | scale_level.isna()
        eval_df = multilandpy.scale_eval_df(
            X_df.loc[:, multiscale_sel].stack(level="scale", future_stack=True),
            self.y_ser,
            criteria,
        )
        return [
            f"{feature}_{scale}"
            for feature, scale in eval_df.loc[
                eval_df.groupby("feature")[criteria].idxmax().values
            ]
            .set_index("feature")["scale"]
            .to_dict()
            .items()
        ] + list(self.X_df.columns[~(scale_level.str.isnumeric() | scale_level.isna())])

    def fit_model(self, *, model_class=None, features=None, **model_kwargs):
        """Fit a regression model to the data.

        Parameters
        ----------
        model_class : class, optional
            Class of the regression model to fit. Must be a statsmodels or spreg model
            class if using spatial weights. Defaults to spreg.OLS.
        features : list, optional
            List of feature names to use for the model. If None, all features are used.
        model_kwargs : dict, optional
            Additional keyword arguments for the model class.

        Returns
        -------
        model : statsmodels or spreg model
        """
        if features is None:
            features = self.X_df.columns
        if model_class is None:
            model_class = spreg.OLS
        if "w" in model_kwargs:
            w = model_kwargs.pop("w")
        else:
            w = self.w
        for key in ["spat_diag", "moran"]:
            if key not in model_kwargs:
                model_kwargs[key] = True
        model = model_class(
            self.y_ser,
            self.X_df.loc[:, features],
            w=w,
            **model_kwargs,
        )
        return model

    def predict(self, model, X_df, *, features=None, w=None, pred_label="pred"):
        """Predict using the fitted model.

        Parameters
        ----------
        model : statsmodels results or spreg model
            Fitted regression model to use for predictions.
        X_df : pd.DataFrame
            Sample features to use for predictions.
        features : list, optional
            List of feature names to use for the model. If None, all features are used.
        w : libpysal.weights.W, optional
            Spatial weights to use for predictions. If None, no spatial weights are
            used.
        pred_label : str, optional
            Label for the predicted series. Defaults to "pred".

        Returns
        -------
        pred_ser : pd.Series
            Series of predicted values for each sample.
        """
        # scale features with the scaler fitted on the training data
        X_df = pd.DataFrame(
            # preprocessing.StandardScaler().fit_transform(X_df),
            self.scaler.transform(X_df),
            index=X_df.index,
            columns=X_df.columns,
        )
        if features is not None:
            X_df = X_df.loc[:, features]
        if w is not None:
            X_df = np.column_stack([X_df, weights.lag_spatial(w, X_df)])
        else:
            X_df = X_df.values

        return pd.Series(
            (X_df @ model.betas[1:] + model.betas[0]).flatten(), name=pred_label
        )

    def predict_raster(self, model, X_gdf, *, features=None, w=None, pred_label="pred"):
        """Predict using the fitted model on a geo-data frame and return a data array.

        Parameters
        ----------
        model : statsmodels results or spreg model
            Fitted regression model to use for predictions.
        X_gdf : gpd.GeoDataFrame
            Sample features to use for predictions, must contain a "geometry" column.
        features : list, optional
            List of feature names to use for the model. If None, all features are used.
        w : libpysal.weights.W, optional
            Spatial weights to use for predictions. If None, no spatial weights are
            used.
        pred_label : str, optional
            Label for the predicted series. Defaults to "pred".

        Returns
        -------
        pred_da : xarray.DataArray
            DataArray of predicted values for each sample, indexed by y and x
            coordinates from the geo-data frame.
        """
        pred_gdf = gpd.GeoDataFrame(
            self.predict(
                model,
                X_gdf.drop(columns="geometry"),
                features=features,
                w=w,
                pred_label=pred_label,
            ),
            geometry=X_gdf["geometry"],
        )
        for coord in ["x", "y"]:
            pred_gdf[coord] = getattr(pred_gdf["geometry"], coord)
        return (
            pred_gdf.reset_index(drop=True)
            .set_index(["y", "x"])
            .drop(columns="geometry")
            .to_xarray()[pred_label]
        )

    def regplot(
        self,
        y_pred,
        *,
        obs_label=None,
        pred_label=None,
        **regplot_kwargs,
    ):
        """Create a regression plot of observed vs predicted values.

        Parameters
        ----------
        y_pred : pd.Series or np.ndarray
            Predicted values to plot against observed values.
        obs_label, pred_label : str, optional
            Labels for the observed and predicted series, respectively.
        regplot_kwargs : dict, optional
            Additional keyword arguments for seaborn's regplot function.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes object containing the regression plot.
        """
        y_obs = self.y_ser
        if obs_label is not None:
            y_obs = y_obs.rename(obs_label)
        if pred_label is not None:
            y_pred = pd.Series(y_pred, name=pred_label)

        # if hue_labels is not None and "scatter_kws" not in regplot_kwargs:
        #     regplot_kwargs["scatter_kws"] = {
        #         "color": pd.Series(hue_labels).map(
        #             {
        #                 label: color
        #                 for label, color in zip(
        #                     hue_labels.unique(), sns.color_palette(palette)
        #                 )
        #             }
        #         )
        #     }

        ax = sns.regplot(x=y_obs, y=y_pred, **regplot_kwargs)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        _min = min(x_min, y_min)
        _max = max(x_max, y_max)
        ax.set_xlim(_min, _max)
        ax.set_ylim(_min, _max)
        ax.set_aspect("equal")

        return ax

    def lmplot(
        self, y_pred, *, obs_label=None, pred_label=None, hue=None, **lmplot_kwargs
    ):
        """Create a linear model plot of observed vs predicted values."""
        y_obs = self.y_ser
        if obs_label is None:
            obs_label = getattr(y_obs, "name") or "Observed"
        if pred_label is None:
            pred_label = getattr(y_pred, "name") or "Predicted"

        plot_df = pd.concat(
            [
                y_obs.reset_index(drop=True),
                pd.Series(y_pred).reset_index(drop=True),
            ],
            axis="columns",
            ignore_index=True,
        )
        plot_df.columns = [obs_label, pred_label]
        # if pd_types.is_list_like(hue):
        #     plot_df["hue"] = pd.Series(hue, name="hue").reset_index(drop=True)
        #     hue = "hue"

        return sns.lmplot(
            data=plot_df, x=obs_label, y=pred_label, hue=hue, **lmplot_kwargs
        )
