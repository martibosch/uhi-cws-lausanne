"""Plotting utils."""

from collections.abc import Mapping

import contextily as cx
import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from uhi_cws_lausanne.utils import KwargsType


def comparison_lineplot(
    ts_df_dict: Mapping[str, pd.DataFrame],
    *,
    value_label: str = "T (°C)",
    station_label: str = "station_id",
    source_label: str = "Source",
    **lineplot_kwargs: KwargsType,
) -> mpl.axes.Axes:
    """Lineplot comparing CWS and official stations time series.

    Parameters
    ----------
    ts_df_dict : mapping of str to pandas.DataFrame
        Dictionary of time series dataframes, where the keys are the source labels and
        the values are the time series data frames of measurements (rows) for each
        station (columns)
    value_label : str, optional
        Label for the values, by default "T" (for temperature).
    station_label : str, optional
        Label for the stations, by default "station_id".
    source_label : str, optional
        Label for the source, by default "source".
    lineplot_kwargs : mapping, optional
        Keyword arguments to pass to `seaborn.lineplot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axes of the plot.

    """
    return sns.lineplot(
        data=pd.concat(
            [
                ts_df_dict[source]
                .reset_index()
                .melt(id_vars="time", var_name=station_label, value_name=value_label)
                .assign(**{source_label: source})
                for source in ts_df_dict
            ],
            ignore_index=True,
        ),
        x="time",
        y=value_label,
        hue=source_label,
        **lineplot_kwargs,
    )


def comparison_lineplots(
    ts_df_dict: Mapping[str, pd.DataFrame],
    *,
    value_label: str = "T (°C)",
    station_label: str = "station_id",
    source_label: str = "Source",
    **facetgrid_kwargs: KwargsType,
) -> sns.FacetGrid:
    """Lineplots comparing CWS and official stations time series.

    Parameters
    ----------
    ts_df_dict : mapping of str to pandas.DataFrame
        Dictionary of time series dataframes, where the keys are the source labels and
        the values are the time series data frames of measurements (rows) for each
        station (columns), multi-indexed by the heatwave identifier and the time.
    value_label, station_label, source_label : str, optional
        Label for the values, stations and source, by default "T (°C)", "station_id" and
        "Source" respectively.
    facetgrid_kwargs : mapping, optional
        Keyword arguments to pass to `seaborn.FacetGrid`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot.

    """
    if facetgrid_kwargs is None:
        _facetgrid_kwargs = {}
    else:
        _facetgrid_kwargs = facetgrid_kwargs.copy()
        _ = _facetgrid_kwargs.pop("sharex", None)
    g = sns.FacetGrid(
        pd.concat(
            [
                ts_df_dict[source]
                .reset_index()
                .melt(
                    id_vars=["time", "heatwave"],
                    var_name="station_id",
                    value_name=value_label,
                )
                .assign(**{source_label: source})
                for source in ts_df_dict
            ]
        ),
        col="heatwave",
        hue=source_label,
        sharex=False,
        **_facetgrid_kwargs,
    )
    g.map(sns.lineplot, "time", value_label)
    g.set_xticklabels(rotation=45)
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    g.add_legend()

    return g


def hourly_lineplot(
    ts_df: pd.DataFrame,
    *,
    value_label: str = "T",
    ax: mpl.axes.Axes = None,
    legend: bool = True,
) -> mpl.axes.Axes:
    """Plot of hourly time series of a variable for each heatwave."""
    if ax is None:
        _, ax = plt.subplots()
    # y = ts_df.columns[0]
    for heatwave, heatwave_ts_df in ts_df.groupby(level="heatwave"):
        heatwave_ts_df = heatwave_ts_df.stack(future_stack=True).reset_index(
            name=value_label
        )
        sns.lineplot(
            # heatwave_ts_df.assign(
            #     **{"hour": heatwave_ts_df.index.get_level_values("time").hour}
            # ),
            heatwave_ts_df.assign(**{"hour": heatwave_ts_df["time"].dt.hour}),
            x="hour",
            # y=y,
            y=value_label,
            ax=ax,
            label=heatwave,
            legend=legend,
        )

    return ax


def plot_map_by_var(
    station_gser: gpd.GeoSeries,
    var_ser: pd.Series,
    *,
    var_label: str = None,
    legend: bool = True,
    edgecolor: str = "black",
    attribution: str | bool = False,
    set_axis_off: bool = True,
    add_basemap_kws: dict | None = None,
    **plot_kws: KwargsType,
) -> mpl.axes.Axes:
    """Plot stations by variable.

    Parameters
    ----------
    station_gser : gpd.GeoSeries
        GeoSeries of stations.
    var_ser : pd.Series
        Series of variable values.
    var_label : str, optional
        Label given the variable of `var_ser`. If None, the name of `var_ser` is used.
        If None and `var_ser` has no name, "var" is used.
    legend : bool, optional
        Whether to add a legend, by default True.
    edgecolor : str, optional
        Color of the edges, by default "black".
    attribution : str | bool, optional
        Attribution of the basemap, by default False.
    set_axis_off : bool, optional
        Whether to set the axis off, by default True.
    add_basemap_kws : dict, optional
        Keyword arguments to pass to `contextily.add_basemap`, by default None.
    plot_kws : dict, optional
        Keyword arguments to pass to `geopardas.GeoDataFrame.plot`, by default None.

    Returns
    -------
    matplotlib.axes.Axes
        Axes of the plot.

    """
    if plot_kws is None:
        plot_kws = {}
    if add_basemap_kws is None:
        add_basemap_kws = {}
    if var_label is None:
        var_label = var_ser.name
    if var_label is None:
        var_label = "var"
    ax = gpd.GeoDataFrame({var_label: var_ser}, geometry=station_gser).plot(
        var_label, legend=legend, edgecolor=edgecolor, **plot_kws
    )
    cx.add_basemap(ax, crs=station_gser.crs, attribution=attribution, **add_basemap_kws)
    if set_axis_off:
        ax.set_axis_off()
    return ax


def compare_maps(
    stations_gdf: gpd.GeoDataFrame,
    var_ser: pd.Series,
    *,
    var_label: str = None,
    by_col: str = "source",
    figwidth: float = None,
    figheight: float = None,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "coolwarm",
    legend: bool = True,
) -> mpl.figure.Figure:
    """Side-by-side plot of maps of a variable.

    Parameters
    ----------
    stations_gdf : gpd.GeoDataFrame
        GeoDataFrame of stations.
    var_ser : pd.Series
        Series of variable values.
    var_label : str, optional
        Label given the variable of `var_ser`. If None, the name of `var_ser` is used.
        If None and `var_ser` has no name, "var" is used.
    by_col : str, optional
        Column of `stations_gdf` to group by, by default "source".
    figwidth, figheight : float, optional
        Width and height of the figure. If None, the default values from the matplotlib
        configuration are used.
    vmin, vmax : float, optional
        Minimum and maximum values of the color scale, by default None. If None, the
        minimum and maximum values of `var_ser` are used.
    cmap : str, optional
        Colormap to use, by default "coolwarm".
    legend : bool, optional
        Whether to add a legend, by default True.

    Returns
    -------
    mpl.figure.Figure
        Figure of the plot.

    """
    if figwidth is None:
        figwidth, _ = plt.rcParams["figure.figsize"]
    if figheight is None:
        _, figheight = plt.rcParams["figure.figsize"]

    # heatwaves = ts_df.index.get_level_values("heatwave").unique()
    # station_T_mean_gdf = stations_gdf[["source", "geometry"]].assign(
    #     T_mean=ts_df.groupby(station_id_col)["T"].mean()
    # )
    if var_label is None:
        var_label = var_ser.name
    if var_label is None:
        var_label = "var"
    station_var_gdf = stations_gdf[[by_col, "geometry"]].assign(**{var_label: var_ser})

    station_var_gb = station_var_gdf.groupby(by_col, sort=False)
    num_cols = len(station_var_gb)
    fig, axes = plt.subplots(
        1,
        num_cols,
        figsize=(figwidth * num_cols, figheight),
        layout="constrained",
    )

    # common legend
    if vmin is None:
        vmin = station_var_gdf[var_label].min()
    if vmax is None:
        vmax = station_var_gdf[var_label].max()
    # common extent
    extent = station_var_gdf.total_bounds

    for (source, source_gdf), ax in zip(station_var_gb, axes):
        ax.set_xlim(extent[[0, 2]])
        ax.set_ylim(extent[[1, 3]])
        plot_map_by_var(
            source_gdf["geometry"],
            source_gdf[var_label],
            var_label=var_label,
            legend=False,
            edgecolor="black",
            ax=ax,
            attribution=False,
            set_axis_off=True,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_axis_off()
        ax.set_title(source)

    if legend:
        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # cbar = fig.colorbar(
        #     sm,
        #     ax=ax_row,
        #     location="right",
        #     orientation="vertical",
        #     label="T$_{mean}$ ($^{\circ}$C)",
        #     pad=0.025,
        #     shrink=0.5,
        # )
        cbar_ax = fig.add_axes([1.015, 0.25, 0.01, 0.5])
        _ = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
        cbar_ax.set_ylabel(var_label)

    return fig
