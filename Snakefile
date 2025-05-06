from os import path

PROJECT_NAME = "uhi-cws-lausanne"
CODE_DIR = "uhi_cws_lausanne"

NOTEBOOKS_DIR = "notebooks"
NOTEBOOKS_OUTPUT_DIR = path.join(NOTEBOOKS_DIR, "output")

DATA_DIR = "data"
DATA_RAW_DIR = path.join(DATA_DIR, "raw")
DATA_INTERIM_DIR = path.join(DATA_DIR, "interim")
DATA_PROCESSED_DIR = path.join(DATA_DIR, "processed")


# 0. conda/mamba environment -----------------------------------------------------------
rule create_environment:
    shell:
        "mamba env create -f environment.yml"


rule register_ipykernel:
    shell:
        "python -m ipykernel install --user --name {PROJECT_NAME} --display-name"
        " 'Python ({PROJECT_NAME})'"


# 1. data preprocessing ----------------------------------------------------------------
# 1.0. agglomeration extent ------------------------------------------------------------
rule agglom_extent:
    output:
        path.join(DATA_RAW_DIR, "agglom-extent.gpkg"),
    shell:
        "wget https://zenodo.org/records/15102601/files/agglom-extent.gpkg?download=1 "
        "-O {output}"


# 1.1. official weather stations -------------------------------------------------------

OFFICIAL_DATA_IPYNB_BASENAME = "official-data.ipynb"


rule official_data:
    input:
        agglom_extent=rules.agglom_extent.output,
        notebook=path.join(NOTEBOOKS_DIR, OFFICIAL_DATA_IPYNB_BASENAME),
    output:
        ts_df=path.join(DATA_INTERIM_DIR, "official-ts-df.csv"),
        stations_gdf=path.join(DATA_INTERIM_DIR, "official-stations.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, OFFICIAL_DATA_IPYNB_BASENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p agglom_extent_filepath {input.agglom_extent}"
        " -p dst_ts_df_filepath {output.ts_df}"
        " -p dst_stations_gdf_filepath {output.stations_gdf}"


# 1.2. citizen weather stations (Netatmo) ----------------------------------------------
# 1.2.1. download CWS data -------------------------------------------------------------
CWS_DOWNLOAD_DATA_IPYNB_BASENAME = "cws-download-data.ipynb"


rule cws_download_data:
    input:
        agglom_extent=rules.agglom_extent.output,
        official_ts_df=rules.official_data.output.ts_df,
        notebook=path.join(NOTEBOOKS_DIR, CWS_DOWNLOAD_DATA_IPYNB_BASENAME),
    output:
        ts_df=path.join(DATA_RAW_DIR, "cws-ts-df.csv"),
        stations_gdf=path.join(DATA_RAW_DIR, "cws-stations.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, CWS_DOWNLOAD_DATA_IPYNB_BASENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p agglom_extent_filepath {input.agglom_extent}"
        " -p official_ts_df_filepath {input.official_ts_df}"
        " -p dst_ts_df_filepath {output.ts_df}"
        " -p dst_stations_gdf_filepath {output.stations_gdf}"


# 1.2.2. quality control CWS data ------------------------------------------------------
CWS_QC_IPYNB_BASENAME = "cws-qc.ipynb"


rule cws_qc:
    input:
        cws_ts_df=rules.cws_download_data.output.ts_df,
        official_ts_df=rules.official_data.output.ts_df,
        cws_stations_gdf=rules.cws_download_data.output.stations_gdf,
        notebook=path.join(NOTEBOOKS_DIR, CWS_QC_IPYNB_BASENAME),
    output:
        ts_df=path.join(DATA_INTERIM_DIR, "cws-qc-ts-df.csv"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, CWS_QC_IPYNB_BASENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p cws_ts_df_filepath {input.cws_ts_df}"
        " -p official_ts_df_filepath {input.official_ts_df}"
        " -p cws_stations_gdf_filepath {input.cws_stations_gdf}"
        " -p dst_ts_df_filepath {output.ts_df}"


# 1.3 merge official and CWS data ------------------------------------------------------
MERGE_OFFICIAL_CWS_DATA_IPYNB_BASENAME = "merge-official-cws-data.ipynb"


rule merge_official_cws_data:
    input:
        official_ts_df=rules.official_data.output.ts_df,
        cws_ts_df=rules.cws_qc.output.ts_df,
        official_stations_gdf=rules.official_data.output.stations_gdf,
        cws_stations_gdf=rules.cws_download_data.output.stations_gdf,
        notebook=path.join(NOTEBOOKS_DIR, MERGE_OFFICIAL_CWS_DATA_IPYNB_BASENAME),
    output:
        ts_df=path.join(DATA_PROCESSED_DIR, "ts-df.csv"),
        stations_gdf=path.join(DATA_PROCESSED_DIR, "stations.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, MERGE_OFFICIAL_CWS_DATA_IPYNB_BASENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p official_ts_df_filepath {input.official_ts_df}"
        " -p cws_ts_df_filepath {input.cws_ts_df}"
        " -p official_stations_gdf_filepath {input.official_stations_gdf}"
        " -p cws_stations_gdf_filepath {input.cws_stations_gdf}"
        " -p dst_ts_df_filepath {output.ts_df}"
        " -p dst_stations_gdf_filepath {output.stations_gdf}"


# 1.4 land data (buildings, DEM, tree canopy) ------------------------------------------
LAND_DATA_IPYNB_BASENAME = "land-data.ipynb"
LAND_WORKING_DIR = path.join(NOTEBOOKS_DIR, "land-working-dir")
# TODO: get largest buffer dist from BUFFER_DISTS_YML


rule land_data:
    input:
        agglom_extent=rules.agglom_extent.output,
        notebook=path.join(NOTEBOOKS_DIR, LAND_DATA_IPYNB_BASENAME),
    output:
        buildings=path.join(DATA_PROCESSED_DIR, "buildings.gpkg"),
        dem=path.join(DATA_PROCESSED_DIR, "dem.tif"),
        tree_canopy=path.join(DATA_PROCESSED_DIR, "tree-canopy.tif"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, LAND_DATA_IPYNB_BASENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p agglom_extent_filepath {input.agglom_extent}"
        " -p working_dir {LAND_WORKING_DIR}"
        " -p dst_buildings_gdf_filepath {output.buildings}"
        " -p dst_dem_filepath {output.dem}"
        " -p dst_tree_canopy_filepath {output.tree_canopy}"


# 1.5 regular grid ---------------------------------------------------------------------
REGULAR_GRID_IPYNB_BASENAME = "regular-grid.ipynb"
DST_RES = 100


rule regular_grid:
    input:
        agglom_extent=rules.agglom_extent.output,
        notebook=path.join(NOTEBOOKS_DIR, REGULAR_GRID_IPYNB_BASENAME),
    output:
        regular_grid=path.join(DATA_PROCESSED_DIR, "regular-grid.gpkg"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, REGULAR_GRID_IPYNB_BASENAME),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p agglom_extent_filepath {input.agglom_extent}"
        " -p dst_res {DST_RES}"
        " -p dst_filepath {output.regular_grid}"


# 3. compute land features --------------------------------------------------------------
BUFFER_DISTS_YML = path.join(DATA_RAW_DIR, "buffer-dists.yml")
SITES_GDF_DICT = {
    "stations": rules.merge_official_cws_data.output.stations_gdf,
    "grid": rules.regular_grid.output.regular_grid,
}


# note that we do not need `agglom_extent` as input, we could use the convex hull of the stations/grid, however if we use the same "region" when computing the features, many of the intermediate results can be reused
rule _compute_land_features:
    input:
        agglom_extent=rules.agglom_extent.output,
        sites=lambda wc: SITES_GDF_DICT[wc.sites],
        buildings=rules.land_data.output.buildings,
        dem=rules.land_data.output.dem,
        tree_canopy=rules.land_data.output.tree_canopy,
        notebook=path.join(NOTEBOOKS_DIR, "compute-land-features.ipynb"),
    output:
        features=path.join(DATA_PROCESSED_DIR, "{sites}-features.csv"),
        notebook=path.join(NOTEBOOKS_OUTPUT_DIR, "compute-land-features-{sites}.ipynb"),
    shell:
        "papermill {input.notebook} {output.notebook}"
        " -p agglom_extent_filepath {input.agglom_extent}"
        " -p sites_gdf_filepath {input.sites}"
        " -p buildings_gdf_filepath {input.buildings}"
        " -p tree_canopy_filepath {input.tree_canopy}"
        " -p dem_filepath {input.dem}"
        " -f {BUFFER_DISTS_YML}"
        " -p dst_filepath {output.features}"


rule compute_land_features:
    input:
        expand(
            path.join(DATA_PROCESSED_DIR, "{sites}-features.csv"),
            sites=["stations", "grid"],
        ),
