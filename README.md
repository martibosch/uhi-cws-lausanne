<!-- prettier-ignore-start -->

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/martibosch/uhi-cws-lausanne/main.svg)](https://results.pre-commit.ci/latest/github/martibosch/uhi-cws-lausanne/main)
[![GitHub license](https://img.shields.io/github/license/martibosch/uhi-cws-lausanne.svg)](https://github.com/martibosch/uhi-cws-lausanne/blob/main/LICENSE)

<!-- prettier-ignore-end -->

# Urban heat islands in Lausanne with citizen weather stations

Analysis of the urban heat island (UHI) effect in Lausanne using citizen weather stations (CWS).

![t-pred](https://github.com/martibosch/uhi-cws-lausanne/raw/main/reports/figures/t-t-mean-maps.png)

See the following *key* notebooks:

- Download Netatmo CWS data using [meteora](https://github.com/martibosch/meteora): [notebooks/cws-download-data.ipynb](https://github.com/martibosch/uhi-cws-lausanne/blob/main/notebooks/cws-download-data.ipynb)
- Quality control (QC) of CWS data using [meteora](https://github.com/martibosch/meteora): [notebooks/cws-qc.ipynb](https://github.com/martibosch/uhi-cws-lausanne/blob/main/notebooks/cws-qc.ipynb)
- Exploratory data analysis (EDA) of the urban temperature: [notebooks/uhi-eda.ipynb](https://github.com/martibosch/uhi-cws-lausanne/blob/main/notebooks/uhi-eda.ipynb)
- *Land use regression* linking temperature to spatial preditors: [notebooks/t-regression.ipynb](https://github.com/martibosch/uhi-cws-lausanne/blob/main/notebooks/t-regression.ipynb)

as well as the [Snakefile](https://github.com/martibosch/uhi-cws-lausanne/raw/main/Snakefile) for the overall orchestration of the computation pipeline.

## Acknowledgments

- Based on the [cookiecutter-data-snake :snake:](https://github.com/martibosch/cookiecutter-data-snake) template for reproducible data science.
