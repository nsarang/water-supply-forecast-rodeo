from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from core.config import SHOW_PROGRESS_BAR
from core.data.feature import add_column_suffix, generate_timeseries_features
from core.datasets import *


def prepare_dataset(month, day, df_monthly_flow, df_targets=None):
    historical_years = df_monthly_flow["forecast_year"].unique()

    historical_issue_dates = pd.Series(
        [datetime(year, month, day) for year in historical_years], name="issue_date"
    )

    dataset = (
        get_metadata()
        .rename(columns={"area": "basins_area"})
        .merge(historical_issue_dates, how="cross")
        .query("issue_date.dt.month <= season_end_month")
    )
    dataset["forecast_year"] = dataset["issue_date"].dt.year
    dataset["issue_month"] = dataset["issue_date"].dt.month

    if df_targets is not None:
        dataset = (
            dataset.merge(
                df_targets,
                left_on=["site_id", "forecast_year"],
                right_on=["site_id", "forecast_year"],
            )
            # .dropna()
        )

    # --- FEATURE ENGINEERING ---
    # - VOLUME FEATURES -
    df_monthly_flow = df_monthly_flow.copy()
    df_monthly_flow["date"] = df_monthly_flow[["year", "month"]].apply(
        lambda r: datetime(r["year"], r["month"], 1), axis=1
    )

    volume_features = dataset.progress_apply(
        generate_timeseries_features,
        df_b=df_monthly_flow,
        condition=(
            "site_id == @row.site_id"
            " & forecast_year == @row.forecast_year"
            " & ~(date.dt.year == @row.issue_date.year & date.dt.month >= @row.issue_date.month)"
        ),
        date_col="date",
        feature_engineering=["volume"],
        axis=1,
    ).drop(columns="volume")
    assert len(volume_features) == len(dataset)
    dataset = pd.concat(add_column_suffix(dataset, volume_features, suffixes=("", "_vol")), axis=1)

    # - TELECONN FEATURES -
    teleconn_data = get_all_teleconnections_data()
    teleconn_feature_cols = {
        # "MJO": ["INDEX_2", "INDEX_5"], # ...
        "PDO": ["pdo_index"],
        # "NINO": ["NINO3.4 ANOM", "NINO3 ANOM", "NINO4"], # ...
        "ONI": ["ANOM"],  # TOTAL
        "PNA": ["pna_index"],
        "SOI": ["soi"],
    }
    feature_suffixes = [
        "_roll_std7",
        "_roll_mean5",
        "_ewm_mean7",
        "_autocorr",
        "_roll_max5",
        "_roll_min5",
    ]
    for name, df_dataset in teleconn_data.items():
        df_feats = dataset.progress_apply(
            generate_timeseries_features,
            df_b=df_dataset,
            condition=(
                "date.dt.year < @row.issue_date.year"
                " | (date.dt.year == @row.issue_date.year & date.dt.month < @row.issue_date.month)"
            ),
            date_col="date",
            feature_engineering=teleconn_feature_cols.get(name, []),
            axis=1,
        )
        dataset = pd.concat(
            add_column_suffix(dataset, df_feats, suffixes=("", "_" + name.lower())), axis=1
        )

    dataset["yr_in_2yr_cycle"] = (dataset.issue_date.dt.year - 1800) % 2
    dataset["yr_in_3yr_cycle"] = (dataset.issue_date.dt.year - 1800) % 3
    dataset["yr_in_4yr_cycle"] = (dataset.issue_date.dt.year - 1800) % 4

    # Fourier series components for N-year seasonality
    num_terms = 1  # Number of Fourier terms
    for n in range(1, num_terms + 1):
        for cycle in [2, 3, 4]:
            dataset[f"sin_yr{cycle}_{n}"] = np.sin(
                2 * np.pi * n * dataset[f"yr_in_{cycle}yr_cycle"] / cycle
            )
            dataset[f"cos_yr{cycle}_{n}"] = np.cos(
                2 * np.pi * n * dataset[f"yr_in_{cycle}yr_cycle"] / cycle
            )

    # - EXTRA FEATURES -
    pdsi_dfs, snotel_dfs, cpc_dfs, cdec_dfs, grace_dfs = [[] for _ in range(5)]

    for issue_date in tqdm(historical_issue_dates, disable=not SHOW_PROGRESS_BAR):
        pdsi_dfs.append(
            get_pdsi_features(issue_date, radius1_km=50, radius2_km=200).assign(
                issue_date=issue_date
            )
        )
        snotel_dfs.append(
            get_snotel_features(issue_date, lookback=15).assign(issue_date=issue_date)
        )
        cpc_dfs.append(get_cpc_data(issue_date).assign(issue_date=issue_date))
        cdec_dfs.append(
            get_cdec_data(issue_date, lookback=30, max_distance_km=70).assign(
                issue_date=issue_date
            )
        )
        # grace_dfs.append(get_grace_features(issue_date, radius1_km=50, radius2_km=200).assign(issue_date=issue_date))

    dataset = (
        dataset.merge(pd.concat(snotel_dfs), on=["site_id", "issue_date"], how="left")
        .merge(pd.concat(cpc_dfs), on=["site_id", "issue_date"], how="left")
        .merge(pd.concat(pdsi_dfs), on=["site_id", "issue_date"], how="left")
        .merge(pd.concat(cdec_dfs), on=["site_id", "issue_date"], how="left")
        # .merge(pd.concat(grace_dfs), on=["site_id", "issue_date"], how="left")
    )

    return dataset
