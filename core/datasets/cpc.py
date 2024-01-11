from datetime import datetime

import numpy as np
import pandas as pd
from wsfr_read.climate.cpc_outlooks import (
    read_cpc_outlooks_precip,
    read_cpc_outlooks_temp,
)

from .metadata import get_metadata

METADATA_DF = get_metadata()


def _preprocess_cpc_data(issue_date, cpc_df, site_cds, prev_inclusion=2):
    cpc_df = cpc_df.reset_index()
    cpc_df["date"] = cpc_df[["YEAR", "MN"]].apply(
        lambda r: datetime(r["YEAR"], r["MN"], 1), axis=1
    )

    latest_date = cpc_df["date"].max()
    latest_month = latest_date.month
    site_df = cpc_df[cpc_df["date"] == latest_date].merge(site_cds, left_on="CD", right_on="cd")

    forecast_start = np.maximum(
        site_df["season_start_month"] - prev_inclusion, issue_date.month
    )  # TODO: issue date could be from previous year
    forecast_end = site_df["season_end_month"]

    lead_start = (forecast_start - latest_month) % 12
    lead_end = (forecast_end - latest_month) % 12
    site_df = site_df[site_df["LEAD"].between(lead_start, lead_end)].drop(
        columns=["season_start_month", "season_end_month"]
    )
    return site_df


def get_cpc_data(issue_date, df_metadata=None):
    if df_metadata is None:
        df_metadata = METADATA_DF
    site_cds = df_metadata[["site_id", "season_start_month", "season_end_month", "cd"]].explode(
        "cd"
    )
    try:
        cpc_outlooks_precip = read_cpc_outlooks_precip(issue_date=issue_date, site_id=None)
        cpc_outlooks_temp = read_cpc_outlooks_temp(issue_date=issue_date, site_id=None)
    except:
        return pd.DataFrame(columns=["precip_score", "precip_sd", "temp_mean", "temp_sd"])

    site_precip = _preprocess_cpc_data(cpc_outlooks_precip, site_cds)
    site_temp = _preprocess_cpc_data(cpc_outlooks_temp, site_cds)

    site_precip["precip_score"] = site_precip["F MEAN"] ** site_precip["POWER"]
    site_precip = site_precip.rename(columns={"F SD": "precip_sd"})
    site_precip_agg = site_precip.groupby("site_id")[["precip_score", "precip_sd"]].mean()

    site_temp = site_temp.rename(columns={"F MEAN": "temp_mean", "F SD": "temp_sd"})
    site_temp_agg = site_temp.groupby("site_id")[["temp_mean", "temp_sd"]].mean()

    feats_merged = site_precip_agg.merge(site_temp_agg, on="site_id", how="outer").reset_index()
    return feats_merged
