import pandas as pd
from core.config import SNOTEL_DIR
from datetime import timedelta
from .utils import haversine_distance, weighted_mean
from .metadata import get_metadata


METADATA_DF = get_metadata()
STATIONS_METADATA_DF = pd.read_csv(SNOTEL_DIR / "station_metadata.csv")
STATIONS_MAPPING_DF = pd.read_csv(SNOTEL_DIR / "sites_to_snotel_stations.csv")


def get_snotel_features(issue_date, lookback=15):
    issue_date = pd.to_datetime(issue_date)
    forecast_year = issue_date.year
    fy_data_dir = SNOTEL_DIR / f"FY{forecast_year}"
    files = list(fy_data_dir.glob("*.csv"))
    feature_cols = [
        "PREC_DAILY",
        "SNWD_DAILY",
        "TAVG_DAILY",
        "TMAX_DAILY",
        "TMIN_DAILY",
        "WTEQ_DAILY",
    ]
    empty_df = pd.DataFrame(columns=feature_cols, index=METADATA_DF["site_id"])

    if len(files) == 0:
        return empty_df

    snotel_df = pd.concat(
        (pd.read_csv(fp).assign(snotel_site=fp.stem.replace("_", ":")) for fp in files),
        ignore_index=True,
    )
    snotel_df = snotel_df.reindex(columns=feature_cols + ["date", "snotel_site"])
    snotel_df["date"] = pd.to_datetime(snotel_df["date"])

    snotel_df_filtered = snotel_df[
        snotel_df["date"].between(issue_date - timedelta(lookback), issue_date - timedelta(1))
    ]
    snotel_df_agg = snotel_df_filtered.groupby("snotel_site")[feature_cols].mean()  # over time

    snotel_features = (
        snotel_df_agg.merge(STATIONS_MAPPING_DF, left_on="snotel_site", right_on="stationTriplet")
        .merge(
            STATIONS_METADATA_DF[["stationTriplet", "elevation", "latitude", "longitude"]],
            on="stationTriplet",
        )
        .merge(
            METADATA_DF[["site_id", "latitude", "longitude"]],
            on="site_id",
            suffixes=("_station", "_site"),
        )
    )
    snotel_features["weight"] = 1 / (haversine_distance(snotel_features))
    snotel_features_agg = snotel_features.groupby(["site_id"]).apply(
        weighted_mean, cols=feature_cols, weight_col="weight"
    )
    return snotel_features_agg
