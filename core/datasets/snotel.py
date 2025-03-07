from datetime import timedelta

import pandas as pd

from core.config import SNOTEL_DIR
from core.data.feature import generate_timeseries_features_v3
from core.data.geometry import haversine_distance
from core.data.utils import dict_inter
from core.datasets.metadata import get_metadata
from core.datasets.utils import weighted_mean

STATIONS_METADATA_DF = pd.read_csv(SNOTEL_DIR / "station_metadata.csv")
STATIONS_MAPPING_DF = pd.read_csv(SNOTEL_DIR / "sites_to_snotel_stations.csv")


def get_snotel_features(
    issue_dates,
    lookback=15,
    max_distance_km=200,
    separate_stations=False,
    add_features=False,
    df_metadata=None,
    aggregate_stations=True,
    top_k=10,
):
    if df_metadata is None:
        df_metadata = get_metadata()

    issue_dates = pd.to_datetime(issue_dates)
    assert issue_dates.dt.year.nunique() == 1
    forecast_year = issue_dates.dt.year.unique()[0]

    fy_data_dir = SNOTEL_DIR / f"FY{forecast_year}"
    files = list(fy_data_dir.glob("*.csv"))
    primary_cols = [
        "PREC_DAILY",
        "SNWD_DAILY",
        "TAVG_DAILY",
        "TMAX_DAILY",
        "TMIN_DAILY",
        "WTEQ_DAILY",
    ]
    station_cols = [
        "elevation_station",
        "latitude_station",
        "longitude_station",
        "distance_station",
    ]
    # empty_df = pd.DataFrame(columns=feature_cols, index=METADATA_DF["site_id"])
    empty_df = pd.DataFrame()

    if len(files) == 0:
        return empty_df

    snotel_df = pd.concat(
        (pd.read_csv(fp).assign(snotel_site=fp.stem.replace("_", ":")) for fp in files),
        ignore_index=True,
    )
    snotel_df = snotel_df.reindex(columns=primary_cols + ["date", "snotel_site"])
    snotel_df["date"] = pd.to_datetime(snotel_df["date"])
    snotel_df = snotel_df.merge(issue_dates.rename("issue_date"), how="cross")

    snotel_df_filtered = snotel_df[
        snotel_df["date"].between(
            snotel_df["issue_date"] - timedelta(lookback), snotel_df["issue_date"] - timedelta(1)
        )
    ]

    if add_features:
        efns = {
            "roll_max": lambda c: c.rolling(window=lookback, min_periods=1).max(),
            "roll_min": lambda c: c.rolling(window=lookback, min_periods=1).min(),
            "roll_mean": lambda c: c.rolling(window=lookback, min_periods=1).mean(),
            "ewm_mean": lambda c: c.ewm(span=lookback).mean(),
            "roll_std": lambda c: c.rolling(window=lookback, min_periods=2).std(),
            "autocorr": lambda c: c.corr(c.shift(1), min_periods=3),
        }
        enrichments = {
            "WTEQ_DAILY": dict_inter(efns, ["roll_max", "roll_mean", "roll_std", "autocorr"]),
            "PREC_DAILY": dict_inter(efns, ["roll_max", "roll_mean", "roll_std", "autocorr"]),
            "TAVG_DAILY": dict_inter(efns, ["roll_max", "roll_min", "ewm_mean"]),
        }
        snotel_df_agg = (
            snotel_df_filtered.groupby(["issue_date", "snotel_site"])
            .apply(
                generate_timeseries_features_v3,
                date_col="date",
                features=primary_cols,
                enrichments=enrichments,
            )
            .reset_index()
        )
        feature_cols = primary_cols + [
            col + "_" + suffix for (col, col_enfs) in enrichments.items() for suffix in col_enfs
        ]
    else:
        feature_cols = primary_cols
        snotel_df_agg = (
            snotel_df_filtered.groupby(["issue_date", "snotel_site"])[feature_cols]
            .mean()
            .reset_index()
        )  # over time

    snotel_features = (
        snotel_df_agg.merge(
            STATIONS_METADATA_DF[["stationTriplet", "elevation", "latitude", "longitude"]],
            left_on="snotel_site",
            right_on="stationTriplet",
        ).merge(
            df_metadata[["site_id", "latitude", "longitude", "elevation"]],
            # on="site_id",
            # how="innter",
            how="cross",
            suffixes=("_station", "_site"),
        )
        # .merge(STATIONS_MAPPING_DF, on=["site_id", "stationTriplet"])
    )
    snotel_features["distance_station"] = haversine_distance(snotel_features)
    snotel_features = (
        snotel_features.query(
            f"elevation_station >= elevation_site & distance_station < {max_distance_km}"
        )
        .groupby(["issue_date", "site_id"])
        .apply(lambda df: df.nsmallest(top_k, "distance_station"))
        .reset_index(drop=True)
    )
    snotel_features["weight"] = 1 / snotel_features["distance_station"]

    stations_count = (
        snotel_features.groupby(["issue_date", "site_id"])["stationTriplet"]
        .nunique()
        .rename("stations_count")
    )

    if not aggregate_stations:
        return snotel_features

    if separate_stations:
        snotel_features["in_basin"] = snotel_features["in_basin"].map(
            {False: "out-basin", True: "in-basin"}
        )
        snotel_features_agg = (
            (
                snotel_features.groupby(["issue_date", "site_id", "in_basin"])
                .apply(weighted_mean, cols=feature_cols, weight_col="weight")
                .pivot_table(index=["issue_date", "site_id"], columns=["in_basin"])
            )
            .merge(stations_count, on=["issue_date", "site_id"])
            .reset_index()
        )
        snotel_features_agg.columns = snotel_features_agg.columns.map("_".join)
    else:
        snotel_features_agg = (
            snotel_features.groupby(["issue_date", "site_id"])
            .apply(weighted_mean, cols=feature_cols + station_cols, weight_col="weight")
            .merge(stations_count, on=["issue_date", "site_id"])
            .reset_index()
        )

    return snotel_features_agg
