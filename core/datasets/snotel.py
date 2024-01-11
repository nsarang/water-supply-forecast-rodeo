from datetime import timedelta

import pandas as pd

from core.config import SNOTEL_DIR
from core.data.feature import generate_timeseries_features_v3
from core.data.geometry import haversine_distance
from core.datasets.metadata import get_metadata
from core.datasets.utils import weighted_mean

STATIONS_METADATA_DF = pd.read_csv(SNOTEL_DIR / "station_metadata.csv")
STATIONS_MAPPING_DF = pd.read_csv(SNOTEL_DIR / "sites_to_snotel_stations.csv")


def get_snotel_features(
    issue_date,
    lookback=15,
    max_distance_km=200,
    separate_stations=False,
    add_features=False,
    df_metadata=None,
    aggregate_stations=True,
):
    if df_metadata is None:
        df_metadata = get_metadata()
    issue_date = pd.to_datetime(issue_date)
    forecast_year = issue_date.year
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

    snotel_df_filtered = snotel_df[
        snotel_df["date"].between(issue_date - timedelta(lookback), issue_date - timedelta(1))
    ]

    if add_features:
        efns = {
            "roll_max": lambda c: c.rolling(window=lookback, min_periods=1).max(),
            # "roll_min": lambda c: c.rolling(window=lookback, min_periods=1).min(),
            "roll_mean": lambda c: c.rolling(window=lookback, min_periods=1).mean(),
            "roll_std": lambda c: c.rolling(window=lookback, min_periods=2).std(),
            "autocorr": lambda c: c.corr(c.shift(1), min_periods=3),
        }
        snotel_df_agg = snotel_df_filtered.groupby("snotel_site").apply(
            generate_timeseries_features_v3,
            date_col="date",
            features=primary_cols,
            enrichments={"WTEQ_DAILY": efns, "PREC_DAILY": efns},
        )
        feature_cols = primary_cols + [
            c + "_" + suffix for c in ["WTEQ_DAILY", "PREC_DAILY"] for suffix in efns
        ]
    else:
        feature_cols = primary_cols
        snotel_df_agg = snotel_df_filtered.groupby("snotel_site")[feature_cols].mean()  # over time

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
    snotel_features = snotel_features.query(
        f"elevation_station >= elevation_site & distance_station < {max_distance_km}"
    ).copy()
    snotel_features["weight"] = 1 / snotel_features["distance_station"]
    stations_count = (
        snotel_features.groupby(["site_id"])["stationTriplet"].nunique().rename("stations_count")
    )

    if not aggregate_stations:
        return snotel_features

    if separate_stations:
        snotel_features["in_basin"] = snotel_features["in_basin"].map(
            {False: "out-basin", True: "in-basin"}
        )
        snotel_features_agg = (
            (
                snotel_features.groupby(["site_id", "in_basin"])
                .apply(weighted_mean, cols=feature_cols, weight_col="weight")
                .pivot_table(index=["site_id"], columns=["in_basin"])
            )
            .merge(stations_count, on="site_id")
            .reset_index()
        )
        snotel_features_agg.columns = snotel_features_agg.columns.map("_".join)
    else:
        snotel_features_agg = (
            snotel_features.groupby(["site_id"])
            .apply(weighted_mean, cols=feature_cols + station_cols, weight_col="weight")
            .merge(stations_count, on="site_id")
            .reset_index()
        )

    return snotel_features_agg
