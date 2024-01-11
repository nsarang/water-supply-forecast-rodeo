from datetime import timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from core.config import CDEC_DIR, CDEC_SNOW_STATIONS_FILE
from core.data.geometry import haversine_distance
from core.data.outlier import iqr_outliers

from .metadata import get_metadata
from .utils import weighted_mean

METADATA_DF = get_metadata()


def remove_cdec_outliers(df, **kwargs):
    return df.loc[iqr_outliers(df["value"], **kwargs)]


def process_cdec_station_metadata() -> gpd.GeoDataFrame:
    """
    Load and process CDEC stations metadata into a GeoDataFrame.

    Metadata for all CDEC stations that collect snow data is available on the
    data download page as `cdec_snow_stations.csv`, and should be saved to
    `data/cdec_snow_stations.csv`.

    The metadata includes the union of stations that have a sensor for snow water equivalent
    (sensor number 3) and for snow depth (sensor number 18), downloaded using the CDEC Station
    Search web application.
    https://cdec.water.ca.gov/dynamicapp/staSearch?sta=&sensor_chk=on&sensor=18&collect=NONE+SPECIFIED&dur=&active=&lon1=&lon2=&lat1=&lat2=&elev1=-5&elev2=99000&nearby=&basin=NONE+SPECIFIED&hydro=NONE+SPECIFIED&county=NONE+SPECIFIED&agency_num=160&display=sta
    https://cdec.water.ca.gov/dynamicapp/staSearch?sta=&sensor_chk=on&sensor=3&collect=NONE+SPECIFIED&dur=&active=&lon1=&lon2=&lat1=&lat2=&elev1=-5&elev2=99000&nearby=&basin=NONE+SPECIFIED&hydro=NONE+SPECIFIED&county=NONE+SPECIFIED&agency_num=160&display=sta
    """
    # Load CDEC station metadata
    cdec_stations = pd.read_csv(CDEC_SNOW_STATIONS_FILE)
    cdec_stations.columns = [c.lower().replace(" ", "_") for c in cdec_stations.columns]
    cdec_stations = cdec_stations.rename(columns={"id": "station_id"})

    # Get geodataframe
    cdec_gdf = gpd.GeoDataFrame(
        cdec_stations,
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(cdec_stations["longitude"], cdec_stations["latitude"])
        ],
        crs="WGS84",
    )

    cdec_gdf["elevation"] = cdec_gdf["elevationfeet"].apply(lambda x: float(x.replace(",", "")))
    cdec_gdf = cdec_gdf.drop(columns="elevationfeet")
    return cdec_gdf


def get_cdec_data(issue_dates, lookback=30, max_distance_km=150, top_k=10, df_metadata=None):
    if df_metadata is None:
        df_metadata = METADATA_DF

    issue_dates = pd.to_datetime(pd.Series(issue_dates))
    assert issue_dates.dt.year.nunique() == 1
    forecast_year = issue_dates.dt.year.unique()[0]

    fy_data_dir = CDEC_DIR / f"FY{forecast_year}"
    files = list(fy_data_dir.glob("*.csv"))
    feature_cols = ["SNOW DP", "SNOW WC", "RAIN", "TEMP AV"]
    station_cols = [
        "elevation_station",
        "latitude_station",
        "longitude_station",
        "distance_station",
    ]
    # empty_df = pd.DataFrame(columns=feature_cols)
    empty_df = pd.DataFrame()

    if len(files) == 0:
        return empty_df

    cdec_station_metadata = process_cdec_station_metadata()
    cdec_station_metadata = cdec_station_metadata
    # sites_to_cdec_stations_metadata = pd.read_csv(CDEC_DIR / "sites_to_cdec_stations.csv")

    cdec_sites_metadata = (
        df_metadata[["site_id", "latitude", "longitude", "elevation"]]
        # .merge(sites_to_cdec_stations_metadata, on="site_id", suffixes=("_site", "_station"))
        .merge(
            cdec_station_metadata,
            # on="station_id",
            suffixes=("_site", "_station"),
            # how="inner",
            how="cross",
        )
    )

    cdec_df = pd.concat(
        (pd.read_csv(fp).assign(station=fp.stem.replace("_", ":")) for fp in files),
        ignore_index=True,
    ).replace(-9999, np.nan)

    cdec_df["date"] = pd.to_datetime(cdec_df["date"])

    cdec_df = cdec_df.merge(issue_dates.rename("issue_date"), how="cross")
    cdec_df = cdec_df[
        cdec_df["date"].between(
            cdec_df["issue_date"] - timedelta(lookback), cdec_df["issue_date"] - timedelta(1)
        )
    ]

    cdec_df_prep = (
        cdec_df.groupby(["issue_date", "stationId", "sensorType"])
        .apply(remove_cdec_outliers, scale=2)
        .reset_index(drop=True)
    )
    cdec_df_agg = (
        cdec_df_prep.groupby(["issue_date", "stationId", "sensorType"])["value"]
        .agg(["mean", "count"])
        .reset_index()
    )
    if len(cdec_df_agg) == 0:
        return empty_df

    cdec_site_features = cdec_df_agg.rename(columns={"stationId": "station_id"}).merge(
        cdec_sites_metadata, on="station_id"
    )
    cdec_site_features["distance_station"] = haversine_distance(cdec_site_features)
    cdec_site_features = (
        cdec_site_features.query(
            f"count >= 0.9 * {lookback}"
            "& elevation_station >= elevation_site"
            f"& distance_station <= {max_distance_km}"
        )
        .groupby(["issue_date", "site_id"])
        .apply(lambda df: df.nsmallest(top_k, "distance_station"))
        .reset_index(drop=True)
    )

    if cdec_site_features.empty:
        return empty_df

    cdec_site_features["weight"] = 1 / cdec_site_features["distance_station"]
    cdec_site_features_agg = cdec_site_features.groupby(
        ["issue_date", "site_id", "sensorType"]
    ).apply(weighted_mean, cols=["mean", "count"] + station_cols, weight_col="weight")
    cdec_site_features_pivot = cdec_site_features_agg.pivot_table(
        index=["issue_date", "site_id"], columns=["sensorType"]
    ).rename(columns={"mean": "value"})
    cdec_site_features_pivot.columns = cdec_site_features_pivot.columns.map("_".join)
    cdec_site_features_pivot = cdec_site_features_pivot.reset_index()
    return cdec_site_features_pivot


if __name__ == "__main__":
    df = get_cdec_data(["2020-04-01", "2020-04-15"], lookback=30, max_distance_km=150, top_k=10)
    print(df)
