import numpy as np
import pandas as pd
import geopandas as gpd

from core.data.outlier import iqr_outliers
from .utils import haversine_distance, weighted_mean
from core.config import CDEC_SNOW_STATIONS_FILE, CDEC_DIR
from .metadata import get_metadata


METADATA_DF = get_metadata()


def remove_cdec_outliers(df, **kwargs):
    return df.loc[iqr_outliers(df["value"], **kwargs)]


from shapely.geometry import Point


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

    return cdec_gdf


def get_cdec_data(issue_date):
    issue_date = pd.to_datetime(issue_date)
    forecast_year = issue_date.year
    fy_data_dir = CDEC_DIR / f"FY{forecast_year}"
    files = list(fy_data_dir.glob("*.csv"))
    feature_cols = ["SNOW DP", "SNOW WC", "RAIN", "TEMP AV"]
    empty_df = pd.DataFrame(columns=feature_cols)

    if len(files) == 0:
        return empty_df

    cdec_station_metadata = process_cdec_station_metadata()
    sites_to_cdec_stations_metadata = pd.read_csv(CDEC_DIR / "sites_to_cdec_stations.csv")

    cdec_sites_metadata = (
        METADATA_DF[["site_id", "latitude", "longitude"]]
        .merge(sites_to_cdec_stations_metadata, on="site_id", suffixes=("_site", "_station"))
        .merge(cdec_station_metadata, on="station_id", suffixes=("_site", "_station"))
    )

    cdec_df = pd.concat(
        (pd.read_csv(fp).assign(station=fp.stem.replace("_", ":")) for fp in files),
        ignore_index=True,
    ).replace(-9999, np.nan)
    cdec_df["date"] = pd.to_datetime(cdec_df["date"])
    cdec_df = cdec_df[cdec_df["date"] < issue_date]
    cdec_df_prep = (
        cdec_df.groupby(["stationId", "sensorType"])
        .apply(remove_cdec_outliers, scale=2)
        .reset_index(drop=True)
    )
    cdec_df_agg = (
        cdec_df_prep.groupby(["stationId", "sensorType"])["value"]
        .agg(["mean", "count"])
        .reset_index()
    )
    if len(cdec_df_agg) == 0:
        return empty_df

    cdec_site_features = cdec_df_agg.rename(columns={"stationId": "station_id"}).merge(
        cdec_sites_metadata, on="station_id"
    )
    cdec_site_features["weight"] = 1 / (haversine_distance(cdec_site_features))
    cdec_site_features_agg = cdec_site_features.groupby(["site_id", "sensorType"]).apply(
        weighted_mean, cols=["mean", "count"], weight_col="weight"
    )
    cdec_site_features_pivot = (
        cdec_site_features_agg.reset_index()
        .pivot_table(index="site_id", columns="sensorType")
        .loc[:, pd.IndexSlice["mean", :]]
        .droplevel(0, axis=1)
        .reindex(columns=feature_cols)
        .reset_index()
    )
    return cdec_site_features_pivot
