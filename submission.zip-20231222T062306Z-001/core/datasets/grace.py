import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np

from core.config import GRACE_DIR
from .metadata import get_metadata


METADATA_DF = get_metadata()


def get_grace_features(issue_date, radius1_km=100, radius2_km=300):
    forecast_year = issue_date.year
    year_data_dir = GRACE_DIR / f"FY{forecast_year}"
    files = list(year_data_dir.glob("*.nc4"))
    feature_cols = ["gws_inst", "rtzsm_inst", "sfsm_inst"]
    empty_df = pd.DataFrame(columns=feature_cols, index=METADATA_DF["site_id"])

    if len(files) == 0:
        return empty_df

    xds = xr.open_mfdataset(files)

    if len(files) == 0:
        return empty_df

    xds = xr.open_mfdataset(files)
    dates = xds["time"].to_series()
    prior_dates = dates[dates < issue_date]
    if len(prior_dates) == 0:
        return empty_df

    date_index = prior_dates.argmax()
    grace_df = xds.isel(time=date_index).to_dataframe().reset_index()

    sites_gdf = gpd.GeoDataFrame(
        METADATA_DF[["site_id", "latitude", "longitude"]],
        geometry=gpd.points_from_xy(METADATA_DF.longitude, METADATA_DF.latitude),
        crs="EPSG:4326",
    ).to_crs("ESRI:102004")

    # Define the radius in meters
    radius1 = radius1_km * 1000
    radius2 = radius2_km * 1000

    sites_buffer1_gdf, sites_buffer2_gdf = sites_gdf.copy(), sites_gdf.copy()

    sites_buffer1_gdf["geometry"] = sites_gdf.geometry.buffer(radius1)
    sites_buffer2_gdf["geometry"] = sites_gdf.geometry.buffer(radius2).difference(
        sites_gdf.geometry.buffer(radius1)
    )

    grace_gdf = (
        gpd.GeoDataFrame(
            grace_df,
            geometry=gpd.points_from_xy(grace_df.lon, grace_df.lat),
            crs="EPSG:4326",
        ).to_crs("ESRI:102004")
    ).replace(-999.0, np.nan)

    # Perform the spatial join
    joined1_gdf = gpd.sjoin(grace_gdf, sites_buffer1_gdf, predicate="within")
    joined2_gdf = gpd.sjoin(grace_gdf, sites_buffer2_gdf, predicate="within")

    grace_features = pd.merge(
        joined1_gdf.groupby("site_id")[feature_cols].mean(),
        joined2_gdf.groupby("site_id")[feature_cols].mean(),
        on="site_id",
        suffixes=("_1", "_2"),
    )

    return grace_features
