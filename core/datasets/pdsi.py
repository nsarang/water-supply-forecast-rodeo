import geopandas as gpd
import pandas as pd
import xarray as xr

from core.config import PDSI_DIR

from .metadata import get_metadata

METADATA_DF = get_metadata()


def get_pdsi_features(issue_date, radius1_km=100, radius2_km=300, df_metadata=None):
    if df_metadata is None:
        df_metadata = METADATA_DF
    issue_date = pd.to_datetime(issue_date)
    forecast_year = issue_date.year
    year_data_dir = PDSI_DIR / f"FY{forecast_year}"
    files = list(year_data_dir.glob("*.nc"))
    empty_df = pd.DataFrame(
        columns=["pdsi_1", "pdsi_2"], index=df_metadata["site_id"]
    ).reset_index()

    if len(files) == 0:
        return empty_df

    xds = xr.open_mfdataset(files)
    dates = xds["day"].to_series()
    prior_dates = dates[dates < issue_date]
    if len(prior_dates) == 0:
        return empty_df
    date_index = prior_dates.argmax()
    pdsi_df = xds.isel(day=date_index).to_dataframe().reset_index()

    sites_gdf = gpd.GeoDataFrame(
        df_metadata[["site_id", "latitude", "longitude"]],
        geometry=gpd.points_from_xy(df_metadata.longitude, df_metadata.latitude),
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")

    # Define the radius in meters
    radius1 = radius1_km * 1000
    radius2 = radius2_km * 1000

    sites_buffer1_gdf, sites_buffer2_gdf = sites_gdf.copy(), sites_gdf.copy()

    sites_buffer1_gdf["geometry"] = sites_gdf.geometry.buffer(radius1)
    sites_buffer2_gdf["geometry"] = sites_gdf.geometry.buffer(radius2).difference(
        sites_gdf.geometry.buffer(radius1)
    )

    pdsi_gdf = (
        gpd.GeoDataFrame(
            pdsi_df,
            geometry=gpd.points_from_xy(pdsi_df.lon, pdsi_df.lat),
            crs=xds.geospatial_bounds_crs,
        )
        .rename(columns={"daily_mean_palmer_drought_severity_index": "pdsi"})
        .to_crs("EPSG:5070")
    )

    # Perform the spatial join
    joined1_gdf = gpd.sjoin(pdsi_gdf, sites_buffer1_gdf, predicate="within")
    joined2_gdf = gpd.sjoin(pdsi_gdf, sites_buffer2_gdf, predicate="within")

    pdsi_features = pd.merge(
        joined1_gdf.groupby("site_id")["pdsi"].mean(),
        joined2_gdf.groupby("site_id")["pdsi"].mean(),
        on="site_id",
        suffixes=("_1", "_2"),
    ).reset_index()

    return pdsi_features
