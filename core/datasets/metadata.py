import geopandas as gpd
import pandas as pd
from wsfr_read.sites import read_geospatial

from core.config import CPC_CLIMATE_DIVISIONS_GEO_FILE, DATA_ROOT, METADATA_FILE
from core.data.geometry import haversine_distance_coords

from .utils import copying_lru_cache


@copying_lru_cache(maxsize=None)
def get_metadata(filepath=None, supplements=True):
    if filepath is None:
        filepath = METADATA_FILE

    metadata = pd.read_csv(filepath)
    basins_data = read_geospatial("basins")

    # Supplementary sites
    if supplements:
        df_supp_metadata = get_nrcs_metadata()
        df_supp_metadata[df_supp_metadata["longitude"] >= -130]
        metadata = pd.concat((metadata, df_supp_metadata))

    # Add climate divisions
    cpc_cds = gpd.read_file(CPC_CLIMATE_DIVISIONS_GEO_FILE)
    metadata["cd_original"] = metadata["site_id"].apply(
        get_climate_divisions_for_site_id, basins_gdf=basins_data, climate_divisions_gdf=cpc_cds
    )
    metadata = metadata.merge(assigns_cds(metadata, read_climate_divisions()), on="site_id")

    # Add basin attributes
    # basins_data["area"] = basins_data["geometry"].to_crs("EPSG:5070").area / 1e6
    metadata = metadata.merge(basins_data, on="site_id", how="left")
    return metadata


def get_nrcs_metadata():
    df_supp_metadata = pd.read_csv(DATA_ROOT / "supplementary_nrcs_metadata.csv")
    df_supp_metadata["site_id"] = df_supp_metadata["nrcs_name"].transform(
        lambda v: "s_" + v.replace(" ", "_")
    )
    df_supp_metadata["season_start_month"] = 4
    df_supp_metadata["season_end_month"] = 7
    return df_supp_metadata


def read_climate_divisions():
    colspecs = [(0, 4), (5, 11), (12, 18), (19, 26), (27, 57), (58, 75), (76, -1)]
    colnames = [
        "Division",
        "Latitude",
        "Longitude",
        "Area(MI2)",
        "Description",
        "Division Name",
        "WMO ID",
    ]

    # Read the file
    cpc_divisions = pd.read_fwf(
        "data/cpc_outlooks/regdict.txt", colspecs=colspecs, header=None, skiprows=2, names=colnames
    )
    cpc_divisions["Longitude"] = -cpc_divisions["Longitude"]
    return cpc_divisions


def assigns_cds(metadata, cpc_divisions, max_distance_km=300):
    metadata_cpc = metadata[["site_id", "latitude", "longitude"]].merge(
        cpc_divisions.add_suffix("_cpc"), how="cross"
    )
    metadata_cpc["cd_distance"] = metadata_cpc[
        ["latitude", "longitude", "Latitude_cpc", "Longitude_cpc"]
    ].apply(lambda r: haversine_distance_coords(*r.values), axis=1)
    metadata_cds = (
        metadata_cpc.groupby("site_id")
        .apply(lambda df: df[df["cd_distance"] <= max_distance_km]["Division_cpc"].values)
        .to_frame("cd")
        .reset_index()
    )
    return metadata_cds


def get_climate_divisions_for_site_id(
    site_id: str, basins_gdf, climate_divisions_gdf
) -> list[int]:
    """Returns a list of CPC climate divisions that spatially intersect with a site_id's drainage
    basin."""
    if site_id not in basins_gdf["site_id"].unique():
        return None
    this_site_basin_gdf = basins_gdf.query(f"site_id == '{site_id}'")
    # Inner spatial join to get climate divisions intersecting this drainage basin
    this_site_divisions_gdf = climate_divisions_gdf.sjoin(
        this_site_basin_gdf[["geometry"]], how="inner"
    )
    return this_site_divisions_gdf["CD"].values.tolist()
