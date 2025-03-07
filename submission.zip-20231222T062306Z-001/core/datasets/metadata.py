import pandas as pd
from core.config import METADATA_FILE
from wsfr_read.climate.cpc_outlooks import get_climate_divisions_for_site_id
from wsfr_read.sites import read_geospatial
from .utils import copying_lru_cache


@copying_lru_cache(maxsize=None)
def get_metadata(filepath=None):
    if filepath is None:
        filepath = METADATA_FILE
    metadata = pd.read_csv(filepath)
    metadata["cd"] = metadata["site_id"].apply(get_climate_divisions_for_site_id)

    basins_data = read_geospatial("basins")
    # basins_data["area"] = basins_data["geometry"].to_crs("EPSG:5070").area / 1e6

    metadata = metadata.merge(basins_data, on="site_id")
    return metadata