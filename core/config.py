import os
from pathlib import Path

from loguru import logger

DATA_ROOT = Path(os.getenv("WSFR_DATA_ROOT", Path.cwd() / "data"))
METADATA_FILE = DATA_ROOT / "metadata.csv"
GEOSPATIAL_FILE = DATA_ROOT / "geospatial.gpkg"
CDEC_SNOW_STATIONS_FILE = DATA_ROOT / "cdec_snow_stations.csv"

GRACE_DIR = DATA_ROOT / "grace_indicators"
SNOTEL_DIR = DATA_ROOT / "snotel"
PDSI_DIR = DATA_ROOT / "pdsi"
CDEC_DIR = DATA_ROOT / "cdec"

MJO_FILE = DATA_ROOT / "teleconnections" / "mjo.txt"
NINO_FILE = DATA_ROOT / "teleconnections" / "nino_regions_sst.txt"
ONI_FILE = DATA_ROOT / "teleconnections" / "oni.txt"
PDO_FILE = DATA_ROOT / "teleconnections" / "pdo.txt"
PNA_FILE = DATA_ROOT / "teleconnections" / "pna.txt"
SOI_FILE = DATA_ROOT / "teleconnections" / "soi.txt"

logger.info(f"DATA_ROOT is {DATA_ROOT}")
