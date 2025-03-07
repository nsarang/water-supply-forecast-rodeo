import pandas as pd
from datetime import datetime, timedelta
from core.config import MJO_FILE, NINO_FILE, ONI_FILE, PDO_FILE, PNA_FILE, SOI_FILE
from wsfr_read.teleconnections import (
    read_mjo_data,
    read_nino_regions_sst_data,
    read_oni_data,
    read_pdo_data,
    read_pna_data,
    read_soi_data,
)

MAX_DATE = "2100-01-01"


def get_all_teleconnections_data():
    return {
        "MJO": get_mjo_data(),
        "NINO": get_nino_data(),
        "ONI": get_oni_data(),
        "PDO": get_pdo_data(),
        "PNA": get_pna_data(),
        "SOI": get_soi_data(),
    }


def get_mjo_data():
    mjo_data = read_mjo_data(issue_date=MAX_DATE, path=MJO_FILE).rename(
        columns={"DATE": "date"}
    )
    return mjo_data


def get_nino_data():
    nino_data = read_nino_regions_sst_data(issue_date=MAX_DATE, path=NINO_FILE)
    nino_data["date"] = pd.to_datetime(
        nino_data["YR"].astype("str") + "-" + nino_data["MON"].astype("str"), format="%Y-%m"
    )
    return nino_data


def get_oni_data():
    oni_data = read_oni_data(issue_date=MAX_DATE, path=ONI_FILE)
    months = [datetime(2000, i + 1, 1).strftime("%B") for i in range(12)]
    serial = "".join(map(lambda x: x[0], months))
    oni_data["date"] = oni_data[["SEAS", "YR"]].apply(
        lambda r: datetime(
            year=r["YR"], month=((serial + serial).find(r["SEAS"]) + 3) % len(serial) + 1, day=1
        )
        - timedelta(1),
        axis=1,
    )
    return oni_data


def get_pdo_data():
    pdo_data = read_pdo_data(issue_date=MAX_DATE, path=PDO_FILE)
    pdo_data["date"] = pdo_data[["year", "month"]].apply(
        lambda r: datetime(r["year"], r["month"], 1), axis=1
    )
    return pdo_data


def get_pna_data():
    pna_data = read_pna_data(issue_date=MAX_DATE, path=PNA_FILE)
    pna_data["date"] = pna_data[["year", "month"]].apply(
        lambda r: datetime(r["year"], r["month"], 1), axis=1
    )
    return pna_data


def get_soi_data():
    soi_data = read_soi_data(issue_date=MAX_DATE, path=SOI_FILE)
    soi_data["date"] = soi_data[["year", "month"]].apply(
        lambda r: datetime(r["year"], r["month"], 1), axis=1
    )
    return soi_data
