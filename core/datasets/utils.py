from copy import deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd
from netCDF4 import num2date


def weighted_mean(df, cols, weight_col):
    tr = lambda x: np.nan if x is np.ma.masked else x  # in case all values are nan
    return pd.Series(
        {
            col: tr(
                np.ma.average(
                    np.ma.MaskedArray(df[col], mask=np.isnan(df[col])), weights=df[weight_col]
                )
            )
            for col in cols
        }
    )


def time2date(time_var):
    time_units = time_var.units
    time_calendar = time_var.calendar if hasattr(time_var, "calendar") else "standard"

    # Convert to cftime objects
    cftime_dates = num2date(
        time_var[:], units=time_units, calendar=time_calendar, only_use_cftime_datetimes=False
    )
    return cftime_dates


def copying_lru_cache(maxsize=100, typed=False):
    def decorator(f):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(f)

        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))

        return wrapper

    return decorator
