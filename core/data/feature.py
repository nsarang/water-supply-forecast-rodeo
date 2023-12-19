import pandas as pd
import numpy as np
from functools import reduce
from operator import and_


def generate_timeseries_features(row, df_b, condition, date_col, feature_engineering=[]):
    df_prior = df_b.query(condition).sort_values(date_col)
    if df_prior.empty:
        return pd.Series(index=df_b.columns)

    df_slice = df_prior.iloc[-100:].copy()
    for col in feature_engineering:
        df_slice[col + "_roll_max5"] = df_slice[col].rolling(window=5, min_periods=1).max()
        df_slice[col + "_roll_mean5"] = df_slice[col].rolling(window=5, min_periods=1).mean()
        df_slice[col + "_roll_std7"] = df_slice[col].rolling(window=7, min_periods=2).std()
        df_slice[col + "_ewm_mean7"] = df_slice[col].ewm(span=7).mean()
        df_slice[col + "_autocorr"] = df_slice[col].corr(df_slice[col].shift(1), min_periods=3)

    result = df_slice.iloc[-1]
    result.name = row.name
    return result


def add_column_suffix(df1, df2, suffixes=("_x", "_y"), common_cols_only=True):
    if common_cols_only:
        common_cols = df1.columns.intersection(df2.columns)
        df1_cols, df2_cols = common_cols, common_cols
    else:
        df1_cols, df2_cols = df1.columns, df2.columns
    df1 = df1.rename(columns={col: col + suffixes[0] for col in df1_cols})
    df2 = df2.rename(columns={col: col + suffixes[1] for col in df2_cols})
    return (df1, df2)


def find_closest_data(row, df_b, date_cols, on=[]):
    cond = reduce(and_, [row[col] == df_b[col] for col in on], np.full(len(df_b), True))
    df_b = df_b[cond].dropna()
    prior_data = df_b[row[date_cols[0]] > df_b[date_cols[1]]]
    if not prior_data.empty:
        closest_idx = (row[date_cols[0]] - prior_data[date_cols[1]]).idxmin()
        return df_b.loc[closest_idx]
    else:
        return pd.Series(index=df_b.columns)
