from functools import reduce
from operator import and_

import numpy as np
import pandas as pd


def generate_timeseries_features(row, df_b, condition, date_col, feature_engineering=[]):
    df_prior = df_b.query(condition).sort_values(date_col)
    if df_prior.empty:
        return pd.Series(index=df_b.columns)

    df_slice = df_prior.iloc[-100:].copy()
    for col in feature_engineering:
        df_slice[col + "_roll_max5"] = df_slice[col].rolling(window=5, min_periods=1).max()
        df_slice[col + "_roll_min5"] = df_slice[col].rolling(window=5, min_periods=1).min()
        df_slice[col + "_roll_mean5"] = df_slice[col].rolling(window=5, min_periods=1).mean()
        df_slice[col + "_roll_std7"] = df_slice[col].rolling(window=7, min_periods=2).std()
        df_slice[col + "_ewm_mean7"] = df_slice[col].ewm(span=7).mean()
        df_slice[col + "_autocorr"] = df_slice[col].corr(df_slice[col].shift(1), min_periods=3)

    result = df_slice.iloc[-1]
    result.name = row.name
    return result


def generate_timeseries_features_v1_5(df, date_col, features, enrich=True):
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.sort_values(date_col).iloc[-100:].copy()

    columns = features.copy()
    if enrich:
        for col in features:
            df[col + "_roll_max5"] = df[col].rolling(window=5, min_periods=1).max()
            df[col + "_roll_min5"] = df[col].rolling(window=5, min_periods=1).min()
            df[col + "_roll_mean5"] = df[col].rolling(window=5, min_periods=1).mean()
            df[col + "_roll_std7"] = df[col].rolling(window=7, min_periods=2).std()
            df[col + "_ewm_mean7"] = df[col].ewm(span=7).mean()
            df[col + "_autocorr"] = df[col].corr(df[col].shift(1), min_periods=3)
            df[col + "_count_7"] = df[col].iloc[-7:].count()
            columns += [
                col + suffix
                for suffix in [
                    "_roll_max5",
                    "_roll_min5",
                    "_roll_mean5",
                    "_roll_std7",
                    "_ewm_mean7",
                    "_autocorr",
                    "_count_7",
                ]
            ]
    # result = df[columns].iloc[-1]
    result = df.iloc[-1]
    return result


def generate_timeseries_features_v2(df, date_col, features, enrichments=True, max_rows=100):
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.sort_values(date_col).iloc[-max_rows:].copy()

    columns = features.copy()
    for col, gen_cfg in enrichments.items():
        for output_suffix, (fn_name, agg_fn, agg_kwargs) in gen_cfg.items():
            if fn_name == "autocorr":
                result = df[col].corr(df[col].shift(1), min_periods=3)
            elif fn_name == "count":
                result = df[col].iloc[-7:].count()
            else:
                main_fn = getattr(df[col], fn_name)
                result = getattr(main_fn(**agg_kwargs), agg_fn)()

            output_col = col + "_" + output_suffix
            df[output_col] = result
            columns += [output_col]
    # result = df[columns].iloc[-1]
    result = df.iloc[-1]
    return result


def generate_timeseries_features_v3(df, date_col, features=[], enrichments=True, max_rows=100):
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    df = df.sort_values(date_col).iloc[-max_rows:].copy()

    columns = features.copy()
    for col, gen_cfg in enrichments.items():
        for output_suffix, feat_fn in gen_cfg.items():
            result = feat_fn(df[col])
            output_col = col + "_" + output_suffix
            df[output_col] = result
            columns += [output_col]
    result = df[columns].iloc[-1]
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
