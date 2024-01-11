import gc
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import pandas as pd
import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.sql import DataFrame, SparkSession


def list_sub(l1: List, l2: List):
    """
    Subtract the second list from the first one while perserving the order of
    elemetns.
    """
    return [x for x in l1 if x not in l2]


def join_sdf(
    df_l: DataFrame,
    df_r: DataFrame,
    on: Union[str, List[str]] = None,
    how: str = "inner",
    suffixes: Tuple[str] = None,  # defaults to ("_l", "_r") if prefixes is None
    prefixes: Tuple[str] = None,
    left_on: Union[str, List[str]] = None,
    right_on: Union[str, List[str]] = None,
    drop_duplicates: Literal["left", "right", "both"] = None,
    keep_order: bool = True,
):
    if left_on or right_on:
        raise NotImplementedError

    result = df_l.alias("l").join(df_r.alias("r"), on=on, how=how)
    duplicates = list(set(df_l.columns).intersection(df_r.columns).difference([]))
    columns = [f.col(f"`{c}`") for c in list_sub(result.columns, duplicates)]

    if len(duplicates) > 0:
        if drop_duplicates == "left":
            columns += [f.col("l." + col) for col in duplicates]
        elif drop_duplicates == "right":
            columns += [f.col("r." + col) for col in duplicates]
        else:
            if not (suffixes or prefixes):
                suffixes = ("_l", "_r")
            left_affix, right_affix = suffixes or prefixes
            colfrmt = "{col}{affix}" if suffixes else "{affix}{col}"
            for col in duplicates:
                left_col = f.col("l." + col).alias(colfrmt.format(col=col, affix=left_affix))
                right_col = f.col("r." + col).alias(colfrmt.format(col=col, affix=right_affix))
                columns += [left_col, right_col]

    result = result.select(columns)
    return result


def to_sdf(df: pd.DataFrame):
    df = df.copy()
    date_columns = df.select_dtypes("datetime").columns
    for col in date_columns:
        df[col] = df[col].astype("str")
    sdf = default_session().createDataFrame(df)
    for col in date_columns:
        sdf = sdf.withColumn(col, f.col(col).cast("date"))
    return sdf


def clear_cache(spark=None):
    if spark is None:
        spark = default_session()
    for id, rdd in spark.sparkContext._jsc.getPersistentRDDs().items():
        rdd.unpersist()
    collect_garbages()


def collect_garbages(spark=None):
    gc.collect()
    if spark is None:
        spark = default_session()
    spark.sparkContext._jvm.System.gc()


def default_session(conf: Dict[str, Any] = None) -> SparkSession:
    """
    Returns an SparkSession
    """
    if conf is None:
        conf = dict()
    builder = SparkSession.builder.appName("just-another-spark-session")
    for key, value in conf.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()
