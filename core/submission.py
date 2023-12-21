import json
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

import catboost as cb
import numpy as np
import pandas as pd
import recipes
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm.auto import tqdm

from core.data.feature import add_column_suffix, generate_timeseries_features
from core.data.preprocess import DataProcessor
from core.datasets import *
from core.utils import Vector

tqdm.pandas()


@dataclass
class Ensemble:
    models: List
    data_processors: List[DataProcessor]
    weights: List[float]
    feature_columns: List[str]

    def predict(self, inputs):
        inputs = inputs[self.feature_columns]
        predictions = []
        for model, data_prep in zip(self.models, self.data_processors):
            features, *_ = data_prep.preprocess_transform(inputs, mode="val")
            logits = model.predict(features)
            prediction = data_prep.postprocess_transform(features, logits, mode="val")
            predictions += [prediction]
        output = np.average(predictions, axis=0, weights=self.weights)
        return output


def prepare_dataset(month, day, df_monthly_flow, df_targets=None):
    historical_years = df_monthly_flow["forecast_year"].unique()

    historical_issue_dates = pd.Series(
        [datetime(year, month, day) for year in historical_years], name="issue_date"
    )

    dataset = (
        get_metadata()
        .rename(columns={"area": "basins_area"})
        .merge(historical_issue_dates, how="cross")
        .query("issue_date.dt.month <= season_end_month")
    )
    dataset["forecast_year"] = dataset["issue_date"].dt.year
    dataset["issue_month"] = dataset["issue_date"].dt.month

    if df_targets is not None:
        dataset = (
            dataset.merge(
                df_targets,
                left_on=["site_id", "forecast_year"],
                right_on=["site_id", "forecast_year"],
            )
            # .dropna()
        )

    # --- FEATURE ENGINEERING ---
    # - VOLUME FEATURES -
    df_monthly_flow = df_monthly_flow.copy()
    df_monthly_flow["date"] = df_monthly_flow[["year", "month"]].apply(
        lambda r: datetime(r["year"], r["month"], 1), axis=1
    )

    volume_features = dataset.progress_apply(
        generate_timeseries_features,
        df_b=df_monthly_flow,
        condition=(
            "site_id == @row.site_id"
            " & forecast_year == @row.forecast_year"
            " & ~(date.dt.year == @row.issue_date.year & date.dt.month >= @row.issue_date.month)"
        ),
        date_col="date",
        feature_engineering=["volume"],
        axis=1,
    ).drop(columns="volume")
    assert len(volume_features) == len(dataset)
    dataset = pd.concat(add_column_suffix(dataset, volume_features, suffixes=("", "_vol")), axis=1)

    # - TELECONN FEATURES -
    teleconn_data = get_all_teleconnections_data()
    teleconn_feature_cols = {
        # "MJO": ["INDEX_2", "INDEX_5"], # ...
        "PDO": ["pdo_index"],
        # "NINO": ["NINO3.4 ANOM", "NINO3 ANOM", "NINO4"], # ...
        "ONI": ["ANOM"],  # TOTAL
        "PNA": ["pna_index"],
        "SOI": ["soi"],
    }
    feature_suffixes = [
        "_roll_std7",
        "_roll_mean5",
        "_ewm_mean7",
        "_autocorr",
        "_roll_max5",
        "_roll_min5",
    ]
    for name, df_dataset in teleconn_data.items():
        df_feats = dataset.progress_apply(
            generate_timeseries_features,
            df_b=df_dataset,
            condition=(
                "date.dt.year < @row.issue_date.year"
                " | (date.dt.year == @row.issue_date.year & date.dt.month < @row.issue_date.month)"
            ),
            date_col="date",
            feature_engineering=teleconn_feature_cols.get(name, []),
            axis=1,
        )
        dataset = pd.concat(
            add_column_suffix(dataset, df_feats, suffixes=("", "_" + name.lower())), axis=1
        )

    dataset["yr_in_2yr_cycle"] = (dataset.issue_date.dt.year - 1800) % 2
    dataset["yr_in_3yr_cycle"] = (dataset.issue_date.dt.year - 1800) % 3
    dataset["yr_in_4yr_cycle"] = (dataset.issue_date.dt.year - 1800) % 4

    # Fourier series components for N-year seasonality
    num_terms = 1  # Number of Fourier terms
    for n in range(1, num_terms + 1):
        for cycle in [2, 3, 4]:
            dataset[f"sin_yr{cycle}_{n}"] = np.sin(
                2 * np.pi * n * dataset[f"yr_in_{cycle}yr_cycle"] / cycle
            )
            dataset[f"cos_yr{cycle}_{n}"] = np.cos(
                2 * np.pi * n * dataset[f"yr_in_{cycle}yr_cycle"] / cycle
            )

    # - EXTRA FEATURES -
    pdsi_dfs, snotel_dfs, cpc_dfs, cdec_dfs, grace_dfs = [[] for _ in range(5)]

    for issue_date in tqdm(historical_issue_dates):
        pdsi_dfs.append(
            get_pdsi_features(issue_date, radius1_km=50, radius2_km=200).assign(
                issue_date=issue_date
            )
        )
        snotel_dfs.append(
            get_snotel_features(issue_date, lookback=15).assign(issue_date=issue_date)
        )
        cpc_dfs.append(get_cpc_data(issue_date).assign(issue_date=issue_date))
        cdec_dfs.append(
            get_cdec_data(issue_date, lookback=30, max_distance=70).assign(issue_date=issue_date)
        )
        # grace_dfs.append(get_grace_features(issue_date, radius1_km=50, radius2_km=200).assign(issue_date=issue_date))

    dataset = (
        dataset.merge(pd.concat(snotel_dfs), on=["site_id", "issue_date"], how="left")
        .merge(pd.concat(cpc_dfs), on=["site_id", "issue_date"], how="left")
        .merge(pd.concat(pdsi_dfs), on=["site_id", "issue_date"], how="left")
        .merge(pd.concat(cdec_dfs), on=["site_id", "issue_date"], how="left")
        # .merge(pd.concat(grace_dfs), on=["site_id", "issue_date"], how="left")
    )

    return dataset


def train(dataset, cfg):
    dataset = dataset.copy()
    dataset = dataset[dataset["forecast_year"] >= cfg["forecast_year_cutoff"]]
    dataset = dataset[dataset["volume"] > 0]

    # -- FY WEIGHT --
    start_year = cfg["year_weights"]["start_year"]
    fy_weight = np.full(len(dataset), 1)
    fy_weight[dataset["forecast_year"] >= start_year] = (
        cfg["year_weights"]["scale"]
        * dataset["forecast_year"].nunique()
        / (dataset["forecast_year"].unique() >= start_year).sum()
    )

    # -- VOLUME WEIGHT --
    instance_weight = (
        dataset[["site_id", "volume"]]
        .merge(
            dataset.groupby("site_id")["volume"].median().rename("__average_volume__"),
            on="site_id",
            how="left",
            sort=False,
        )["__average_volume__"]
        .to_numpy()
    )
    instance_weight = cfg["volume_weights"]["baseline"] + cfg["volume_weights"]["scale"] * (
        instance_weight - instance_weight.min()
    ) / (instance_weight.max() - instance_weight.min())

    combined_weight = fy_weight * instance_weight

    cv_sites = dataset["site_id"].astype("category").cat.codes
    cv_groups = (
        (dataset["site_id"] + "_" + (dataset["issue_date"].dt.year // 10).astype("str"))
        .astype("category")
        .cat.codes
    )

    # I/O
    X = dataset[cfg["features"]].copy()
    y = dataset["volume"].copy()

    custom_feature_weights = {
        "site_id": 1,
        # "elevation": 1,
        # "V": 1,
    }

    feature_weights = np.full(X.shape[1], 1.0)
    for k, v in custom_feature_weights.items():
        feature_weights[X.columns.get_loc(k)] = v

    default_model_params = dict(
        # n_estimators=1000,
        learning_rate=0.03,
        depth=8,
        colsample_bylevel=0.6,
        subsample=0.66,
        l2_leaf_reg=0,
        random_strength=0.5,
        feature_weights=feature_weights,
        silent=True,
    )

    sgkfold = StratifiedGroupKFold(n_splits=cfg["cv_splits"], shuffle=True, random_state=1234)
    logs = defaultdict(list)

    for idx, (train_index, test_index) in enumerate(sgkfold.split(X, cv_sites, cv_groups)):
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_val = y.iloc[train_index].copy(), y.iloc[test_index].copy()

        weight_train, weight_val = (
            combined_weight[train_index].copy(),
            combined_weight[test_index].copy(),
        )

        data_processor = DataProcessor(
            normalization_configuration={
                "__target__": "md",
                **{c: "md" for c in X_train.columns if c.startswith("volume_roll_mean5")},
                **{
                    c: "std"
                    for c in X_train.columns
                    if (
                        c
                        not in [
                            "site_id",
                            "issue_month",
                            "latitude",
                            "longitude",
                            "elevation",
                            "basins_area",
                        ]
                    )
                    and (not c.startswith("volume_roll_mean5"))
                },
            }
        )

        data_processor.preprocess_fit(X_train, y_train)
        X_train, y_train, weight_train = data_processor.preprocess_transform(
            X_train, y_train, weight_train, mode="train"
        )
        X_val, y_val, weight_val = data_processor.preprocess_transform(
            X_val, y_val, weight_val, mode="val"
        )

        train_pool = cb.Pool(
            X_train, y_train, weight=weight_train, cat_features=["site_id", "issue_month"]
        )
        val_pool = cb.Pool(
            X_val, y_val, weight=weight_val, cat_features=["site_id", "issue_month"]
        )

        model = Vector(
            [
                cb.CatBoostRegressor(**default_model_params, loss_function="MAE"),
                cb.CatBoostRegressor(**default_model_params, loss_function="Quantile:alpha=0.1"),
                cb.CatBoostRegressor(**default_model_params, loss_function="Quantile:alpha=0.9"),
            ]
        )

        model.fit(train_pool, eval_set=[train_pool, val_pool], early_stopping_rounds=100)

        y_train_pred = np.array(model.predict(X_train))
        y_val_pred = np.array(model.predict(X_val))

        y_train_pred = data_processor.postprocess_transform(X_train, y_train_pred)
        y_val_pred = data_processor.postprocess_transform(X_val, y_val_pred)

        logs["train_index"] += [X_train.index]
        logs["val_index"] += [X_val.index]
        logs["train_pred"] += [y_train_pred]
        logs["val_pred"] += [y_val_pred]
        logs["data_processor"] += [data_processor]
        logs["raw_models"] += [model]

    model = Ensemble(
        logs["raw_models"],
        logs["data_processor"],
        weights=[1] * len(logs["raw_models"]),
        feature_columns=cfg["features"],
    )
    logs["model"] = model
    return logs


def train_models(month, day, data_dir, preprocessed_dir, assets):
    data_dir = Path(data_dir)
    df_train_monthly = pd.read_csv(data_dir / "train_monthly_naturalized_flow.csv")
    df_test_monthly = pd.read_csv(data_dir / "test_monthly_naturalized_flow.csv")
    df_train_targets = pd.read_csv(data_dir / "train.csv").rename(
        columns={"year": "forecast_year"}
    )

    train_features = prepare_dataset(month, day, df_train_monthly, df_train_targets)
    test_features = prepare_dataset(month, day, df_test_monthly)
    configuration = recipes.mvp.cfg
    logs = train(train_features, configuration)

    label = (month, day)
    assets[label] = {
        "train_features": train_features,
        "test_features": test_features,
        "model": logs["model"],
        "logs": logs,
    }


def run_inference(models, features):
    output_raw = np.array(Vector(models).predict(features))
    output = output_raw.mean(axis=0)
    return output


def preprocess(src_dir: Path, data_dir: Path, preprocessed_dir: Path) -> dict[Hashable, Any]:
    pass


def predict(
    site_id: str,
    issue_date: str,
    assets: dict[Any, Any],
    src_dir: Path,
    data_dir: Path,
    preprocessed_dir: Path,
) -> tuple[float, float, float]:
    issue_date = pd.to_datetime(issue_date)
    month, day = issue_date.month, issue_date.day
    label = (month, day)

    if label not in assets:
        train_models(month, day, data_dir, preprocessed_dir, assets)

    features = assets[label]["test_features"].query(
        f"site_id == '{site_id}' & issue_date == '{issue_date}'"
    )
    model = assets[label]["model"]
    output = model.predict(features)
    return output
