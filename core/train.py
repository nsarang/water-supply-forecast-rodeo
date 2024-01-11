from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, List

import catboost as cb
import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold

from core.data.preprocess import DataProcessor
from core.datasets import *
from core.models.catboost import train_catboost
from core.models.lgbm import train_lgbm
from core.utils import Vector


@dataclass
class InferenceEngine:
    model: Any
    data_processor: DataProcessor
    preprocess_fn: Callable = None

    def __call__(self, inputs):
        if self.preprocess_fn is not None:
            inputs = self.preprocess_fn(inputs)
        features, *_ = self.data_processor.preprocess_transform(inputs, mode="val")
        logits = self.model.predict(features)
        prediction = self.data_processor.postprocess_transform(features, logits, mode="val")
        return prediction

    predict = __call__


@dataclass
class Ensemble:
    models: List
    data_processors: List[DataProcessor] = None
    weights: List[float] = None
    feature_columns: List[str] = None

    def predict(self, inputs):
        predictions = []
        if self.feature_columns is not None:
            inputs = inputs[self.feature_columns]

        for idx, model in enumerate(self.models):
            if self.data_processors is not None:
                data_prep = self.data_processors[idx]
                features, *_ = data_prep.preprocess_transform(inputs, mode="val")
                logits = model.predict(features)
                prediction = data_prep.postprocess_transform(features, logits, mode="val")
            else:
                prediction = model.predict(inputs)
            predictions += [prediction]

        if self.weights is None:
            weights = [1] * len(self.models)
        else:
            weights = self.weights
        output = np.average(predictions, axis=0, weights=weights)
        return output


def train_util(dataset, cfg):
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

    sgkfold = StratifiedGroupKFold(n_splits=cfg["cv_splits"], shuffle=True, random_state=1234)
    logs = defaultdict(list)

    for idx, (train_index, test_index) in enumerate(sgkfold.split(X, cv_sites, cv_groups)):
        X_train, X_val = X.iloc[train_index].copy(), X.iloc[test_index].copy()
        y_train, y_val = y.iloc[train_index].copy(), y.iloc[test_index].copy()

        weight_train, weight_val = (
            combined_weight[train_index].copy(),
            combined_weight[test_index].copy(),
        )

        model, data_processor, error, cat_features = train_catboost(
            X_train, y_train, X_val, y_val, weight_train, weight_val
        )
        logs["catboost_raw_models"] += [model]
        logs["catboost_data_processor"] += [data_processor]
        logs["catboost_error"] += [error]

        model, data_processor, error, lgbm_features = train_lgbm(
            X_train, y_train, X_val, y_val, weight_train, weight_val
        )
        logs["lgbm_raw_models"] += [model]
        logs["lgbm_data_processor"] += [data_processor]
        logs["lgbm_error"] += [error]

    catboost_model = Ensemble(
        logs["catboost_raw_models"],
        logs["catboost_data_processor"],
        weights=1 / np.array(logs["catboost_error"]),
        feature_columns=cat_features,
    )
    lgbm_model = Ensemble(
        logs["lgbm_raw_models"],
        logs["lgbm_data_processor"],
        weights=1 / np.array(logs["lgbm_error"]),
        feature_columns=lgbm_features,
    )
    model = Ensemble(
        models=[catboost_model, lgbm_model],
        weights=[5, 1],
    )
    logs["model"] = model
    return logs
