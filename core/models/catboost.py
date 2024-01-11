import os
import tempfile

import catboost as cb
import numpy as np
import pandas as pd

from core.data.preprocess import DataProcessor
from core.objectives.pinball import pinball_loss
from core.utils import Vector


class CatBoostModelWrapper:
    def __init__(self, model=None):
        self.model = model

    def __getstate__(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.model.save_model(tmp.name, format="cbm")
            with open(tmp.name, "rb") as f:
                state = f.read()
        os.remove(tmp.name)
        return state

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with open(tmp.name, "wb") as f:
                f.write(state)
            self.model = cb.CatBoost()
            self.model.load_model(tmp.name, format="cbm")
        os.remove(tmp.name)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name == "model":
            object.__setattr__(self, name, value)
        else:
            setattr(self.model, name, value)


def train_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    weight_train,
    weight_val,
    custom_feature_weights=None,
    **override_params
):
    custom_feature_weights = custom_feature_weights or {}
    feature_weights = np.full(X_train.shape[1], 1.0)
    for k, v in custom_feature_weights.items():
        feature_weights[X_train.columns.get_loc(k)] = v

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
    params = dict(default_model_params, **override_params)

    data_processor = DataProcessor(
        normalization_configuration={
            "__target__": "std",
            **{c: "md" for c in X_train.columns if c.startswith("volume")},
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
                and (not c.startswith("volume"))
            },
        }
    )

    data_processor.preprocess_fit(X_train, y_train)
    X_train_, y_train_, weight_train_ = data_processor.preprocess_transform(
        X_train, y_train, weight_train, mode="train"
    )
    X_val_, y_val_, weight_val_ = data_processor.preprocess_transform(
        X_val, y_val, weight_val, mode="val"
    )

    train_pool = cb.Pool(
        X_train_, y_train_, weight=weight_train_, cat_features=["site_id", "issue_month"]
    )
    val_pool = cb.Pool(X_val_, y_val_, weight=weight_val_, cat_features=["site_id", "issue_month"])

    model = Vector(
        [
            CatBoostModelWrapper(
                cb.CatBoostRegressor(**params, loss_function="Quantile:alpha=0.1")
            ),
            CatBoostModelWrapper(cb.CatBoostRegressor(**params, loss_function="MAE")),
            CatBoostModelWrapper(
                cb.CatBoostRegressor(**params, loss_function="Quantile:alpha=0.9")
            ),
        ]
    )

    model.fit(train_pool, eval_set=[train_pool, val_pool], early_stopping_rounds=100)

    y_train_pred = np.array(model.predict(X_train_))
    y_val_pred = np.array(model.predict(X_val_))

    y_train_pred = data_processor.postprocess_transform(X_train_, y_train_pred)
    y_val_pred = data_processor.postprocess_transform(X_val_, y_val_pred)

    error_val = (
        2
        * np.stack(
            [
                pinball_loss(y_val, y_pred, alpha=q)
                for y_pred, q in zip(y_val_pred, [0.1, 0.5, 0.9])
            ],
            axis=1,
        ).mean()
    )
    return model, data_processor, error_val, X_train_.columns
