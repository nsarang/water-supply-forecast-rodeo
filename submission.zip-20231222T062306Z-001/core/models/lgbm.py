import lightgbm as lgb
import os
import tempfile
import pandas as pd
from core.utils import Vector
from core.data.preprocess import DataProcessor
import numpy as np
from core.objectives.pinball import pinball_loss


class LightGBMModelWrapper:
    def __init__(self, model=None):
        self.model = model

    def __getstate__(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.model.booster_.save_model(tmp.name)
            with open(tmp.name, "rb") as f:
                state = f.read()
        os.remove(tmp.name)
        return state

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with open(tmp.name, "wb") as f:
                f.write(state)
            self.model = lgb.Booster(model_file=tmp.name)
        os.remove(tmp.name)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        if name == "model":
            object.__setattr__(self, name, value)
        else:
            setattr(self.model, name, value)


def train_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    weight_train,
    weight_val,
    quantiles=[0.1, 0.5, 0.9],
    **override_params
):
    default_params = {
        # "objective": "regression_l2",
        # "objective": 'regression_l1',
        "objective": "quantile",
        "n_estimators": 100,
        "learning_rate": 0.1,
        # "min_sum_hessian_in_leaf": 100, # aliases: min_child_weight
        "min_samples_leaf": 15,  # aliases: min_data_in_leaf, min_child_samples
        "colsample_bytree": 0.6,
        "subsample": 0.6,
        # "max_depth": 30,
        "random_state": 2022,
        "reg_alpha": 0,
        "max_bin": 256,
        "verbose": -1,
    }
    params = dict(default_params, **override_params)

    data_processor = DataProcessor(
        normalization_configuration={
            "__target__": "md",
            **{c: "md" for c in X_train.columns if c.startswith("volume_roll_mean5")},
            **{c: "categorical" for c in ["site_id", "issue_month"]},
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
    X_train_, y_train_, weight_train_ = data_processor.preprocess_transform(
        X_train, y_train, weight_train, mode="train"
    )
    X_val_, y_val_, weight_val_ = data_processor.preprocess_transform(
        X_val, y_val, weight_val, mode="val"
    )

    lgbm_model = Vector([LightGBMModelWrapper(lgb.LGBMRegressor(**params, alpha=q)) for q in quantiles])
    cat_cols = ["site_id", "issue_month"]

    # cat_cols = X_train.select_dtypes("object").columns
    # mappers = {col: X_train[col].astype("category").cat.categories for col in cat_cols}
    # for col, cats in mappers.items():
    # X_train[col] = pd.Categorical(X_train[col], categories=cats)
    # X_val[col] = pd.Categorical(X_val[col], categories=cats)

    lgbm_model.fit(
        X_train_,
        y_train_,
        sample_weight=weight_train_,
        eval_set=[(X_train_, y_train_), (X_val_, y_val_)],
        eval_sample_weight=[weight_train_, weight_val_],
        eval_names=["train", "val"],
        categorical_feature=cat_cols,
    )

    y_train_pred = np.array(lgbm_model.predict(X_train_))
    y_val_pred = np.array(lgbm_model.predict(X_val_))

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
    return lgbm_model, data_processor, error_val, X_train_.columns
