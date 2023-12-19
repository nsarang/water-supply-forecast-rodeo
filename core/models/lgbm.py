import lightgbm as lgb
import pandas as pd


def train_lgbm(X_train, y_train, X_val, y_val, weight_train, weight_val, **override_params):
    default_params = {
        "objective": "regression_l2",
        # "objective": 'regression_l1',
        "n_estimators": 100,
        # "learning_rate": 0.1,
        # "min_sum_hessian_in_leaf": 100, # aliases: min_child_weight
        "min_samples_leaf": 5,  # aliases: min_data_in_leaf, min_child_samples
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        # "max_depth": 50,
        "random_state": 2022,
        "reg_alpha": 0,
        "max_bin": 256,
        "verbose": 1,
    }
    params = dict(default_params, **override_params)

    lgbm_model = lgb.LGBMRegressor(**params)
    cat_cols = X_train.select_dtypes("object").columns
    mappers = {col: X_train[col].astype("category").cat.categories for col in cat_cols}
    for col, cats in mappers.items():
        X_train[col] = pd.Categorical(X_train[col], categories=cats)
        X_val[col] = pd.Categorical(X_val[col], categories=cats)

    lgbm_model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_sample_weight=[weight_train, weight_val],
        eval_names=["train", "val"],
        categorical_feature=cat_cols,
    )
    return lgbm_model
