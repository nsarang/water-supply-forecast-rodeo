import pandas as pd
from collections import defaultdict
from .outlier import iqr_outliers


class DataProcessor:
    def __init__(self, normalization_configuration=None):
        self.norm_cfg = normalization_configuration or {}
        self.assets = defaultdict(dict)

    def preprocess_fit(self, X, y):
        for col, type in self.norm_cfg.items():
            if col == "__target__":
                values = y
            else:
                values = X[col]
            self._normalize_fit(X, values, type)

    def preprocess_transform(self, X, y=None, weight=None, mode="train"):
        if mode == "train":
            mask = (
                y.groupby(X["site_id"])
                .apply(iqr_outliers, scale=3)
                .droplevel(level=0)
                .reindex(y.index)
            )
            X, y, weight = X.loc[mask], y[mask], weight[mask]

        for col, type in self.norm_cfg.items():
            if col == "__target__":
                y = self._normalize_transform(X, y, type)
            else:
                X.loc[:, col] = self._normalize_transform(X, X[col], type)

        return X, y, weight

    def postprocess_transform(self, X, y, mode="train"):
        type = self.norm_cfg["__target__"]
        return self._normalize_inverse_transform(X, y, type)

    def _normalize_fit(self, X, feature, type):
        name = feature.name
        if type == "std":
            self.assets["norm"][name] = {
                "mean": feature.groupby(X["site_id"]).mean().reset_index(name="mean"),
                "std": feature.groupby(X["site_id"]).std().reset_index(name="std"),
            }
        elif type == "md":
            self.assets["norm"][name] = {
                "std": feature.groupby(X["site_id"]).std().reset_index(name="std")
            }
        elif type == "basins_area":
            pass

    def _normalize_transform(self, X, feature, type):
        config = self.assets["norm"][feature.name]
        if type == "std":
            mean, std = config["mean"], config["std"]
            mean, std = self._broadcast(X, mean), self._broadcast(X, std)
            feature = (feature - mean) / std
        elif type == "md":
            mean = self._broadcast(X, config["mean"])
            feature = feature / mean
        elif type == "basins_area":
            feature = feature / X["basins_area"]
        return feature

    def _normalize_inverse_transform(self, X, feature, type):
        config = self.assets["norm"][feature.name]
        if type == "std":
            mean, std = config["mean"], config["std"]
            mean, std = self._broadcast(X, mean), self._broadcast(X, std)
            feature = (feature * std) + mean
        elif type == "md":
            mean = self._broadcast(X, config["mean"])
            feature = feature * mean
        elif type == "basins_area":
            feature = feature * X["basins_area"]
        return feature

    def _broadcast(self, X, series):
        site_ids = X["site_id"]
        scale_ordered = (
            pd.merge(site_ids.reset_index(), series, on="site_id")
            .set_index("index")
            .reindex(site_ids.index)[series.name]
            .to_numpy()
        )
        return scale_ordered
