import os
import sys

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import core.recipes as recipes
from loguru import logger
from tqdm.auto import tqdm


from core.datasets import *
from core.utils import Vector
from core.config import SHOW_PROGRESS_BAR
from core.data_loader import prepare_dataset
from core.train import train_util
import pickle


def load_model(month, day, src_dir, data_dir, preprocessed_dir, assets):
    label = (month, day)
    assets[label] = {}
    model_file = src_dir / f"{month}_{day}_.pkl"

    
    if not os.path.isfile(model_file):
        df_train_monthly = pd.read_csv(data_dir / "train_monthly_naturalized_flow.csv")
        df_train_targets = pd.read_csv(data_dir / "train.csv").rename(
            columns={"year": "forecast_year"}
        )

        train_features = prepare_dataset(month, day, df_train_monthly, df_train_targets)
        configuration = recipes.mvp_config
        logs = train_util(train_features, configuration)

        assets[label] = {
            "train_features": train_features,
            "model": logs["model"],
            "logs": logs,
        }
        with open(model_file, 'wb') as fh:
            pickle.dump(logs["model"], fh)

    with open(model_file, 'rb') as fh:
        model = pickle.load(fh)
        assets[label]["model"] = model

    df_test_monthly = pd.read_csv(data_dir / "test_monthly_naturalized_flow.csv")
    test_features = prepare_dataset(month, day, df_test_monthly)
    predictions = model.predict(test_features)
    test_features["predictions"] = np.transpose(predictions, (1, 0))[:, :3].astype(float).tolist()
    assets[label]["test_features"] = test_features


def run_inference(models, features):
    output_raw = np.array(Vector(models).predict(features))
    output = output_raw.mean(axis=0)
    return output


def preprocess(src_dir: Path, data_dir: Path, preprocessed_dir: Path) -> dict[Hashable, Any]:
    return {}


def predict(
    site_id: str,
    issue_date: str,
    assets: dict[Any, Any],
    src_dir: Path,
    data_dir: Path,
    preprocessed_dir: Path,
) -> tuple[float, float, float]:
    tqdm.pandas(disable=not SHOW_PROGRESS_BAR)

    issue_date = pd.to_datetime(issue_date)
    month, day = issue_date.month, issue_date.day
    label = (month, day)

    if label not in assets:
        load_model(month, day, src_dir, data_dir, preprocessed_dir, assets)

    output = assets[label]["test_features"].query(
        f"site_id == '{site_id}' & issue_date == '{issue_date}'"
    )["predictions"].iloc[0]

    return output
