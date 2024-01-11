from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from objectives.quantile import QuantileLoss
from torch.utils.data import DataLoader, TensorDataset

from core.data.preprocess import DataProcessor
from core.models.ema import EMA
from core.models.nn.odst import ODST, DenseODST
from core.objectives.pinball import pinball_loss

from .autoclip import AutoClip
from .lookahead import Lookahead
from .mlp import MLP, TabularMLP
from .ralamb import RangerLars
from .transformer import TabularTransformer
from .utils import to_torch


class ODST_MLP(nn.Module):
    def __init__(self, input_size, num_categories, embedding_size=2):
        super().__init__()
        self.odst = ODST(
            in_features=input_size + embedding_size,
            num_trees=12,
            depth=6,
            tree_dim=3,
            output_type="mean",
        )
        # self.odst = DenseODST(
        #     input_dim=input_size + embedding_size,
        #     num_layers=4,
        #     layer_dim=6,
        #     depth=2,
        #     tree_dim=3,
        #     output_type="mean",
        # )
        self.emb = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_size)

    def forward(self, x):
        x_cat = self.emb(x[:, :1].long())[:, 0]
        x_num = x[:, 1:]
        x_ = torch.cat((x_cat, x_num), dim=1)
        return self.odst(x_)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = x.drop(columns="site_id").fillna(0).values
        x = to_torch(x, dtype=torch.float32)
        output = self.forward(x)
        result = output.cpu().numpy().transpose((1, 0))
        return result


def train_nn(
    X_train,
    y_train,
    X_val,
    y_val,
    weight_train,
    weight_val,
    custom_feature_weights=None,
    verbose=True,
    epochs=100,
    early_stopping_patience=10,
    learning_rate=0.01,
    weight_decay=1e-2,
    batch_size=128,
    scheduler_step_size=10,
    scheduler_gamma=0.1,
    model_config=None,
    **override_params,
):
    # custom_feature_weights = custom_feature_weights or {}
    # feature_weights = np.full(X_train.shape[1], 1.0)
    # for k, v in custom_feature_weights.items():
    #     feature_weights[X_train.columns.get_loc(k)] = v

    data_processor = DataProcessor(
        normalization_configuration={
            "__target__": "std",
            **{c: "categorical-codes" for c in ["site_id-2"]},
            # **{c: "md" for c in X_train.columns if c.startswith("volume")},
            **{
                c: "gstd"
                for c in X_train.columns
                if c
                in [
                    "latitude",
                    "longitude",
                    "elevation",
                    "basins_area",
                    "elevation_station",
                    "latitude_station",
                    "longitude_station",
                    "distance_station",
                ]
            },
            **{
                c: "std"
                for c in X_train.columns
                if (
                    c
                    not in [
                        "site_id",
                        "site_id-2",
                        "issue_month",
                        "latitude",
                        "longitude",
                        "elevation",
                        "basins_area",
                    ]
                )
                # and (not c.startswith("volume"))
            },
        }
    )

    def _preprocess(X):
        X["site_id-2"] = X["site_id"]
        cats = ["site_id-2"]
        columns = cats + [c for c in X.columns if c not in cats]
        X = X[columns]
        return X

    X_train = _preprocess(X_train)
    X_val = _preprocess(X_val)

    data_processor.preprocess_fit(X_train, y_train)
    X_train_, y_train_, weight_train_ = data_processor.preprocess_transform(
        X_train, y_train, weight_train, mode="train"
    )
    X_val_, y_val_, weight_val_ = data_processor.preprocess_transform(
        X_val, y_val, weight_val, mode="val"
    )

    X_train_2 = X_train_.drop(columns="site_id").fillna(0).values
    X_val_2 = X_val_.drop(columns="site_id").fillna(0).values

    # from sklearn.impute import KNNImputer
    # imputer = KNNImputer(n_neighbors=5)
    # X_train_2 = imputer.fit_transform(X_train_2)
    # X_val_2 = imputer.transform(X_val_2)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_2, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_.values, dtype=torch.float32)
    weight_train_tensor = torch.tensor(weight_train_, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_2, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_.values, dtype=torch.float32)
    weight_val_tensor = torch.tensor(weight_val_, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weight_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, weight_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model_config = model_config or {}
    model = MLP(
        input_size=X_train_2.shape[1] - 1,
        num_categories=X_train_["site_id-2"].nunique() + 1,
        embedding_size=2,
    )
    # model = ODST_MLP(
    #     input_size=X_train_2.shape[1] - 1,
    #     num_categories=X_train_["site_id-2"].nunique() + 1,
    #     embedding_size=2,
    # )
    # model = TabularMLP(
    #     input_size=X_train_2.shape[1], output_size=3, hidden_layers=[64, 32, 16], dropout_p=0.1
    # )
    # model = TabularTransformer(
    #     num_numerical_features=None,
    #     categorical_embedding_sizes=[X_train_["site_id-2"].nunique() + 1],
    #     **model_config,
    #     device="cpu",
    # )

    ema = EMA(
        model,
        beta=0.5,  # exponential moving average factor
        update_after_step=20,  # only after this number of .update() calls will it start updating
        update_every=5,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    # Loss function and optimizer
    # criterion = nn.L1Loss(reduction="none")
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = Lookahead(optimizer)
    # optimizer = RangerLars(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    gradclip = AutoClip(clip_percentile=99)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
    )  # Learning rate scheduler
    # scheduler = transformers.get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=3, num_training_steps=500
    # )
    # scheduler = transformers.get_cosine_schedule_with_warmup(
    #     optimizer, num_warmup_steps=3, num_training_steps=25, num_cycles=0.5
    # )
    # scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer, num_warmup_steps=3, num_training_steps=500, num_cycles=3
    # )

    # Early stopping and checkpointing
    best_val_loss = float("inf")
    early_stopping_counter = 0
    checkpoint_path = "model_checkpoint.pth"

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for inputs, targets, batch_weights in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = criterion(outputs.squeeze(), targets)
            weighted_loss = (
                losses * batch_weights[:, None]
            ).mean()  # Apply weights and then take the mean
            weighted_loss.backward()
            gradclip(model)
            optimizer.step()
            train_losses.append(weighted_loss.item())
            # ema.update()

        # Validation
        eval_model = model
        # eval_model = ema.ema_model
        eval_model.eval()
        with torch.no_grad():
            val_losses = []
            for inputs, targets, batch_weights in val_loader:
                outputs = eval_model(inputs)
                losses = criterion(outputs.squeeze(), targets)
                weighted_loss = (losses * batch_weights[:, None]).mean()
                val_losses.append(weighted_loss.item())

        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f} - Val loss: {avg_val_loss:.4f}"
            )

        # Learning rate scheduling
        scheduler.step()

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            best_model = deepcopy(eval_model)
            # torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    # Load the best model
    # best_model = ema.ema_model
    # model.load_state_dict(torch.load(checkpoint_path))

    y_train_pred = best_model.predict(X_train_tensor)
    y_val_pred = best_model.predict(X_val_tensor)

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

    return best_model, data_processor, error_val, X_train_.columns
