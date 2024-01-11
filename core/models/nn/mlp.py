import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import to_torch


class MLPO(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 3),
            # nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = x.drop(columns="site_id").fillna(0).values
        x = to_torch(x, dtype=torch.float32)
        output = self.forward(x)
        result = output.cpu().numpy().transpose((1, 0))
        return result


class MLP(nn.Module):
    def __init__(self, input_size, num_categories, embedding_size=2):
        super(MLP, self).__init__()
        self.fp1 = nn.Linear(input_size + embedding_size, 96)
        # self.act1 = nn.ReLU()
        self.act1 = nn.PReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.fp2 = nn.Linear(96, 32)
        # self.act2 = nn.ReLU()
        self.act2 = nn.PReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.fp3 = nn.Linear(32, 3)
        self.emb = nn.Embedding(num_embeddings=num_categories, embedding_dim=embedding_size)

    def forward(self, x):
        x_cat = self.emb(x[:, :1].long())[:, 0]
        x_num = x[:, 1:]
        x_ = torch.cat((x_cat, x_num), dim=1)
        x1 = self.dropout1(self.act1(self.fp1(x_)))
        x2 = self.dropout2(self.act2(self.fp2(x1)))
        x3 = self.fp3(x2)
        return x3

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = x.drop(columns="site_id").fillna(0).values
        x = to_torch(x, dtype=torch.float32)
        output = self.forward(x)
        result = output.cpu().numpy().transpose((1, 0))
        return result


class LBRD_Block(nn.Module):
    def __init__(self, input_size, output_size, dropout_p=0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.block(x)


class TabularMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=None, dropout_p=0.5):
        """
        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features (e.g., number of classes in classification).
            hidden_layers (list of int): Size of each hidden layer.
            dropout_p (float): Dropout probability.
        """
        super().__init__()
        self.layers = nn.ModuleList()

        hidden_layers = hidden_layers or []
        layer_widths = [input_size] + hidden_layers
        for input_dim, output_dim in zip(layer_widths[:-1], layer_widths[1:]):
            self.layers.append(LBRD_Block(input_dim, output_dim, dropout_p=dropout_p))

        # Output layer
        self.layers.append(nn.Linear(layer_widths[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = x.drop(columns="site_id").fillna(0).values
        output = self.forward(x)
        result = output.cpu().numpy().transpose((1, 0))
        return result
