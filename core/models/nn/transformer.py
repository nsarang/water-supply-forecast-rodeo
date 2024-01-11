import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, BertConfig, BertModel

from .utils import to_torch


class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_numerical_features,
        categorical_embedding_sizes,
        hidden_size=16,
        output_size=3,
        device="cpu",
        **transform_config,
    ):
        super().__init__()
        self.device = device

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_categories, embedding_dim=hidden_size)
                for num_categories in categorical_embedding_sizes
            ]
        )
        for layer in self.embeddings:
            torch.nn.init.zeros_(layer.weight)

        # Linear layer for numerical features
        self.numerical_projection = nn.Linear(1, hidden_size)

        # Get transformer configuration (use a small-sized transformer like Electra)
        default_config = dict(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            vocab_size=1,
        )
        updated_config = dict(default_config, **transform_config)
        bert_config = BertConfig(**updated_config)
        self.transformer = AutoModel.from_config(bert_config)

        # Output layer (adjust according to your task)
        self.output_layer = nn.Linear(bert_config.hidden_size, output_size)

        self.to(self.device)

    def forward(self, xx):
        categorical_features, numerical_features = xx[:, :1], xx[:, 1:]
        categorical_features = to_torch(categorical_features, device=self.device, dtype=torch.long)
        numerical_features = to_torch(numerical_features, device=self.device, dtype=torch.float32)

        # Embedding categorical features
        x_categorical = [
            embedding(categorical_features[:, i]) for i, embedding in enumerate(self.embeddings)
        ]
        x_categorical = torch.stack(x_categorical, dim=1)

        # Projecting numerical features
        x_numerical = self.numerical_projection(numerical_features.unsqueeze(-1))
        # Combining features
        x = torch.cat((x_categorical, x_numerical), dim=1)

        # Applying transformer
        transformer_out = self.transformer(inputs_embeds=x).last_hidden_state

        # Pooling (e.g., taking the first token's output)
        pooled_output = transformer_out[:, 0]

        # Output layer
        return self.output_layer(pooled_output).cpu()

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = x.drop(columns="site_id").fillna(0).values
        x = to_torch(x, dtype=torch.float32)
        output = self.forward(x)
        result = output.cpu().numpy().transpose((1, 0))
        return result
