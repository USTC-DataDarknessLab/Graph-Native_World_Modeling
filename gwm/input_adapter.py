
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DomainInvariantInputAdapter(nn.Module):

    common_feature_dim = 9
    property_pool_bins = 128
    summary_dim = 5
    feature_dim = common_feature_dim + 2 * property_pool_bins + summary_dim

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        if self.output_dim < 1:
            raise ValueError("output_dim must be positive")
        self.normalization = nn.LayerNorm(self.feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.output_dim),
            nn.GELU(),
        )

    @classmethod
    def summarize(cls, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"expected [num_nodes, num_features], got {tuple(x.shape)}")
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        common = x[:, : cls.common_feature_dim]
        if common.shape[1] < cls.common_feature_dim:
            common = F.pad(common, (0, cls.common_feature_dim - common.shape[1]))
        properties = x[:, cls.common_feature_dim :]
        if properties.shape[1] == 0:
            mean_pool = x.new_zeros((x.shape[0], cls.property_pool_bins))
            max_pool = mean_pool.clone()
            summary = x.new_zeros((x.shape[0], cls.summary_dim))
        else:


            values = properties.unsqueeze(1)
            mean_pool = F.adaptive_avg_pool1d(values, cls.property_pool_bins).squeeze(1)
            max_pool = F.adaptive_max_pool1d(values, cls.property_pool_bins).squeeze(1)
            mean = properties.mean(dim=-1)
            std = properties.std(dim=-1, unbiased=False)
            rms = properties.square().mean(dim=-1).sqrt()
            nonzero = properties.ne(0).to(dtype=properties.dtype).mean(dim=-1)
            maximum = properties.amax(dim=-1)
            summary = torch.stack((mean, std, rms, nonzero, maximum), dim=-1)
        return torch.cat((common, mean_pool, max_pool, summary), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.summarize(x)
        return self.projection(self.normalization(features))


__all__ = ["DomainInvariantInputAdapter"]
