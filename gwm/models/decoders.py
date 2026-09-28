
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EdgeDecoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(
        self,
        source_hidden: torch.Tensor,
        destination_hidden: torch.Tensor,
        pair_context: torch.Tensor | None = None,
    ) -> torch.Tensor:



        del pair_context
        return self.network(torch.cat([source_hidden, destination_hidden], dim=-1)).squeeze(-1)


class PairContextEdgeDecoder(EdgeDecoder):

    def __init__(self, latent_dim: int, pair_context_dim: int):
        if pair_context_dim < 1:
            raise ValueError("pair_context_dim must be positive.")
        super().__init__(latent_dim)
        self.pair_context_dim = int(pair_context_dim)



        with torch.random.fork_rng(devices=[]):




            self.context_linear = nn.Linear(self.pair_context_dim, 1)
            self.context_network = nn.Sequential(
                nn.Linear(2 * latent_dim + self.pair_context_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 1),
            )



        nn.init.zeros_(self.context_linear.weight)
        nn.init.zeros_(self.context_linear.bias)
        nonlinear_output = self.context_network[-1]
        assert isinstance(nonlinear_output, nn.Linear)
        nn.init.zeros_(nonlinear_output.weight)
        nn.init.zeros_(nonlinear_output.bias)

    def forward(
        self,
        source_hidden: torch.Tensor,
        destination_hidden: torch.Tensor,
        pair_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pair_context is None:
            pair_context = source_hidden.new_zeros(
                (*source_hidden.shape[:-1], self.pair_context_dim)
            )
        if pair_context.shape != (*source_hidden.shape[:-1], self.pair_context_dim):
            raise ValueError(
                "pair_context must have shape [..., pair_context_dim] matching edge rows."
            )
        pair_context = pair_context.to(
            device=source_hidden.device, dtype=source_hidden.dtype
        )
        base = super().forward(source_hidden, destination_hidden)
        context_input = torch.cat(
            [source_hidden, destination_hidden, pair_context], dim=-1
        )
        return (
            base
            + self.context_linear(pair_context).squeeze(-1)
            + self.context_network(context_input).squeeze(-1)
        )


class EdgeAdditionDecoder(EdgeDecoder):
    pass


class EdgeDeletionDecoder(EdgeDecoder):
    pass


class TopologyOperationCountDecoder(nn.Module):

    def __init__(self, latent_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 2),
        )

    def forward(self, z_next: torch.Tensor) -> torch.Tensor:
        if z_next.ndim not in {2, 3}:
            raise ValueError(
                "TopologyOperationCountDecoder expects [num_nodes, latent_dim] "
                "or [batch, num_nodes, latent_dim]."
            )



        return F.softplus(self.network(z_next.mean(dim=-2)))


class NodeOperationCountDecoder(nn.Module):

    def __init__(self, latent_dim: int, num_operations: int = 3):
        super().__init__()
        if num_operations < 1:
            raise ValueError("num_operations must be positive.")
        self.num_operations = int(num_operations)
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, self.num_operations),
        )

    def forward(self, z_next: torch.Tensor) -> torch.Tensor:
        if z_next.ndim not in {2, 3}:
            raise ValueError(
                "NodeOperationCountDecoder expects [num_nodes, latent_dim] "
                "or [batch, num_nodes, latent_dim]."
            )
        return F.softplus(self.network(z_next.mean(dim=-2)))


class NodeChangeDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        source_dim: int = 0,
        causal_context_dim: int = 0,
        history_prior: bool = False,
    ):
        super().__init__()
        self.source_dim = int(source_dim)
        self.causal_context_dim = int(causal_context_dim)
        self.history_prior_enabled = bool(history_prior)
        if self.source_dim < 0:
            raise ValueError("source_dim must be non-negative.")
        if self.causal_context_dim < 0:
            raise ValueError("causal_context_dim must be non-negative.")
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )
        if self.source_dim:
            with torch.random.fork_rng(devices=[]):
                self.source_network = nn.Sequential(
                    nn.Linear(self.source_dim, latent_dim),
                    nn.GELU(),
                    nn.Linear(latent_dim, 1),
                )
            output = self.source_network[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        else:
            self.source_network = None
        if self.causal_context_dim:



            with torch.random.fork_rng(devices=[]):
                self.causal_gate = nn.Sequential(
                    nn.Linear(self.causal_context_dim, latent_dim),
                    nn.GELU(),
                    nn.Linear(latent_dim, 1),
                )
                self.causal_expert = nn.Sequential(
                    nn.Linear(latent_dim + self.causal_context_dim, latent_dim),
                    nn.GELU(),
                    nn.Linear(latent_dim, 1),
                )
            expert_output = self.causal_expert[-1]
            assert isinstance(expert_output, nn.Linear)
            nn.init.zeros_(expert_output.weight)
            nn.init.zeros_(expert_output.bias)
        else:
            self.causal_gate = None
            self.causal_expert = None
        if self.history_prior_enabled:
            if self.causal_context_dim < 1:
                raise ValueError("history_prior requires causal_context_dim > 0")
            with torch.random.fork_rng(devices=[]):
                self.history_prior = nn.Linear(self.causal_context_dim, 1)
            nn.init.zeros_(self.history_prior.weight)
            nn.init.zeros_(self.history_prior.bias)
        else:
            self.history_prior = None

    def forward(
        self,
        h_next: torch.Tensor,
        source_context: torch.Tensor | None = None,
        causal_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.network(h_next).squeeze(-1)
        if self.source_network is not None:
            if source_context is None:
                source_context = h_next.new_zeros(
                    (*h_next.shape[:-1], self.source_dim)
                )
            if source_context.shape != (*h_next.shape[:-1], self.source_dim):
                raise ValueError(
                    "source_context must match h_next except for its last dimension."
                )
            output = output + self.source_network(
                source_context.to(device=h_next.device, dtype=h_next.dtype)
            ).squeeze(-1)
        if self.causal_expert is not None:
            if causal_context is None:
                causal_context = h_next.new_zeros(
                    (*h_next.shape[:-1], self.causal_context_dim)
                )
            if causal_context.shape != (
                *h_next.shape[:-1],
                self.causal_context_dim,
            ):
                raise ValueError(
                    "causal_context must match h_next except for its last dimension."
                )
            causal_context = causal_context.to(
                device=h_next.device, dtype=h_next.dtype
            )
            assert self.causal_gate is not None
            gate = torch.sigmoid(self.causal_gate(causal_context)).squeeze(-1)
            expert_input = torch.cat([h_next, causal_context], dim=-1)
            output = output + gate * self.causal_expert(expert_input).squeeze(-1)
            if self.history_prior is not None:
                output = output + self.history_prior(causal_context).squeeze(-1)
        return output


class NodeActivityDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        *,
        separate_heads: bool = False,
        activation: str = "relu",
        addition_source_dim: int = 0,
        history_context_dim: int = 0,
        history_prior: bool = False,
        history_prior_remove_only: bool = False,
    ):
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("NodeActivityDecoder activation must be relu or gelu.")
        self.separate_heads = bool(separate_heads)
        self.activation_name = str(activation)
        self.addition_source_dim = int(addition_source_dim)
        self.history_context_dim = int(history_context_dim)
        self.history_prior_enabled = bool(history_prior)
        self.history_prior_remove_only = bool(history_prior_remove_only)
        if self.addition_source_dim < 0:
            raise ValueError("addition_source_dim must be non-negative.")
        if self.history_context_dim < 0:
            raise ValueError("history_context_dim must be non-negative.")

        def _activation() -> nn.Module:
            return nn.ReLU() if activation == "relu" else nn.GELU()

        if self.separate_heads:
            self.activation_network = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                _activation(),
                nn.Linear(latent_dim, 1),
            )
            self.deactivation_network = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                _activation(),
                nn.Linear(latent_dim, 1),
            )
            self.network = None
        else:
            self.network = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                _activation(),
                nn.Linear(latent_dim, 2),
            )
            self.activation_network = None
            self.deactivation_network = None




        if self.addition_source_dim:
            with torch.random.fork_rng(devices=[]):
                self.addition_source_network = nn.Sequential(
                    nn.Linear(self.addition_source_dim, latent_dim),
                    _activation(),
                    nn.Linear(latent_dim, 1),
                )
            output = self.addition_source_network[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        else:
            self.addition_source_network = None




        if self.history_context_dim:
            with torch.random.fork_rng(devices=[]):
                self.history_gate = nn.Sequential(
                    nn.Linear(self.history_context_dim, latent_dim),
                    _activation(),
                    nn.Linear(latent_dim, 2),
                )
                self.history_expert = nn.Sequential(
                    nn.Linear(latent_dim + self.history_context_dim, latent_dim),
                    _activation(),
                    nn.Linear(latent_dim, 2),
                )
            history_output = self.history_expert[-1]
            assert isinstance(history_output, nn.Linear)
            nn.init.zeros_(history_output.weight)
            nn.init.zeros_(history_output.bias)
        else:
            self.history_gate = None
            self.history_expert = None
        if self.history_prior_enabled:
            if self.history_context_dim < 1:
                raise ValueError("history_prior requires history_context_dim > 0")
            with torch.random.fork_rng(devices=[]):
                self.history_prior = nn.Linear(self.history_context_dim, 2)
            nn.init.zeros_(self.history_prior.weight)
            nn.init.zeros_(self.history_prior.bias)
        else:
            self.history_prior = None

    def forward(
        self,
        z_next: torch.Tensor,
        addition_source_context: torch.Tensor | None = None,
        history_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.separate_heads:
            assert self.activation_network is not None
            assert self.deactivation_network is not None
            output = torch.cat(
                [self.activation_network(z_next), self.deactivation_network(z_next)],
                dim=-1,
            )
        else:
            assert self.network is not None
            output = self.network(z_next)
        if self.addition_source_network is not None:
            if addition_source_context is None:
                addition_source_context = z_next.new_zeros(
                    (*z_next.shape[:-1], self.addition_source_dim)
                )
            if addition_source_context.shape != (*z_next.shape[:-1], self.addition_source_dim):
                raise ValueError(
                    "addition_source_context must match z_next except for its last dimension."
                )
            output = output.clone()
            output[..., 0] = output[..., 0] + self.addition_source_network(
                addition_source_context.to(device=z_next.device, dtype=z_next.dtype)
            ).squeeze(-1)
        if self.history_expert is not None:
            if history_context is None:
                history_context = z_next.new_zeros(
                    (*z_next.shape[:-1], self.history_context_dim)
                )
            if history_context.shape != (
                *z_next.shape[:-1],
                self.history_context_dim,
            ):
                raise ValueError(
                    "history_context must match z_next except for its last dimension."
                )
            history_context = history_context.to(
                device=z_next.device, dtype=z_next.dtype
            )
            assert self.history_gate is not None
            gate = torch.sigmoid(self.history_gate(history_context))
            output = output + gate * self.history_expert(
                torch.cat([z_next, history_context], dim=-1)
            )
            if self.history_prior is not None:
                prior = self.history_prior(history_context)
                if self.history_prior_remove_only:
                    output = output.clone()
                    output[..., 1] = output[..., 1] + prior[..., 1]
                else:
                    output = output + prior
        return output


class NodeStateDecoder(nn.Module):

    def __init__(
        self, latent_dim: int, output_dim: int, *, zero_init_output: bool = False
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )
        if zero_init_output:
            output = self.network[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, h_next: torch.Tensor) -> torch.Tensor:
        return self.network(h_next)


class NodePropertyDecoder(nn.Module):

    def __init__(
        self, latent_dim: int, output_dim: int, *, zero_init_output: bool = False
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )
        if zero_init_output:



            output = self.network[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, z_next: torch.Tensor) -> torch.Tensor:
        return self.network(z_next)


class ChangeAwareGaussianPropertyDecoder(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        *,
        min_logvar: float = -12.0,
        max_logvar: float = 6.0,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or output_dim < 1:
            raise ValueError("latent_dim and output_dim must be positive")
        if min_logvar >= max_logvar:
            raise ValueError("min_logvar must be smaller than max_logvar")
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)



        self.change_decoder = NodeChangeDecoder(latent_dim)
        self.magnitude_context = nn.Sequential(
            nn.Linear(latent_dim + 1, latent_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(latent_dim, output_dim)
        self.logvar_head = nn.Linear(latent_dim, output_dim)

    def forward(self, z_next: torch.Tensor) -> dict[str, torch.Tensor]:
        if z_next.ndim != 2 or z_next.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected future latents [rows,{self.latent_dim}], got {tuple(z_next.shape)}"
            )
        change_logits = self.change_decoder(z_next)
        change_probability = torch.sigmoid(change_logits)
        magnitude_hidden = self.magnitude_context(



            torch.cat([z_next, change_probability.detach().unsqueeze(-1)], dim=-1)
        )
        return {
            "change_logits": change_logits,
            "change_probability": change_probability,
            "mu": self.mu_head(magnitude_hidden),
            "logvar": self.logvar_head(magnitude_hidden).clamp(
                self.min_logvar, self.max_logvar
            ),
        }
