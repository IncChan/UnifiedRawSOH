"""Models used by the isolated Paper-Backup experiments.

The sequence baselines in this module deliberately have different forward
paths.  In particular, :class:`VanillaMambaSOHModel` is one continuous stream;
it does not reuse the proposed model's CC/CV branches or its bridge.  The
proposed model is wrapped separately so its historical implementation can be
reused without inheriting its cycle/lifetime auxiliary heads.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..c5b_model import _make_mixer, pool_hidden_states
from ..c5b_model import PaperRawSOHModel


def _coerce_mask(sequence: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"sequence must have shape [batch, length, channels], got {tuple(sequence.shape)}")
    if mask is None:
        return torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
    if mask.shape != sequence.shape[:2]:
        raise ValueError(
            f"mask must have shape {tuple(sequence.shape[:2])}, got {tuple(mask.shape)}"
        )
    return mask.to(device=sequence.device, dtype=torch.bool)


class _SequenceHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim < 2:
            raise ValueError("head_hidden_dim must be at least 2")
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _MaskedSequenceRegressor(nn.Module):
    input_contract = "joint_raw_sequence"
    model_family = "sequence"

    def __init__(self, input_dim: int, d_model: int, head_hidden_dim: int, dropout: float):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.pooling = "last_mean"
        self.head = _SequenceHead(2 * self.d_model, int(head_hidden_dim), float(dropout))

    def encode(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encode(sequence, mask))


class VanillaMambaSOHModel(_MaskedSequenceRegressor):
    """A genuine single-stream joint-sequence Mamba regressor.

    It is used for E2 full/terminal comparisons.  The model receives one
    ordered sequence containing CC followed by CV; no phase-specific branch,
    explicit phase ID, strategy ID, or CC-to-CV bridge is constructed here.
    """

    model_id = "VanillaMamba"

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 32,
        num_layers: int = 3,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank="auto",
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
        use_boundary_token: bool = False,
        backend: str = "mamba_ssm.Mamba",
    ):
        super().__init__(input_dim, d_model, head_hidden_dim, dropout)
        self.backend = str(backend)
        self.use_boundary_token = bool(use_boundary_token)
        self.boundary_token = (
            nn.Parameter(torch.zeros(self.d_model))
            if self.use_boundary_token
            else None
        )
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(self.d_model),
                        "mixer": _make_mixer(
                            self.d_model,
                            int(d_state),
                            int(d_conv),
                            int(expand),
                            dt_rank,
                            layer_idx,
                            self.backend,
                        ),
                        "dropout": nn.Dropout(float(dropout)),
                    }
                )
                for layer_idx in range(int(num_layers))
            ]
        )
        self.final_norm = nn.LayerNorm(self.d_model)

    def _insert_boundary(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        boundary_index: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_boundary_token:
            if boundary_index is not None:
                raise ValueError("boundary_index was provided but boundary tokens are disabled")
            return hidden, mask
        if boundary_index is None:
            raise ValueError("Boundary-aware Vanilla Mamba requires boundary_index")
        boundary_index = boundary_index.to(device=hidden.device, dtype=torch.long).reshape(-1)
        if boundary_index.numel() != hidden.size(0):
            raise ValueError("One boundary_index is required for each sequence")
        if torch.any(boundary_index <= 0) or torch.any(boundary_index >= hidden.size(1)):
            raise ValueError("boundary_index must lie strictly inside the physical sequence")
        rows, masks = [], []
        token = self.boundary_token.view(1, -1)
        for batch_index, split_index in enumerate(boundary_index.tolist()):
            rows.append(
                torch.cat(
                    (
                        hidden[batch_index, :split_index],
                        token,
                        hidden[batch_index, split_index:],
                    ),
                    dim=0,
                )
            )
            masks.append(
                torch.cat(
                    (
                        mask[batch_index, :split_index],
                        torch.ones(1, dtype=torch.bool, device=mask.device),
                        mask[batch_index, split_index:],
                    ),
                    dim=0,
                )
            )
        return torch.stack(rows, dim=0), torch.stack(masks, dim=0)

    def encode(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        boundary_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = _coerce_mask(sequence, mask)
        hidden = self.input_norm(self.input_proj(sequence))
        hidden, mask = self._insert_boundary(hidden, mask, boundary_index)
        for layer in self.layers:
            hidden = hidden + layer["dropout"](layer["mixer"](layer["norm"](hidden)))
        hidden = self.final_norm(hidden)
        return pool_hidden_states(hidden, mask, pooling="last_mean")

    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        boundary_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.head(self.encode(sequence, mask, boundary_index))


class SingleStreamMambaSOHModel(VanillaMambaSOHModel):
    """The same honest one-stream architecture for CC-only or CV-only views."""

    model_id = "SingleStreamMamba"
    input_contract = "single_phase_raw_sequence"


class TransformerSOHModel(_MaskedSequenceRegressor):
    """Masked encoder-only Transformer for an ordered raw charging sequence."""

    model_id = "Transformer"

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
        max_length: int = 4096,
    ):
        super().__init__(input_dim, d_model, head_hidden_dim, dropout)
        if int(d_model) % int(num_heads) != 0:
            raise ValueError("Transformer d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.max_length = int(max_length)
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(self.d_model)

    def _position(self, length: int, device, dtype) -> torch.Tensor:
        if length > self.max_length:
            raise ValueError(
                f"Transformer sequence length {length} exceeds configured max_length {self.max_length}"
            )
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        encoding = torch.zeros(length, self.d_model, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def encode(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = _coerce_mask(sequence, mask)
        hidden = self.input_norm(self.input_proj(sequence))
        hidden = hidden + self._position(hidden.size(1), hidden.device, hidden.dtype)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.final_norm(hidden)
        return pool_hidden_states(hidden, mask, pooling="last_mean")


class LSTMSOHModel(_MaskedSequenceRegressor):
    model_id = "LSTM"

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ):
        super().__init__(input_dim, d_model, head_hidden_dim, dropout)
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        self.encoder = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.final_norm = nn.LayerNorm(self.d_model)

    def encode(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = _coerce_mask(sequence, mask)
        hidden, _ = self.encoder(self.input_proj(sequence))
        return pool_hidden_states(self.final_norm(hidden), mask, pooling="last_mean")


class RawCNNSOHModel(_MaskedSequenceRegressor):
    model_id = "RawCNN"

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ):
        super().__init__(input_dim, d_model, head_hidden_dim, dropout)
        channels = max(8, int(d_model) // 2)
        self.input_proj = nn.Conv1d(self.input_dim, channels, kernel_size=5, padding=2)
        self.convs = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(channels, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.final_norm = nn.LayerNorm(self.d_model)

    def encode(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = _coerce_mask(sequence, mask)
        hidden = self.convs(self.input_proj(sequence.transpose(1, 2))).transpose(1, 2)
        hidden = self.final_norm(hidden)
        return pool_hidden_states(hidden, mask, pooling="last_mean")


class PhaseMambaSOHOnly(PaperRawSOHModel):
    """Paper-Backup Ours with the historical cycle path disabled explicitly."""

    model_id = "Ours"
    input_contract = "phase_separated_terminal_raw"

    def __init__(self, **kwargs):
        kwargs = dict(kwargs)
        self.active_phase = str(kwargs.pop("active_phase", "both"))
        if self.active_phase not in {"both", "cc", "cv"}:
            raise ValueError("Ours active_phase must be 'both', 'cc' or 'cv'")
        if kwargs.get("use_cycle_prediction", False):
            raise ValueError("Paper-Backup Ours is SOH-only: use_cycle_prediction must be false")
        if kwargs.get("use_predicted_cycle_for_soh", False):
            raise ValueError("Paper-Backup Ours is SOH-only: predicted cycle injection must be false")
        kwargs["use_cycle_prediction"] = False
        kwargs["use_predicted_cycle_for_soh"] = False
        kwargs.setdefault("backend", "mamba_ssm.Mamba")
        super().__init__(**kwargs)
        if self.cycle_head is not None or self.cycle_adapter is not None:
            raise RuntimeError("Paper-Backup Ours unexpectedly constructed a cycle auxiliary path")

    def encode_signal_feature(
        self,
        cc_signal,
        cv_signal,
        cc_mask=None,
        cv_mask=None,
        cc_time=None,
        cv_time=None,
        cc_temperature=None,
        cv_temperature=None,
    ):
        # For CC-only/CV-only input ablations the dataset replaces every
        # channel of the removed phase with a fixed zero tensor.  Running the
        # unchanged two-branch network keeps all trainable parameters and
        # isolates missing cycle-specific phase information from capacity.
        if self.active_phase == "cc":
            cv_signal = torch.zeros_like(cv_signal)
            cv_time = None if cv_time is None else torch.zeros_like(cv_time)
            cv_temperature = (
                None
                if cv_temperature is None
                else torch.zeros_like(cv_temperature)
            )
        elif self.active_phase == "cv":
            cc_signal = torch.zeros_like(cc_signal)
            cc_time = None if cc_time is None else torch.zeros_like(cc_time)
            cc_temperature = (
                None
                if cc_temperature is None
                else torch.zeros_like(cc_temperature)
            )
        return super().encode_signal_feature(
            cc_signal=cc_signal,
            cv_signal=cv_signal,
            cc_mask=cc_mask,
            cv_mask=cv_mask,
            cc_time=cc_time,
            cv_time=cv_time,
            cc_temperature=cc_temperature,
            cv_temperature=cv_temperature,
        )

    def forward(
        self,
        cc_signal,
        cv_signal,
        cc_mask=None,
        cv_mask=None,
        cc_time=None,
        cv_time=None,
        cc_temperature=None,
        cv_temperature=None,
        t0_temperature_norm=None,
    ):
        """SOH-only public forward signature with no lifetime argument."""

        return self.forward_with_aux(
            cc_signal=cc_signal,
            cv_signal=cv_signal,
            cc_mask=cc_mask,
            cv_mask=cv_mask,
            cc_time=cc_time,
            cv_time=cv_time,
            cc_temperature=cc_temperature,
            cv_temperature=cv_temperature,
            t0_temperature_norm=t0_temperature_norm,
        )["soh_pred"]


__all__ = [
    "LSTMSOHModel",
    "PhaseMambaSOHOnly",
    "RawCNNSOHModel",
    "SingleStreamMambaSOHModel",
    "TransformerSOHModel",
    "VanillaMambaSOHModel",
]
