from __future__ import annotations

import torch
from torch import nn


def _kaiming(m: nn.Module) -> None:
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(mod.weight, mode="fan_in", nonlinearity="relu")
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0.0)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.conv(x)))


class TCNTemporalEncoder(nn.Module):
    """[N, L] -> [N, d_tmp] via dilated 1D TCN."""

    def __init__(self, hist_len: int, d_tmp: int, dropout: float = 0.1, channels: int = 32):
        super().__init__()
        self.proj_in = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList([_TCNBlock(channels, d, dropout) for d in (1, 2, 4)])
        self.proj_out = nn.Linear(channels, d_tmp)
        _kaiming(self)

    def forward(self, ret_hist: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(ret_hist.unsqueeze(1))
        for blk in self.blocks:
            h = blk(h)
        return self.proj_out(h[:, :, -1])


class GRUTemporalEncoder(nn.Module):
    """[N, L] -> [N, d_tmp] via 1-layer GRU."""

    def __init__(self, hist_len: int, d_tmp: int, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=d_tmp, num_layers=1, batch_first=True)

    def forward(self, ret_hist: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(ret_hist.unsqueeze(-1))
        return h.squeeze(0)
