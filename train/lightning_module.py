from __future__ import annotations

from typing import Optional

import numpy as np
import pytorch_lightning as pl_lit
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from domain.config import GraphConfig, ModelConfig, TrainConfig, EvalConfig
from domain.types import DayBatch
from graph import GraphFeatureModule
from model import RankMLP, ic_loss, weighted_pairwise_rank_loss, graph_regularizer


class GraphRankLit(pl_lit.LightningModule):
    def __init__(
        self,
        graph_cfg: GraphConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        eval_cfg: EvalConfig,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg
        self.graph_module = GraphFeatureModule(graph_cfg)
        self.mlp = RankMLP(graph_cfg.dims.d_model, model_cfg)
        self.prev_latent_adj: Optional[torch.Tensor] = None
        self._val_ic: list[float] = []
        self._val_ret: list[float] = []
        self._test_outputs: list = []

    def _move_batch(self, batch: list[DayBatch]) -> list[DayBatch]:
        return [d.to(self.device) for d in batch]

    def _top_return(self, scores: torch.Tensor, labels: torch.Tensor, liquid: torch.Tensor) -> float:
        top_k = self.eval_cfg.top_k
        money = self.eval_cfg.money
        idx = torch.argsort(scores, descending=True)[:top_k]
        liq = torch.clamp(liquid[idx], min=0.0)
        ret = labels[idx]
        cum = torch.cumsum(liq, dim=0)
        prev = torch.cat([torch.zeros(1, device=liq.device), cum[:-1]])
        hold = torch.minimum(liq, torch.clamp(money - prev, min=0.0))
        return float((hold * ret).sum().item() / money)

    def forward(self, batch: DayBatch) -> torch.Tensor:
        graph_out = self.graph_module(batch)
        return self.mlp(graph_out.embedding)

    def training_step(self, batch, batch_idx):
        losses = []
        for day in self._move_batch(batch):
            graph_out = self.graph_module(day)
            score = self.mlp(graph_out.embedding)

            latent = next((r for r in graph_out.relations if "latent" in r.name), None)
            if latent is not None:
                self.prev_latent_adj = latent.adj.detach()

            rank_l = weighted_pairwise_rank_loss(score, day.norm_label)
            ic_l = ic_loss(score, day.norm_label)
            reg_l = graph_regularizer(graph_out.relations, self.prev_latent_adj)
            loss = self.train_cfg.w_rank * rank_l + self.train_cfg.w_ic * ic_l + self.train_cfg.w_reg * reg_l
            losses.append(loss)

        loss = torch.stack(losses).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        for day in self._move_batch(batch):
            score = self(day)
            y = day.origin_label
            ic = float(((score - score.mean()) * (y - y.mean())).mean().item() / (score.std().item() * y.std().item() + 1e-8))
            self._val_ic.append(ic)
            self._val_ret.append(self._top_return(score.detach(), y.detach(), day.liquid.detach()))

    def on_validation_epoch_end(self):
        val_ic = float(np.mean(self._val_ic)) if self._val_ic else 0.0
        val_top = float(np.mean(self._val_ret)) if self._val_ret else 0.0
        self.log("val_ic", val_ic, prog_bar=True)
        self.log("val_top_ret", val_top, prog_bar=True)
        self.log("val_composite", 50.0 * val_top + val_ic, prog_bar=True)
        self._val_ic.clear()
        self._val_ret.clear()

    def test_step(self, batch, batch_idx):
        for day in self._move_batch(batch):
            graph_out = self.graph_module(day)
            score = self.mlp(graph_out.embedding)
            embedding = graph_out.embedding.detach().cpu().numpy()
            self._test_outputs.append((day.date, list(day.codes), embedding, score.detach().cpu().numpy()))

    def on_test_epoch_end(self):
        self.test_records = self._test_outputs.copy()
        self._test_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.train_cfg.lr, weight_decay=self.train_cfg.weight_decay)

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor="val_composite", mode="max", patience=self.train_cfg.early_stop_patience),
            ModelCheckpoint(monitor="val_composite", mode="max", save_top_k=1, filename="{epoch}-{val_composite:.4f}"),
        ]
