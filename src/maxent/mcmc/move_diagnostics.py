from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class _OneMoveDiag:
    # proposal / accepted absolute ΔH
    sum_abs_dH_prop: float = 0.0
    sum_abs_dH_acc: float = 0.0
    # proposal / accepted Δn norms
    sum_dn_l1_prop: float = 0.0
    sum_dn_l1_acc: float = 0.0
    sum_dn_l2_prop: float = 0.0
    sum_dn_l2_acc: float = 0.0
    # for quantiles of |ΔH| (accepted)
    buf_abs_dH_acc: List[torch.Tensor] = field(default_factory=list)


@dataclass
class MoveDiagAccumulator:
    """Accumulate per-move diagnostics without polluting the sampling loop.

    This is intentionally minimal and side-effect free.
    """

    max_store: int = 5000
    moves: Dict[str, _OneMoveDiag] = field(
        default_factory=lambda: {
            "pca": _OneMoveDiag(),
            "ws": _OneMoveDiag(),
            "cg": _OneMoveDiag(),
        }
    )

    def _append_buffer(self, buf: List[torch.Tensor], vals: torch.Tensor) -> None:
        # vals: (k,) tensor on CPU/GPU. We store CPU float64 to compute quantiles cheaply.
        if vals.numel() == 0:
            return
        vals_cpu = vals.detach().to("cpu", dtype=torch.float64).flatten()
        if len(buf) < self.max_store:
            buf.append(vals_cpu)
        else:
            # reservoir-ish: keep the last max_store chunks (simple + deterministic)
            # This keeps memory bounded and behavior stable.
            buf.pop(0)
            buf.append(vals_cpu)

    @staticmethod
    def _l1_l2_dn(n_prop: torch.Tensor, n_curr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # n_* : (B, Nx) or (Nx,)
        dn = (n_prop - n_curr).reshape(n_prop.shape[0], -1) if n_prop.ndim > 1 else (n_prop - n_curr).reshape(1, -1)
        l1 = dn.abs().sum(dim=-1)
        l2 = torch.linalg.vector_norm(dn, ord=2, dim=-1)
        return l1, l2

    def register_trial(self, name: str):
        self.trials[name] = self.trials.get(name, 0) + 1

    def register_accept(self, name: str):
        self.accepts[name] = self.accepts.get(name, 0) + 1
        
    def update_proposal(self, move: str, dH: torch.Tensor, n_prop: torch.Tensor, n_curr: torch.Tensor) -> None:
        md = self.moves[move]
        dH_abs = dH.abs().detach()
        md.sum_abs_dH_prop += float(dH_abs.sum().item())
        l1, l2 = self._l1_l2_dn(n_prop.detach(), n_curr.detach())
        md.sum_dn_l1_prop += float(l1.sum().item())
        md.sum_dn_l2_prop += float(l2.sum().item())

    def update_accepted(self, move: str, dH_acc: torch.Tensor, n_prop_acc: torch.Tensor, n_curr_acc: torch.Tensor) -> None:
        md = self.moves[move]
        dH_abs = dH_acc.abs().detach()
        md.sum_abs_dH_acc += float(dH_abs.sum().item())
        l1, l2 = self._l1_l2_dn(n_prop_acc.detach(), n_curr_acc.detach())
        md.sum_dn_l1_acc += float(l1.sum().item())
        md.sum_dn_l2_acc += float(l2.sum().item())
        self._append_buffer(md.buf_abs_dH_acc, dH_abs)

    def finalize(self, trials: Dict[str, int], accepts: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """Return a JSON-serializable dict for logging/saving."""
        out: Dict[str, Dict[str, float]] = {}
        for move, md in self.moves.items():
            t = max(1, int(trials.get(move, 0)))
            a = max(0, int(accepts.get(move, 0)))
            # Avoid division by zero; proposals are counted even if none accepted.
            mean_abs_dH_prop = md.sum_abs_dH_prop / t
            mean_dn_l1_prop = md.sum_dn_l1_prop / t
            mean_dn_l2_prop = md.sum_dn_l2_prop / t

            mean_abs_dH_acc = md.sum_abs_dH_acc / max(1, a)
            mean_dn_l1_acc = md.sum_dn_l1_acc / max(1, a)
            mean_dn_l2_acc = md.sum_dn_l2_acc / max(1, a)

            # quantiles for accepted |ΔH|
            if len(md.buf_abs_dH_acc) > 0:
                cat = torch.cat(md.buf_abs_dH_acc, dim=0)
                q10 = float(torch.quantile(cat, 0.10).item())
                q50 = float(torch.quantile(cat, 0.50).item())
                q90 = float(torch.quantile(cat, 0.90).item())
            else:
                q10 = q50 = q90 = float("nan")

            out[move] = {
                "mean_abs_dH_prop": float(mean_abs_dH_prop),
                "mean_abs_dH_acc": float(mean_abs_dH_acc),
                "q10_abs_dH_acc": q10,
                "q50_abs_dH_acc": q50,
                "q90_abs_dH_acc": q90,
                "mean_dn_l1_prop": float(mean_dn_l1_prop),
                "mean_dn_l1_acc": float(mean_dn_l1_acc),
                "mean_dn_l2_prop": float(mean_dn_l2_prop),
                "mean_dn_l2_acc": float(mean_dn_l2_acc),
            }
        return out