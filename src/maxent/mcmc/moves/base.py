from dataclasses import dataclass
import torch


@dataclass
class MoveResult:
    log_lr: torch.Tensor
    proposal_n: torch.Tensor
    proposal_x: torch.Tensor
    proposal_H: torch.Tensor


class BaseMove:
    """
    全ての move の基底クラス
    """
    def propose(self, *args, **kwargs) -> MoveResult:
        raise NotImplementedError