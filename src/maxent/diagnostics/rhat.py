import numpy as np
import torch


class GelmanRubin:
    @staticmethod
    def _rhat_from_chains_np(arr: np.ndarray) -> np.ndarray:
        """
        Classical Gelman-Rubin R-hat from chains array of shape (C, L, D).
        Returns ndarray (D,). Non-computable bins are NaN.
        """
        C, L, D = arr.shape
        out = np.full((D,), np.nan, dtype=np.float64)
        if C < 2 or L < 2:
            return out

        chain_means = arr.mean(axis=1)       # (C, D)
        mean_all = chain_means.mean(axis=0)  # (D,)
        B = (L / (C - 1)) * ((chain_means - mean_all) ** 2).sum(axis=0)  # (D,)

        chain_vars = arr.var(axis=1, ddof=1)  # (C, D)
        W = chain_vars.mean(axis=0)            # (D,)

        V_hat = ((L - 1) / L) * W + (1.0 / L) * B
        good = W > 0.0
        ratio = np.zeros_like(W)
        ratio[good] = V_hat[good] / W[good]
        ratio = np.clip(ratio, a_min=0.0, a_max=None)
        out[good] = np.sqrt(ratio[good])
        return out

    @classmethod
    def gelman_rubin_rhat(cls, samples: torch.Tensor) -> torch.Tensor:
        """
        Backward-compatible API.
        samples shape: (R, L, D) interpreted as R independent chains.
        """
        arr = samples.detach().cpu().numpy()
        rhat = cls._rhat_from_chains_np(arr)
        return torch.from_numpy(rhat)

    @classmethod
    def split_rhat_per_replica(
        cls,
        samples: torch.Tensor,
        n_split: int = 4,
    ) -> torch.Tensor:
        """
        Split-Rhat within each replica chain.

        Args:
            samples: Tensor of shape (R, L, D), one chain per replica/alpha.
            n_split: number of temporal splits per chain (>=2).

        Returns:
            Tensor of shape (R, D). Entries are NaN if not computable.
        """
        if n_split < 2:
            raise ValueError(f"n_split must be >=2, got {n_split}")

        arr = samples.detach().cpu().numpy()  # (R, L, D)
        R, L, D = arr.shape
        out = np.full((R, D), np.nan, dtype=np.float64)

        seg_len = L // n_split
        if seg_len < 2:
            return torch.from_numpy(out)

        L_eff = seg_len * n_split
        arr = arr[:, :L_eff, :]                  # (R, L_eff, D)
        arr = arr.reshape(R, n_split, seg_len, D)  # (R, C=n_split, L_seg, D)

        for r in range(R):
            out[r] = cls._rhat_from_chains_np(arr[r])

        return torch.from_numpy(out)
