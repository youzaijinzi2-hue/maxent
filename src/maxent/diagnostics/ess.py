import math
import torch


class AutocorrTime:
    @classmethod
    def integrated_autocorr_time(
        cls,
        x: torch.Tensor,
        max_lag: int | None = None,
        *,
        force_cpu_on_cuda: bool = True,
        max_fft_bytes: int = 512 * 1024 * 1024,
    ) -> torch.Tensor:
        """
        Estimate integrated autocorrelation time (IAT) for x of shape (R, L, D).

        This uses FFT-based ACF and averages ACF over D, but computes D in chunks
        to avoid large temporary allocations.
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (R, L, D), got {tuple(x.shape)}")

        R, L, D = x.shape
        if L < 2:
            return torch.ones((R,), device=x.device, dtype=torch.float64)

        if max_lag is None:
            max_lag = min(L // 2, 500)
        max_lag = int(max(1, min(max_lag, L - 1)))
        if max_lag <= 1:
            return torch.ones((R,), device=x.device, dtype=torch.float64)

        base_device = x.device
        work_device = torch.device("cpu") if (force_cpu_on_cuda and x.is_cuda) else base_device
        dtype = torch.float64

        nfft = 1 << ((2 * L - 1).bit_length())

        # Rough temporary-memory model per feature channel.
        # input+acf ~ R*L*16 bytes, fft spectrum ~ R*(nfft/2+1)*16 bytes.
        bytes_per_feature = R * (L * 16 + (nfft // 2 + 1) * 16)
        chunk_d = max(1, min(D, int(max_fft_bytes // max(bytes_per_feature, 1))))

        acf_r_sum = torch.zeros((R, L), device=work_device, dtype=dtype)
        processed_d = 0

        for d0 in range(0, D, chunk_d):
            d1 = min(D, d0 + chunk_d)
            d_chunk = d1 - d0

            x_chunk = x[:, :, d0:d1]
            if x_chunk.device != work_device:
                x_chunk = x_chunk.to(device=work_device, dtype=dtype)
            else:
                x_chunk = x_chunk.to(dtype=dtype)

            x_center = x_chunk - x_chunk.mean(dim=1, keepdim=True)  # (R, L, d_chunk)
            x2 = x_center.permute(0, 2, 1).reshape(-1, L)  # (R*d_chunk, L)

            Xf = torch.fft.rfft(x2, n=nfft, dim=1)
            acf_all = torch.fft.irfft(Xf * torch.conj(Xf), n=nfft, dim=1)[..., :L]
            acf_all = acf_all / torch.clamp(acf_all[:, 0:1], min=1e-30)

            acf_r_sum += acf_all.view(R, d_chunk, L).sum(dim=1)
            processed_d += d_chunk

            del x_chunk, x_center, x2, Xf, acf_all

        acf_r = acf_r_sum / float(processed_d)
        rho = acf_r[:, 1:max_lag]  # keep previous lag policy

        threshold = 1.96 / math.sqrt(L)
        pos_mask = rho > threshold
        full = pos_mask.all(dim=1)
        lengths = torch.where(
            full,
            torch.full((R,), pos_mask.size(1), device=work_device, dtype=torch.long),
            pos_mask.to(torch.int64).argmin(dim=1),
        )

        sums = torch.zeros(R, device=work_device, dtype=dtype)
        for i in range(R):
            li = int(lengths[i].item())
            if li > 0:
                sums[i] = rho[i, :li].sum()

        tau_int = torch.clamp(1.0 + 2.0 * sums, min=1.0)
        return tau_int.to(device=base_device, dtype=torch.float64)
