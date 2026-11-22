from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DualAveragingStepAdapter:
    """Dual-averaging step-size adapter (Nesterov/Hoffman-Gelman style)."""

    step: float
    target_accept: float
    step_min: float = 1e-3
    step_max: float = 1e1
    gamma: float = 0.05
    t0: float = 10.0
    kappa: float = 0.75
    step_init: float = 0.0
    h_bar: float = 0.0
    log_step: float = 0.0
    log_step_avg: float = 0.0
    mu: float = 0.0
    iters: int = 0
    frozen: bool = False

    def __post_init__(self):
        self.step = float(self.step)
        if (not math.isfinite(self.step)) or (self.step <= 0.0):
            self.step = 0.05
        self.target_accept = float(self.target_accept)
        if not math.isfinite(self.target_accept):
            self.target_accept = 0.50
        self.target_accept = min(max(self.target_accept, 1e-3), 1.0 - 1e-3)
        self.step_min = float(self.step_min)
        self.step_max = float(self.step_max)
        if (not math.isfinite(self.step_min)) or (self.step_min <= 0.0):
            self.step_min = 1e-3
        if (not math.isfinite(self.step_max)) or (self.step_max <= self.step_min):
            self.step_max = max(1e1, self.step_min * 10.0)
        self.step = min(max(self.step, self.step_min), self.step_max)
        self.gamma = float(self.gamma)
        if (not math.isfinite(self.gamma)) or (self.gamma <= 0.0):
            self.gamma = 0.05
        self.t0 = float(self.t0)
        if (not math.isfinite(self.t0)) or (self.t0 < 0.0):
            self.t0 = 10.0
        self.kappa = float(self.kappa)
        if (not math.isfinite(self.kappa)) or (self.kappa <= 0.5) or (self.kappa > 1.0):
            self.kappa = 0.75
        self.step_init = self.step
        self.log_step = math.log(self.step)
        self.log_step_avg = self.log_step
        self.mu = math.log(10.0 * self.step)

    def update(self, accept_rate: float) -> float:
        if self.frozen:
            return self.step
        if not math.isfinite(accept_rate):
            return self.step
        a = min(max(float(accept_rate), 0.0), 1.0)
        self.iters += 1
        eta = 1.0 / (self.iters + self.t0)
        self.h_bar = (1.0 - eta) * self.h_bar + eta * (self.target_accept - a)
        self.log_step = self.mu - (math.sqrt(self.iters) / self.gamma) * self.h_bar
        w = self.iters ** (-self.kappa)
        self.log_step_avg = w * self.log_step + (1.0 - w) * self.log_step_avg
        self.step = math.exp(self.log_step)
        self.step = min(max(self.step, self.step_min), self.step_max)
        return self.step

    def freeze(self) -> float:
        if not self.frozen:
            self.step = math.exp(self.log_step_avg)
            self.step = min(max(self.step, self.step_min), self.step_max)
            self.frozen = True
        return self.step
