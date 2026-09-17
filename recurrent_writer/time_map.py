import torch

CHUNK_SPAN = 8.0

TIME_MAP = "affine"


def _validate(lo: float, hi: float) -> float:
    span = float(hi) - float(lo)
    if not span > 0.0:
        raise ValueError(f"time bounds require hi > lo, got lo={lo} hi={hi}")
    return span


class TimeMap:
    # map world time [lo, hi] to [-1, 1] with a matching inverse

    name = "base"

    def __init__(self, chunk_span: float = CHUNK_SPAN):
        if not float(chunk_span) > 0.0:
            raise ValueError(f"chunk_span must be > 0, got {chunk_span}")
        self.chunk_span = float(chunk_span)

    def to_norm(self, t, lo, hi):
        raise NotImplementedError

    def from_norm(self, tau, lo, hi):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(chunk_span={self.chunk_span})"


class AffineTimeMap(TimeMap):
    name = "affine"

    def to_norm(self, t, lo, hi):
        span = _validate(lo, hi)
        return 2.0 * (t - lo) / span - 1.0

    def from_norm(self, tau, lo, hi):
        span = _validate(lo, hi)
        return 0.5 * (tau + 1.0) * span + lo


class AdaptivePowerTimeMap(TimeMap):
    name = "adaptive_power"

    def exponent(self, lo, hi) -> float:
        return _validate(lo, hi) / self.chunk_span

    def to_norm(self, t, lo, hi):
        span = _validate(lo, hi)
        p = span / self.chunk_span
        # clamp rounding errors before taking a fractional power
        a = ((t - lo) / span).clamp(0.0, 1.0)
        return 2.0 * a.pow(p) - 1.0

    def from_norm(self, tau, lo, hi):
        span = _validate(lo, hi)
        p = span / self.chunk_span
        a = (0.5 * (tau + 1.0)).clamp(0.0, 1.0)
        return a.pow(1.0 / p) * span + lo


class LogAgeTimeMap(TimeMap):
    name = "log_age"

    def to_norm(self, t, lo, hi):
        span = _validate(lo, hi)
        c = self.chunk_span
        s = (hi - t).clamp(0.0, span)                  # age is zero at the current time
        denom = torch.log1p(torch.as_tensor(span / c, dtype=s.dtype))
        return 1.0 - 2.0 * torch.log1p(s / c) / denom

    def from_norm(self, tau, lo, hi):
        span = _validate(lo, hi)
        c = self.chunk_span
        denom = torch.log1p(torch.as_tensor(span / c, dtype=tau.dtype))
        frac = (0.5 * (1.0 - tau)).clamp(0.0, 1.0)
        s = c * torch.expm1(frac * denom)
        return hi - s


TIME_MAPS = {
    AffineTimeMap.name: AffineTimeMap,
    AdaptivePowerTimeMap.name: AdaptivePowerTimeMap,
    LogAgeTimeMap.name: LogAgeTimeMap,
}


def make_time_map(name: str, chunk_span: float = CHUNK_SPAN) -> TimeMap:
    if name not in TIME_MAPS:
        raise ValueError(
            f"unknown time_map {name!r}; expected one of {sorted(TIME_MAPS)}")
    return TIME_MAPS[name](chunk_span=chunk_span)
