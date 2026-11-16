"""Collect frame timings and FPS statistics."""

import time

import numpy as np

# Timing stages in report order.
STAGES = (
    "capture_ms",
    "inference_ms",
    "post_ms",
    "tracking_ms",
    "draw_ms",
    "total_ms",
)


def stats(values) -> dict:
    """Return summary statistics, or None values for empty input."""
    if len(values) == 0:
        return {k: None for k in ("mean", "std", "min", "p50", "p95", "p99", "max")}

    a = np.asarray(values, dtype=np.float64)
    p50, p95, p99 = np.percentile(a, [50, 95, 99])

    return {
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std()), 3),
        "min": round(float(a.min()), 3),
        "p50": round(float(p50), 3),
        "p95": round(float(p95), 3),
        "p99": round(float(p99), 3),
        "max": round(float(a.max()), 3),
    }


def per_second_fps(timestamps, start: float, duration_s: float):
    """Count frames in complete seconds, including pauses."""
    full_seconds = int(duration_s)
    if full_seconds < 1:
        return None

    counts = np.zeros(full_seconds, dtype=np.int64)

    if len(timestamps):
        idx = np.floor(np.asarray(timestamps) - start).astype(np.int64)
        valid = idx[(idx >= 0) & (idx < full_seconds)]
        np.add.at(counts, valid, 1)

    return {
        "seconds": full_seconds,
        "mean": round(float(counts.mean()), 2),
        "std": round(float(counts.std()), 2),
        "min": int(counts.min()),
        "p5": round(float(np.percentile(counts, 5)), 2),
        "max": int(counts.max()),
    }


class MetricsCollector:
    def __init__(self):
        self.reset()

    def reset(self, start: float = None):
        self.start = time.perf_counter() if start is None else start
        self.timestamps = []
        self.values = {stage: [] for stage in STAGES}
        self.detections = []
        self.visible = []

    def record(self, timings: dict, detections: int, visible: bool, timestamp: float = None):
        self.timestamps.append(time.perf_counter() if timestamp is None else timestamp)
        for stage in STAGES:
            if stage in timings:
                self.values[stage].append(float(timings[stage]))
        self.detections.append(int(detections))
        self.visible.append(bool(visible))

    def summary(self, duration_s: float) -> dict:
        n = len(self.timestamps)

        if n == 0 or duration_s <= 0:
            return {"frames": n}

        detections = np.asarray(self.detections)
        visible = np.asarray(self.visible)

        return {
            "frames": n,
            "fps": {
                "overall": round(n / duration_s, 2),
                "per_second": per_second_fps(self.timestamps, self.start, duration_s),
            },
            "latency_ms": {stage: stats(vals) for stage, vals in self.values.items() if vals},
            "detection": {
                "avg_persons_per_frame": round(float(detections.mean()), 3),
                "frames_with_person_percent": round(float((detections > 0).mean() * 100), 2),
                "target_visible_percent": round(float(visible.mean() * 100), 2),
            },
        }
