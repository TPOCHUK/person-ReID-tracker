import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .metrics_collector import stats

SYSTEM_METRICS = (
    "cpu_process_percent",
    "cpu_process_norm_percent",
    "cpu_system_percent",
    "ram_process_mb",
    "ram_system_percent",
    "gpu_percent",
    "vram_mb",
)


def _system_summary(samples: list) -> dict:
    result = {}
    for metric in SYSTEM_METRICS:
        values = [s[metric] for s in samples if s.get(metric) is not None]
        result[metric] = stats(values) if values else None
    return result


def _ram_trend_mb_per_min(samples: list):
    """Estimate RSS growth. A positive slope alone does not prove a leak."""
    pts = [(s["t"], s["ram_process_mb"]) for s in samples if s.get("ram_process_mb") is not None]
    if len(pts) < 3:
        return None
    t = np.array([p[0] for p in pts])
    ram = np.array([p[1] for p in pts])
    slope_per_s = np.polyfit(t, ram, 1)[0]
    return round(float(slope_per_s * 60), 3)


class ReportGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        test_name: str,
        description: str,
        duration_s: float,
        warmup_s: float,
        source: str,
        config: dict,
        hardware: dict,
        performance: dict,
        system_samples: list,
    ) -> tuple:
        system = _system_summary(system_samples)
        ram_trend = _ram_trend_mb_per_min(system_samples)

        data = {
            "metadata": {
                "test_name": test_name,
                "description": description,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(duration_s, 2),
                "warmup_seconds": warmup_s,
                "source": source,
                "system_samples": len(system_samples),
            },
            "config": config,
            "hardware": hardware,
            "performance": performance,
            "system": system,
            "ram_trend_mb_per_min": ram_trend,
        }

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = self.output_dir / f"{test_name}_{stamp}"

        json_path = base.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        md_path = base.with_suffix(".md")
        md_path.write_text(_to_markdown(data), encoding="utf-8")

        return json_path, md_path


def _to_markdown(d: dict) -> str:
    m, perf, sysm = d["metadata"], d["performance"], d["system"]
    lat = perf.get("latency_ms", {})
    fps = perf.get("fps", {})
    ps = fps.get("per_second") or {}

    lines = [
        f"# Benchmark: {m['test_name']}",
        "",
        m["description"],
        "",
        f"- Джерело: {m['source']}",
        f"- Тривалість: {m['duration_seconds']} с (прогрів {m['warmup_seconds']} с)",
        f"- Кадрів: {perf.get('frames')}",
        "",
        "## FPS",
        "",
        f"- Середній за весь тест: **{fps.get('overall')}**",
        f"- По секундах: mean {ps.get('mean')}, min {ps.get('min')}, p5 {ps.get('p5')}, max {ps.get('max')}",
        "",
        "## Латентність, мс",
        "",
        "| Етап | mean | p50 | p95 | p99 | max |",
        "|---|---|---|---|---|---|",
    ]

    for stage, s in lat.items():
        lines.append(
            f"| {stage} | {s['mean']} | {s['p50']} | {s['p95']} | {s['p99']} | {s['max']} |"
        )

    cpu = sysm.get("cpu_process_norm_percent") or {}
    ram = sysm.get("ram_process_mb") or {}
    lines += [
        "",
        "## Ресурси",
        "",
        f"- CPU процесу (нормовано по ядрах), mean/max: {cpu.get('mean')} / {cpu.get('max')} %",
        f"- RAM процесу, mean/max: {ram.get('mean')} / {ram.get('max')} MB",
        f"- Тренд RAM: {d['ram_trend_mb_per_min']} MB/хв",
        "",
        "## Залізо",
        "",
    ]
    for k, v in d["hardware"].items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines) + "\n"
