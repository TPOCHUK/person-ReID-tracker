"""Run warmup, measure frames and save a report."""

import time
from pathlib import Path

from .metrics_collector import MetricsCollector
from .report_generator import ReportGenerator
from .system_monitor import SystemMonitor

TEST_CONFIGS = {
    "yolo_inference": {
        "description": "Лише YOLO + вбудований трекер (без ReID і оверлею)",
        "duration": 300,
        "warmup": 10,
        "mode": "inference",
    },
    "full_pipeline": {
        "description": "Повний CV pipeline: YOLO + утримання цілі + ReID + оверлей",
        "duration": 300,
        "warmup": 10,
        "mode": "full",
    },
    "system_load": {
        "description": "Повний pipeline з акцентом на ресурси (CPU/RAM/GPU)",
        "duration": 300,
        "warmup": 10,
        "mode": "full",
    },
    "stability": {
        "description": "Довгий запуск: стабільність FPS і тренд пам'яті (витік?)",
        "duration": 1800,
        "warmup": 30,
        "mode": "full",
    },
}


def _process_frame(pipeline, frame, mode):
    """Return timings, target visibility and detection count."""
    if mode == "inference":
        boxes, inference_ms = pipeline.infer(frame)
        return {"inference_ms": inference_ms}, False, len(boxes) if boxes is not None else 0

    visible = pipeline.process(frame)
    return dict(pipeline.last_timings), visible, pipeline.last_detections


def run_benchmark(
    *,
    pipeline,
    source,
    test_name: str,
    output_dir: str,
    config_snapshot: dict,
    hardware: dict,
    duration_s: float = None,
    warmup_s: float = None,
    log=print,
):
    if test_name not in TEST_CONFIGS:
        raise ValueError(f"Unknown benchmark: {test_name}. Available: {list(TEST_CONFIGS)}")

    cfg = dict(TEST_CONFIGS[test_name])
    if duration_s is not None:
        cfg["duration"] = duration_s
    if warmup_s is not None:
        cfg["warmup"] = warmup_s

    if cfg["duration"] <= 0 or cfg["warmup"] < 0:
        raise ValueError("Duration must be positive and warmup cannot be negative")

    mode = cfg["mode"]
    monitor = SystemMonitor(interval=1.0)
    collector = MetricsCollector()

    log(f"[{test_name}] прогрів {cfg['warmup']} с…")
    warmup_end = time.perf_counter() + cfg["warmup"]
    while time.perf_counter() < warmup_end:
        ok, frame = source.read()
        if not ok:
            raise RuntimeError("Source ended during warmup")
        _process_frame(pipeline, frame, mode)

    log(f"[{test_name}] вимірювання {cfg['duration']} с…")
    monitor.start()

    start = time.perf_counter()
    collector.reset(start)
    next_log = start + 10

    try:
        while True:
            now = time.perf_counter()
            if now - start >= cfg["duration"]:
                break

            t0 = time.perf_counter()
            ok, frame = source.read()
            capture_ms = (time.perf_counter() - t0) * 1000
            if not ok:
                raise RuntimeError("Source ended during measurement")

            timings, visible, detections = _process_frame(pipeline, frame, mode)
            timings["capture_ms"] = capture_ms
            collector.record(timings, detections=detections, visible=visible)

            if now >= next_log:
                elapsed = now - start
                log(f"  {elapsed:6.0f} / {cfg['duration']} с, кадрів: {len(collector.timestamps)}")
                next_log = now + 10

        duration = time.perf_counter() - start
    finally:
        samples = monitor.stop()

    report_dir = Path(output_dir)
    writer = ReportGenerator(report_dir)

    json_path, md_path = writer.write(
        test_name=test_name,
        description=cfg["description"],
        duration_s=duration,
        warmup_s=cfg["warmup"],
        source=source.describe(),
        config=config_snapshot,
        hardware=hardware,
        performance=collector.summary(duration),
        system_samples=samples,
    )

    log(f"[{test_name}] готово: {json_path}")
    return json_path, md_path
