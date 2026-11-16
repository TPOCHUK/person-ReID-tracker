"""Run benchmarks without the web server."""

import argparse
import sys
from pathlib import Path

from app.benchmark.benchmark_runner import TEST_CONFIGS, run_benchmark
from app.benchmark.hardware import collect_hardware_info
from app.benchmark.sources import FrameSource
from app.config.settings import settings
from app.cv.factory import build_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CV pipeline benchmark")

    p.add_argument("--list", action="store_true", help="показати доступні тести і вийти")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--test", choices=list(TEST_CONFIGS), help="один тест")
    group.add_argument("--all", action="store_true", help="усі тести по черзі (stability — 30 хв!)")

    p.add_argument(
        "--source",
        default=None,
        help="індекс камери (0) або шлях до відеофайлу; за замовчуванням — CAMERA_SOURCE",
    )
    p.add_argument("--duration", type=float, default=None, help="перевизначити тривалість, с")
    p.add_argument("--warmup", type=float, default=None, help="перевизначити прогрів, с")
    p.add_argument("--output", default=None, help="папка для звітів")

    # Перевизначення налаштувань без редагування .env
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--cv-threads", type=int, default=None)
    p.add_argument("--tracker", default=None, help="напр. bytetrack.yaml або botsort.yaml")

    return p.parse_args(argv)


def build_settings(args):
    overrides = {}
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.device is not None:
        overrides["device"] = args.device
    if args.torch_threads is not None:
        overrides["torch_threads"] = args.torch_threads
    if args.cv_threads is not None:
        overrides["cv_threads"] = args.cv_threads
    if args.tracker is not None:
        overrides["yolo_tracker"] = args.tracker
    if args.output is not None:
        overrides["benchmark_output_dir"] = args.output
    return type(settings)(**(settings.model_dump() | overrides))


def config_snapshot(s):
    """Усі параметри, що впливають на цифри. Попадають у звіт."""
    values = s.model_dump(exclude={"camera_source", "benchmark_output_dir"})
    for key in ("yolo_model", "reid_model"):
        values[key] = Path(values[key]).name
    values["half"] = s.half and s.device.startswith("cuda")
    return values


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        for name, cfg in TEST_CONFIGS.items():
            print(
                f"{name:16} {cfg['duration']:5d} с  (прогрів {cfg['warmup']} с)  {cfg['description']}"
            )
        return 0

    if not args.test and not args.all:
        print("Вкажіть --test <назва> або --all (див. --list)", file=sys.stderr)
        return 2

    if args.duration is not None and args.duration <= 0:
        raise ValueError("Duration must be positive")
    if args.warmup is not None and args.warmup < 0:
        raise ValueError("Warmup cannot be negative")
    s = build_settings(args)
    tests = list(TEST_CONFIGS) if args.all else [args.test]
    source_arg = args.source if args.source is not None else s.camera_source

    hardware = collect_hardware_info()
    snapshot = config_snapshot(s)

    for name in tests:
        pipeline = build_pipeline(s)
        source = FrameSource(source_arg, s.camera_width, s.camera_height)
        try:
            run_benchmark(
                pipeline=pipeline,
                source=source,
                test_name=name,
                output_dir=s.benchmark_output_dir,
                config_snapshot=snapshot,
                hardware=hardware,
                duration_s=args.duration,
                warmup_s=args.warmup,
            )
        finally:
            source.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
