from unittest.mock import Mock

import pytest

from app.benchmark import benchmark_runner as runner


def test_measurement_error_stops_monitor(monkeypatch, tmp_path):
    monitor = Mock()
    monkeypatch.setattr(runner, "SystemMonitor", Mock(return_value=monitor))
    source = Mock()
    source.read.return_value = (False, None)
    with pytest.raises(RuntimeError, match="measurement"):
        runner.run_benchmark(
            pipeline=Mock(),
            source=source,
            test_name="full_pipeline",
            output_dir=str(tmp_path),
            config_snapshot={},
            hardware={},
            duration_s=1,
            warmup_s=0,
            log=lambda _: None,
        )
    monitor.stop.assert_called_once()


def test_inference_mode_does_not_run_full_pipeline():
    pipeline = Mock()
    pipeline.infer.return_value = ([], 12.0)
    timings, visible, count = runner._process_frame(pipeline, object(), "inference")
    assert timings == {"inference_ms": 12.0}
    assert not visible and count == 0
    pipeline.process.assert_not_called()
