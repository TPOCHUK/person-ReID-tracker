from app.benchmark.metrics_collector import MetricsCollector, per_second_fps, stats


def test_stats_percentiles():
    s = stats(list(range(1, 101)))  # 1..100

    assert s["min"] == 1
    assert s["max"] == 100
    assert s["p50"] == 50.5
    assert s["p95"] == 95.05
    assert s["p99"] == 99.01


def test_stats_empty_returns_none_values():
    s = stats([])
    assert s["mean"] is None and s["p95"] is None


def test_per_second_fps_fills_pauses_with_zero_and_drops_partial_second():
    start = 1000.0
    timestamps = [
        1000.1,
        1000.5,
        1000.9,  # секунда 0: 3 кадри
        1002.2,
        1002.4,  # секунда 2: 2 кадри (секунда 1 — пауза)
        1003.1,
        1003.2,
        1003.3,
        1003.4,  # неповна секунда 3 (duration = 3.5)
    ]

    res = per_second_fps(timestamps, start, duration_s=3.5)

    assert res["seconds"] == 3  # неповна секунда відкинута
    assert res["min"] == 0  # пауза видна
    assert res["max"] == 3
    assert res["mean"] == round(5 / 3, 2)


def test_per_second_fps_too_short_returns_none():
    assert per_second_fps([1.0], 1.0, duration_s=0.5) is None


def test_collector_summary_and_visibility():
    c = MetricsCollector()
    c.reset(start=0.0)

    for i in range(10):
        c.record(
            {"inference_ms": 10.0 + i, "total_ms": 12.0 + i},
            detections=1 if i % 2 == 0 else 0,
            visible=i < 5,
            timestamp=i * 0.1,
        )

    summary = c.summary(duration_s=1.0)

    assert summary["frames"] == 10
    assert summary["fps"]["overall"] == 10.0
    assert summary["latency_ms"]["inference_ms"]["min"] == 10.0
    assert summary["detection"]["target_visible_percent"] == 50.0
    assert summary["detection"]["frames_with_person_percent"] == 50.0


def test_collector_empty_summary_does_not_crash():
    c = MetricsCollector()
    c.reset(start=0.0)
    assert c.summary(duration_s=5.0) == {"frames": 0}
