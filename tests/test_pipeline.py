from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.cv.pipeline import CVPipeline
from app.cv.tracker import TargetTracker


def test_reset_clears_published_target_and_detector_state():
    detector_tracker = Mock()
    detector = SimpleNamespace(predictor=SimpleNamespace(trackers=[detector_tracker]))
    pipeline = CVPipeline(detector, TargetTracker(Mock()))
    pipeline.latest_frame = np.zeros((10, 10, 3))
    pipeline.target_found = True
    pipeline.fps = 20
    pipeline.reset()
    assert pipeline.latest_frame is None
    assert pipeline.snapshot["state"] == "idle"
    assert not pipeline.snapshot["found"]
    assert pipeline.snapshot["fps"] == 0
    detector_tracker.reset.assert_called_once()
