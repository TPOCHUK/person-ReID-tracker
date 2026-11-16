"""Обробка кадру: YOLO, наш трекер, рамка і готовий стан для API."""

import time
from dataclasses import dataclass

import cv2

from app.cv import tracker as trk
from app.cv.types import detections_from_boxes


@dataclass(frozen=True)
class PipelineConfig:
    imgsz: int = 256
    detector_conf: float = 0.25
    device: str = "cpu"
    half: bool = False
    tracker_yaml: str = "bytetrack.yaml"
    max_lost_frames: int = 30


class CVPipeline:
    def __init__(self, detector, tracker, config: PipelineConfig = None):
        self.detector = detector
        self.tracker = tracker
        self.cfg = config or PipelineConfig()

        self.target_found = False
        self.latest_frame = None  # Ready frame for the stream.
        self.frame_seq = 0  # Skip duplicate frames in the stream.

        self.fps = 0.0
        self._last_time = None

        self.last_detections = 0
        self.last_timings = {}
        self.snapshot = self._target_snapshot()

    def reset(self):
        self.tracker.reset()
        self.target_found = False
        self.latest_frame = None
        self.fps = 0.0
        self._last_time = None
        self.last_timings = {}
        self.last_detections = 0
        predictor = getattr(self.detector, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()
        self.snapshot = self._target_snapshot()

    def _target_snapshot(self):
        return {
            "found": bool(self.target_found),
            "state": self.tracker.state,
            "target_id": self.tracker.target_id,
            "bbox": self.tracker.target_bbox,
            "lost_frames": self.tracker.lost_frames,
            "fps": round(self.fps, 1),
            "inference_ms": round(self.last_timings.get("inference_ms", 0.0), 2),
            "total_ms": round(self.last_timings.get("total_ms", 0.0), 2),
        }

    def infer(self, frame):
        """Run YOLO tracking and measure its time."""
        start = time.perf_counter()

        kwargs = dict(
            persist=True,
            verbose=False,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.detector_conf,
            device=self.cfg.device,
            classes=[0],  # Person only.
            tracker=self.cfg.tracker_yaml,
        )
        if self.cfg.half:  # CUDA only.
            kwargs["half"] = True

        results = self.detector.track(frame, **kwargs)

        return results[0].boxes, (time.perf_counter() - start) * 1000

    def process(self, frame) -> bool:
        """Повний цикл для одного кадру. Повертає, чи видно ціль."""
        t_start = time.perf_counter()

        boxes, inference_ms = self.infer(frame)
        t_infer = time.perf_counter()

        detections = detections_from_boxes(boxes)
        t_post = time.perf_counter()

        h, w = frame.shape[:2]
        visible = self.tracker.step(detections, frame, w // 2, h // 2)
        t_track = time.perf_counter()

        _draw_overlay(frame, self.tracker, self.cfg.max_lost_frames)
        t_draw = time.perf_counter()

        # Do not modify the frame after publishing.
        self.latest_frame = frame
        self.frame_seq += 1
        self.target_found = visible
        self.last_detections = len(detections)

        self.last_timings = {
            "inference_ms": inference_ms,
            "post_ms": (t_post - t_infer) * 1000,
            "tracking_ms": (t_track - t_post) * 1000,
            "draw_ms": (t_draw - t_track) * 1000,
            "total_ms": (t_draw - t_start) * 1000,
        }

        self._update_fps(t_draw)
        # Publish status only after the frame is ready.
        self.snapshot = self._target_snapshot()
        return visible

    def _update_fps(self, now: float):
        # Smooth the displayed FPS.
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0:
                instant = 1.0 / dt
                self.fps = instant if self.fps == 0 else 0.9 * self.fps + 0.1 * instant
        self._last_time = now


def _draw_overlay(frame, tracker, max_lost):
    """Draw the tracker state on this frame."""

    state = tracker.state
    bbox = tracker.target_bbox

    if state == trk.IDLE:
        cv2.putText(frame, "NO TARGET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return

    if state == trk.TRACKING and bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"TARGET #{tracker.target_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return

    if state == trk.RECOVERING:
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"REID RECOVERY {tracker.lost_frames}/{max_lost}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        return

    # LOST
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        frame,
        f"TARGET LOST {tracker.lost_frames}/{max_lost}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
