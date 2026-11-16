"""Утримуємо одну ціль: вибір, втрата, ReID-пошук і скидання."""

import math
from dataclasses import dataclass
from typing import Optional

from app.cv.types import Detection

IDLE = "idle"
TRACKING = "tracking"
LOST = "lost"
RECOVERING = "recovering"


@dataclass(frozen=True)
class TrackerConfig:
    conf_threshold: float = 0.6
    reid_update_interval: int = 10
    reid_update_conf: float = 0.8
    reid_recovery_start: int = 5
    reid_recovery_interval: int = 5
    max_lost_frames: int = 30
    recovery_max_candidates: int = 8


class TargetTracker:
    def __init__(self, reid, config: Optional[TrackerConfig] = None):
        self.reid = reid
        self.cfg = config or TrackerConfig()
        self.reset()

    def reset(self):
        self.target_id = None
        self.target_embedding = None
        self.target_bbox = None

        self.lost_frames = 0
        self.frames_since_reid_update = 0
        self.frames_since_recovery = 0

        # Was recovery tried since the last sighting?
        self.recovery_attempted = False
        self.state = IDLE

    def step(self, detections, frame, frame_cx: int, frame_cy: int) -> bool:
        """Оновлює стан за кадр. Повертає True, якщо ціль видно зараз."""

        candidates = [d for d in detections if self._is_candidate(d)]

        if self.target_id is None:
            self._acquire(candidates, frame, frame_cx, frame_cy)
            visible = self.target_id is not None

        else:
            visible = self._check(candidates, frame)
            self._update_lost(visible)

            if not visible and self._recover(candidates, frame):
                visible = True

            self._forget_if_needed()

        self._update_state(visible)
        return visible

    def _is_candidate(self, det: Detection) -> bool:
        # Keep confident people with a tracker ID.
        return det.cls == 0 and det.conf > self.cfg.conf_threshold and det.track_id is not None

    def _acquire(self, candidates, frame, frame_cx, frame_cy):
        """Pick the person closest to the frame center."""
        if not candidates:
            return

        ordered = sorted(
            candidates,
            key=lambda d: math.hypot(d.center[0] - frame_cx, d.center[1] - frame_cy),
        )

        for det in ordered:
            embedding = self.reid.get_embedding(frame, *det.bbox)
            if embedding is None:
                continue  # Try the next usable crop.

            self.target_id = det.track_id
            self.target_bbox = det.bbox
            self.target_embedding = embedding
            self.lost_frames = 0
            self.frames_since_reid_update = 0
            return

    def _check(self, candidates, frame) -> bool:
        """Check the current ID and refresh its appearance."""
        for det in candidates:
            if det.track_id != self.target_id:
                continue

            self.target_bbox = det.bbox
            self.frames_since_reid_update += 1

            if (
                self.frames_since_reid_update >= self.cfg.reid_update_interval
                and det.conf >= self.cfg.reid_update_conf
            ):
                embedding = self.reid.get_embedding(frame, *det.bbox)
                if embedding is not None:
                    self.target_embedding = self.reid.update_embedding(
                        self.target_embedding, embedding
                    )
                    self.frames_since_reid_update = 0

            return True

        return False

    def _update_lost(self, visible: bool):
        if visible:
            self.lost_frames = 0
            self.frames_since_recovery = 0
        else:
            self.lost_frames += 1
            self.frames_since_recovery += 1

    def _recover(self, candidates, frame) -> bool:
        """Пошук втраченої цілі серед людей у кадрі за ембедінгом."""

        if self.lost_frames < self.cfg.reid_recovery_start:
            return False

        if self.frames_since_recovery < self.cfg.reid_recovery_interval:
            return False

        self.frames_since_recovery = 0
        self.recovery_attempted = True

        if not candidates:
            return False

        candidates = self._nearest_to_last_known(candidates)

        embeddings = {}
        for det in candidates:
            embedding = self.reid.get_embedding(frame, *det.bbox)
            if embedding is not None:
                embeddings[det.track_id] = embedding

        if not embeddings:
            return False

        matched_id = self.reid.find_matching_target(self.target_embedding, embeddings)
        if matched_id is None:
            return False

        self.target_id = matched_id
        self.target_bbox = next(d.bbox for d in candidates if d.track_id == matched_id)
        self.lost_frames = 0
        self.frames_since_recovery = 0
        self.recovery_attempted = False
        return True

    def _nearest_to_last_known(self, candidates):
        if self.target_bbox is None:
            return candidates[: self.cfg.recovery_max_candidates]

        x1, y1, x2, y2 = self.target_bbox
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2

        ordered = sorted(
            candidates,
            key=lambda d: math.hypot(d.center[0] - lx, d.center[1] - ly),
        )
        return ordered[: self.cfg.recovery_max_candidates]

    def _forget_if_needed(self):
        if self.lost_frames < self.cfg.max_lost_frames:
            return
        self.reset()

    def _update_state(self, visible: bool):
        if self.target_id is None:
            self.state = IDLE
        elif visible:
            self.state = TRACKING
            self.recovery_attempted = False
        elif self.recovery_attempted:
            self.state = RECOVERING
        else:
            self.state = LOST
