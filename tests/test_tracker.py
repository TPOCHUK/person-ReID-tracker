"""Tracker scenarios with fake appearance embeddings."""

import numpy as np

from app.cv.reid import ReIDManager
from app.cv.tracker import IDLE, LOST, RECOVERING, TRACKING, TargetTracker, TrackerConfig
from app.cv.types import Detection, detections_from_arrays

FRAME_CX, FRAME_CY = 320, 240
PERSON_A_X = 100  # особа A: x1 = 100
PERSON_B_X = 2100  # особа B: x1 = 2100 (далеко від центру)


class FakeReID:
    """Той самий інтерфейс, що й ReIDManager, але без моделі."""

    def get_embedding(self, frame, x1, y1, x2, y2):
        if x2 - x1 < 4 or y2 - y1 < 8:
            return None
        identity = int(x1) // 1000
        vec = np.zeros(4, dtype=np.float32)
        vec[identity] = 1.0
        return vec

    embedding_alpha = 0.2
    update_embedding = ReIDManager.update_embedding

    def find_matching_target(self, target, embeddings):
        best_id, best = None, -1.0
        for tid, emb in embeddings.items():
            sim = ReIDManager.cosine_similarity(target, emb)
            if sim > best:
                best_id, best = tid, sim
        return best_id if best >= 0.7 else None


def person(track_id, x1, conf=0.9, cls=0):
    """Людина з bbox 100x200 з лівим верхнім кутом (x1, 100)."""
    return Detection(track_id, cls, conf, x1, 100, x1 + 100, 300)


def make_tracker(**overrides):
    cfg = dict(
        conf_threshold=0.6,
        reid_update_interval=10,
        reid_update_conf=0.8,
        reid_recovery_start=2,
        reid_recovery_interval=1,
        max_lost_frames=5,
        recovery_max_candidates=8,
    )
    cfg.update(overrides)
    return TargetTracker(FakeReID(), TrackerConfig(**cfg))


def test_acquires_person_closest_to_center():
    tr = make_tracker()
    dets = [person(2, PERSON_B_X), person(1, PERSON_A_X)]

    visible = tr.step(dets, None, FRAME_CX, FRAME_CY)

    assert visible is True
    assert tr.target_id == 1  # A ближче до центру
    assert tr.state == TRACKING


def test_ignores_non_person_and_low_confidence():
    tr = make_tracker()
    dets = [
        person(1, PERSON_A_X, cls=2),  # не людина
        person(2, PERSON_A_X, conf=0.5),  # нижче порогу
    ]

    assert tr.step(dets, None, FRAME_CX, FRAME_CY) is False
    assert tr.target_id is None
    assert tr.state == IDLE


def test_box_too_small_gets_no_embedding_and_is_not_acquired():
    tr = make_tracker()
    tiny = Detection(1, 0, 0.9, 100, 100, 102, 103)  # 2x3 px

    assert tr.step([tiny], None, FRAME_CX, FRAME_CY) is False
    assert tr.target_id is None


def test_recovers_same_person_after_track_id_changes():
    tr = make_tracker()

    # 1) A захоплено
    tr.step([person(1, PERSON_A_X)], None, FRAME_CX, FRAME_CY)
    assert tr.target_id == 1

    # 2) A зникла, є лише B → починаємо втрачати
    assert tr.step([person(2, PERSON_B_X)], None, FRAME_CX, FRAME_CY) is False
    assert tr.state == LOST

    # 3) B усе ще тут, lost дійшов до порогу → спроба ReID, але не та людина
    assert tr.step([person(2, PERSON_B_X)], None, FRAME_CX, FRAME_CY) is False
    assert tr.state == RECOVERING

    # 4) A повертається з НОВИМ track_id=7 → трекер має її впізнати
    visible = tr.step([person(7, PERSON_A_X), person(2, PERSON_B_X)], None, FRAME_CX, FRAME_CY)

    assert visible is True
    assert tr.target_id == 7
    assert tr.state == TRACKING
    assert tr.lost_frames == 0


def test_forgets_target_after_max_lost_frames():
    tr = make_tracker(max_lost_frames=3)
    tr.step([person(1, PERSON_A_X)], None, FRAME_CX, FRAME_CY)

    for _ in range(3):
        tr.step([], None, FRAME_CX, FRAME_CY)

    assert tr.target_id is None
    assert tr.target_embedding is None
    assert tr.state == IDLE


def test_embedding_updates_periodically_for_confident_detections():
    tr = make_tracker(reid_update_interval=2, reid_update_conf=0.8)
    tr.step([person(1, PERSON_A_X, conf=0.95)], None, FRAME_CX, FRAME_CY)
    before = tr.target_embedding.copy()

    # Підміняємо fake так, щоб новий ембедінг відрізнявся від старого
    class ShiftedReID(FakeReID):
        def get_embedding(self, frame, x1, y1, x2, y2):
            v = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            return v

    tr.reid = ShiftedReID()
    for _ in range(2):
        tr.step([person(1, PERSON_A_X, conf=0.95)], None, FRAME_CX, FRAME_CY)

    assert not np.allclose(tr.target_embedding, before)


def test_detections_from_arrays_handles_missing_ids():
    xyxy = np.array([[10.7, 20.2, 50.9, 80.1]])
    dets = detections_from_arrays(xyxy, conf=[0.9], cls=[0], ids=None)

    assert dets[0].track_id is None
    assert dets[0].bbox == (10, 20, 50, 80)
    assert dets[0].center == (30, 50)
