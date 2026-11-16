"""Compare people using normalized appearance embeddings."""

from typing import Optional

import numpy as np

MIN_BOX_W = 4
MIN_BOX_H = 8


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return None
    return (vec / norm).astype(np.float32)


class ReIDManager:
    def __init__(
        self,
        model_path: str,
        similarity_threshold: float = 0.7,
        embedding_alpha: float = 0.2,
        device: str = "cpu",
        half: bool = False,
    ):
        from ultralytics.trackers.utils.reid import ReID

        self.model = ReID(model_path, device=device, fp16=half)
        self.similarity_threshold = similarity_threshold
        self.embedding_alpha = embedding_alpha

    def get_embedding(self, frame, x1, y1, x2, y2) -> Optional[np.ndarray]:
        """Return an embedding, or None for an invalid crop."""

        h_img, w_img = frame.shape[:2]

        # Clip the box to the frame.
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w_img, int(x2))
        y2 = min(h_img, int(y2))

        w = x2 - x1
        h = y2 - y1

        if w < MIN_BOX_W or h < MIN_BOX_H:
            return None

        # ReID expects the box center, width and height.
        detection = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, w, h]], dtype=np.float32)

        embeddings = self.model(frame, detection)
        if not embeddings or embeddings[0] is None:
            return None
        return _normalize(np.asarray(embeddings[0], dtype=np.float32).reshape(-1))

    @staticmethod
    def cosine_similarity(a, b) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0 or not np.isfinite(na * nb):
            return -1.0
        return float(np.dot(a, b) / (na * nb))

    def update_embedding(self, old, new):
        """Blend the old and current appearance."""
        updated = (1 - self.embedding_alpha) * old + self.embedding_alpha * new
        normalized = _normalize(updated)
        return old if normalized is None else normalized

    def find_matching_target(self, target_embedding, embeddings: dict):
        """Return the best ID above the similarity threshold."""

        best_similarity = -1.0
        best_id = None

        for track_id, embedding in embeddings.items():
            similarity = self.cosine_similarity(target_embedding, embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_id = track_id

        if best_similarity >= self.similarity_threshold:
            return best_id

        return None
