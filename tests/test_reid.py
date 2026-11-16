from unittest.mock import Mock

import numpy as np
import pytest

from app.cv.reid import ReIDManager, _normalize


def manager(output=None, alpha=0.2):
    reid = ReIDManager.__new__(ReIDManager)
    reid.model = Mock(return_value=[output])
    reid.embedding_alpha = alpha
    reid.similarity_threshold = 0.7
    return reid


def test_reid_receives_center_box_after_clipping():
    reid = manager(np.array([3, 4]))
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    embedding = reid.get_embedding(frame, -10, 20, 80, 150)
    np.testing.assert_allclose(reid.model.call_args.args[1], [[40, 60, 80, 80]])
    np.testing.assert_allclose(embedding, [0.6, 0.8])


@pytest.mark.parametrize("output", [None, np.zeros(4), np.array([np.nan, 1])])
def test_invalid_embedding_is_ignored(output):
    reid = manager(output)
    assert reid.get_embedding(np.zeros((100, 100, 3)), 0, 0, 20, 40) is None


def test_small_box_does_not_call_model():
    reid = manager()
    assert reid.get_embedding(np.zeros((100, 100, 3)), 0, 0, 2, 4) is None
    reid.model.assert_not_called()


def test_embedding_uses_configured_alpha():
    reid = manager(alpha=1.0)
    np.testing.assert_allclose(
        reid.update_embedding(np.array([1.0, 0.0]), np.array([0.0, 1.0])), [0, 1]
    )


def test_nonfinite_similarity_is_rejected():
    assert ReIDManager.cosine_similarity(np.array([np.nan]), np.ones(1)) == -1
    assert _normalize(np.zeros(3)) is None
