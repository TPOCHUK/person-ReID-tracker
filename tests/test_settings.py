import pytest
from pydantic import ValidationError

from app.config.settings import ROOT_DIR, Settings


@pytest.mark.parametrize(
    "field,value",
    [
        ("stream_max_fps", 0),
        ("torch_threads", 0),
        ("max_lost_frames", -1),
        ("reid_update_interval", 0),
        ("jpeg_quality", 101),
        ("reid_embedding_alpha", 2),
    ],
)
def test_invalid_settings_fail_early(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_relative_model_paths_use_project_root():
    settings = Settings(
        _env_file=None, yolo_model="models/detector.pt", camera_source="videos/test.mp4"
    )
    assert settings.yolo_model == str(ROOT_DIR / "models/detector.pt")
    assert settings.camera_source == str(ROOT_DIR / "videos/test.mp4")
