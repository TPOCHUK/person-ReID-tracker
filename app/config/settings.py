"""Settings from environment variables and the local .env file."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Target tracking.
    conf_threshold: float = Field(0.6, ge=0, le=1)
    similarity_threshold: float = Field(0.7, ge=-1, le=1)
    max_lost_frames: int = Field(30, gt=0)
    reid_update_interval: int = Field(10, gt=0)
    reid_update_conf: float = Field(0.8, ge=0, le=1)
    reid_recovery_interval: int = Field(5, gt=0)
    reid_recovery_start: int = Field(5, ge=0)
    reid_embedding_alpha: float = Field(0.2, ge=0, le=1)
    recovery_max_candidates: int = Field(8, gt=0)

    # Models and CPU threads.
    yolo_model: str = str(ROOT_DIR / "yolov8n.pt")
    reid_model: str = str(ROOT_DIR / "yolo26n-reid.onnx")
    yolo_tracker: str = "bytetrack.yaml"
    detector_conf: float = Field(0.25, ge=0, le=1)
    imgsz: int = Field(256, gt=0)
    device: str = "cpu"
    half: bool = False
    torch_threads: int = Field(2, gt=0)
    cv_threads: int = Field(1, ge=0)

    # Camera and browser stream.
    camera_source: str = "0"
    camera_width: int = Field(640, gt=0)
    camera_height: int = Field(480, gt=0)
    jpeg_quality: int = Field(70, ge=1, le=100)
    stream_max_fps: float = Field(15.0, gt=0)
    benchmark_output_dir: str = str(ROOT_DIR / "benchmark_results")

    @field_validator("yolo_model", "reid_model", "benchmark_output_dir")
    @classmethod
    def resolve_path(cls, value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else ROOT_DIR / path)

    @field_validator("camera_source")
    @classmethod
    def resolve_source(cls, value):
        if value.isdigit() or "://" in value:
            return value
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else ROOT_DIR / path)

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
