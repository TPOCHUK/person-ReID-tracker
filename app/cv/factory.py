"""Build the same pipeline for the server and benchmark."""

from pathlib import Path

import cv2

from app.config.settings import Settings
from app.cv.pipeline import CVPipeline, PipelineConfig
from app.cv.reid import ReIDManager
from app.cv.tracker import TargetTracker, TrackerConfig


def build_pipeline(s: Settings) -> CVPipeline:
    # Load heavy dependencies only when building the pipeline.
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(s.torch_threads)
    cv2.setNumThreads(s.cv_threads)

    # Use FP16 only on CUDA.
    half = s.half and s.device.startswith("cuda")

    for model_path in (s.yolo_model, s.reid_model):
        if not Path(model_path).is_file():
            raise RuntimeError(f"Model file not found: {model_path}")

    detector = YOLO(s.yolo_model)
    reid = ReIDManager(
        s.reid_model,
        similarity_threshold=s.similarity_threshold,
        embedding_alpha=s.reid_embedding_alpha,
        device=s.device,
        half=half,
    )

    tracker = TargetTracker(
        reid,
        TrackerConfig(
            conf_threshold=s.conf_threshold,
            reid_update_interval=s.reid_update_interval,
            reid_update_conf=s.reid_update_conf,
            reid_recovery_start=s.reid_recovery_start,
            reid_recovery_interval=s.reid_recovery_interval,
            max_lost_frames=s.max_lost_frames,
            recovery_max_candidates=s.recovery_max_candidates,
        ),
    )

    return CVPipeline(
        detector,
        tracker,
        PipelineConfig(
            imgsz=s.imgsz,
            detector_conf=s.detector_conf,
            device=s.device,
            half=half,
            tracker_yaml=s.yolo_tracker,
            max_lost_frames=s.max_lost_frames,
        ),
    )
