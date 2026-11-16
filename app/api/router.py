import asyncio

import cv2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.config.settings import settings
from app.cv.pipeline import CVPipeline
from app.dependencies import get_pipeline, get_system_service
from app.schemas.system import SystemStatus
from app.schemas.target import TargetResponse
from app.service.system import SystemService

router = APIRouter()


@router.get("/target", response_model=TargetResponse)
def get_target(pipeline: CVPipeline = Depends(get_pipeline)):
    return pipeline.snapshot


@router.post("/system/start")
async def start(service: SystemService = Depends(get_system_service)):
    try:
        await service.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {"status": "started"}


@router.post("/system/stop")
async def stop_system(service: SystemService = Depends(get_system_service)):
    await service.stop()
    return {"status": "stopped"}


@router.get("/status", response_model=SystemStatus)
def status(service: SystemService = Depends(get_system_service)):
    return {"running": service.running}


async def generate_frames(pipeline: CVPipeline):
    """Stream new frames with a configurable FPS limit."""

    min_interval = 1.0 / settings.stream_max_fps
    last_seq = -1

    while True:
        seq = pipeline.frame_seq
        frame = pipeline.latest_frame

        if frame is None or seq == last_seq:
            await asyncio.sleep(0.02)
            continue

        last_seq = seq

        # Keep JPEG encoding off the event loop.
        ok, buffer = await asyncio.to_thread(
            cv2.imencode,
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality],
        )
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        await asyncio.sleep(min_interval)


@router.get("/video/stream")
async def video_stream(pipeline: CVPipeline = Depends(get_pipeline)):
    return StreamingResponse(
        generate_frames(pipeline),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
