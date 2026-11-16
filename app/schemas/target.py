from pydantic import BaseModel


class TargetResponse(BaseModel):
    found: bool
    state: str
    target_id: int | None
    bbox: tuple[int, int, int, int] | None
    lost_frames: int
    fps: float
    inference_ms: float
    total_ms: float
