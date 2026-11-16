from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from app.dependencies import get_pipeline, get_system_service
from app.main import app


def test_api_status_target_and_camera_error():
    snapshot = dict(
        found=False,
        state="idle",
        target_id=None,
        bbox=None,
        lost_frames=0,
        fps=0.0,
        inference_ms=0.0,
        total_ms=0.0,
    )
    pipeline = Mock(snapshot=snapshot)
    service = Mock(running=False)
    service.start = AsyncMock(side_effect=RuntimeError("Camera unavailable"))
    service.stop = AsyncMock()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_system_service] = lambda: service
    try:
        # No context manager: real model lifespan is tested separately.
        client = TestClient(app)
        assert client.get("/").status_code == 200
        assert client.get("/static/script.js").status_code == 200
        assert client.get("/status").json() == {"running": False}
        assert client.get("/target").json() == snapshot
        assert client.post("/system/start").status_code == 503
        service.start.side_effect = None
        assert client.post("/system/start").json() == {"status": "started"}
        assert client.post("/system/stop").json() == {"status": "stopped"}
    finally:
        app.dependency_overrides.clear()
