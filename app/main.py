import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import router
from app.camera.camera import Camera
from app.config.settings import settings
from app.cv.factory import build_pipeline
from app.service.system import SystemService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = build_pipeline(settings)

    app.state.pipeline = pipeline
    app.state.system_service = SystemService(
        pipeline,
        camera_factory=lambda: Camera(
            settings.camera_source,
            settings.camera_width,
            settings.camera_height,
        ),
    )

    try:
        yield
    finally:
        await app.state.system_service.stop()


app = FastAPI(title="Person ReID Tracker", lifespan=lifespan)

app.include_router(router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
