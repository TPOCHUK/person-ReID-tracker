import asyncio
import logging
from typing import Callable

log = logging.getLogger(__name__)


class SystemService:
    """Start and stop one capture task. Камера належить цій задачі."""

    def __init__(self, pipeline, camera_factory: Callable):
        self.pipeline = pipeline
        self._camera_factory = camera_factory
        self._task = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self):
        async with self._lock:
            if self.running:
                return
            camera = await asyncio.to_thread(self._camera_factory)
            try:
                self.pipeline.reset()
            except Exception:
                await asyncio.to_thread(camera.release)
                raise
            self._stop_event.clear()
            self._task = asyncio.create_task(self._run(camera))

    async def _run(self, camera):
        try:
            while not self._stop_event.is_set():
                ok, frame = await asyncio.to_thread(camera.read)
                if not ok or frame is None:
                    log.info("Source ended, stopping capture")
                    break
                await asyncio.to_thread(self.pipeline.process, frame)
        except Exception:
            log.exception("Capture failed")
        finally:
            try:
                await asyncio.to_thread(camera.release)
            finally:
                self.pipeline.reset()

    async def stop(self):
        async with self._lock:
            if self._task is None:
                return
            self._stop_event.set()
            # Wait for the current read/inference before releasing the camera.
            await asyncio.shield(self._task)
            self._task = None
