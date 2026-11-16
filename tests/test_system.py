import asyncio
import time
from unittest.mock import Mock

from app.service.system import SystemService


class Camera:
    def __init__(self, fail=False):
        self.released = False
        self.fail = fail

    def read(self):
        time.sleep(0.005)
        if self.fail:
            raise RuntimeError("read failed")
        return True, object()

    def release(self):
        self.released = True


def test_concurrent_start_and_stop_release_one_camera():
    async def scenario():
        camera = Camera()
        factory = Mock(return_value=camera)
        pipeline = Mock()
        service = SystemService(pipeline, factory)
        await asyncio.gather(service.start(), service.start())
        assert service.running
        factory.assert_called_once()
        await asyncio.sleep(0.02)
        await asyncio.gather(service.stop(), service.stop())
        assert camera.released and not service.running
        assert pipeline.reset.call_count == 2
        assert pipeline.process.call_count > 0
        await service.start()
        assert factory.call_count == 2
        await service.stop()

    asyncio.run(scenario())


def test_capture_error_releases_camera_and_allows_restart():
    async def scenario():
        cameras = [Camera(fail=True), Camera()]
        service = SystemService(Mock(), Mock(side_effect=cameras))
        await service.start()
        await asyncio.sleep(0.05)
        assert not service.running and cameras[0].released
        await service.stop()
        await service.start()
        assert service.running
        await service.stop()
        assert cameras[1].released

    asyncio.run(scenario())


def test_slow_camera_open_does_not_block_requests():
    async def scenario():
        camera = Camera()
        started = asyncio.Event()

        def factory():
            time.sleep(0.1)
            return camera

        service = SystemService(Mock(), factory)
        task = asyncio.create_task(service.start())

        async def heartbeat():
            await asyncio.sleep(0.01)
            started.set()

        await heartbeat()
        assert started.is_set() and not task.done()
        await task
        await service.stop()

    asyncio.run(scenario())


def test_end_of_file_releases_camera_and_clears_state():
    async def scenario():
        camera = Mock()
        camera.read.return_value = (False, None)
        pipeline = Mock()
        service = SystemService(pipeline, lambda: camera)
        await service.start()
        await asyncio.sleep(0.05)
        assert not service.running
        camera.release.assert_called_once()
        pipeline.process.assert_not_called()
        assert pipeline.reset.call_count == 2
        await service.stop()

    asyncio.run(scenario())


def test_open_error_leaves_service_stopped():
    async def scenario():
        factory = Mock(side_effect=RuntimeError("unavailable"))
        service = SystemService(Mock(), factory)
        try:
            await service.start()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Expected an open error")
        assert not service.running
        await service.stop()

    asyncio.run(scenario())
