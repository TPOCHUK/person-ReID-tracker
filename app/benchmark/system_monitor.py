import os
import threading
import time

import psutil


def _init_gpu():
    """Повертає (pynvml, handle) або None, якщо NVIDIA GPU недоступний."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


class SystemMonitor:
    """Sample process and system resources in a background thread."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.cpu_count = psutil.cpu_count(logical=True) or 1

        self._gpu = _init_gpu()
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0

    def start(self):
        # Prime CPU counters before collecting samples.
        self.process.cpu_percent(None)
        psutil.cpu_percent(None)

        self._samples = []
        self._stop.clear()
        self._t0 = time.perf_counter()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._samples.append(self._sample())
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def _sample(self) -> dict:
        cpu_proc = self.process.cpu_percent(None)

        sample = {
            "t": round(time.perf_counter() - self._t0, 3),
            "cpu_process_percent": cpu_proc,
            "cpu_process_norm_percent": round(cpu_proc / self.cpu_count, 2),
            "cpu_system_percent": psutil.cpu_percent(None),
            "ram_process_mb": round(self.process.memory_info().rss / 1024**2, 2),
            "ram_system_percent": psutil.virtual_memory().percent,
            "gpu_percent": None,
            "vram_mb": None,
        }

        if self._gpu is not None:
            try:
                pynvml, handle = self._gpu
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                sample["gpu_percent"] = util.gpu
                sample["vram_mb"] = round(mem.used / 1024**2, 2)
            except Exception:
                pass

        return sample

    def stop(self) -> list:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2)
        return list(self._samples)
