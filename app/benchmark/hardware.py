"""Record hardware and library versions."""

import os
import platform

import cv2
import psutil


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpu_name():
    try:
        import pynvml

        pynvml.nvmlInit()
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return name.decode() if isinstance(name, bytes) else name
    except Exception:
        return None


def _version(module_name: str):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def collect_hardware_info() -> dict:
    torch_version = _version("torch")
    cuda = None
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        pass

    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_mb": round(psutil.virtual_memory().total / 1024**2),
        "gpu": _gpu_name(),
        "torch": torch_version,
        "torch_cuda_available": cuda,
        "ultralytics": _version("ultralytics"),
        "opencv": cv2.__version__,
    }
