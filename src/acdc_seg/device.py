"""CUDA-first device selection with a documented CPU fallback."""

from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return ``cuda`` when a GPU is visible, otherwise ``cpu``.

    The Pixi environment always installs *CUDA-enabled* PyTorch. That is
    different from having a GPU at runtime: without an NVIDIA driver / GPU,
    ``torch.cuda.is_available()`` is False and we train on CPU so the demo
    still runs.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> dict[str, object]:
    """Runtime diagnostics for the README / ``pixi run check-cuda`` task."""
    device = device or get_device()
    info: dict[str, object] = {
        "torch": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cudnn": bool(torch.backends.cudnn.is_available()) if torch.cuda.is_available() else False,
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(idx)
        info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return info
