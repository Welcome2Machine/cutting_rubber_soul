"""Report whether the project-local research environment is ready.

This checker is intentionally local and does not download packages, weights, or data.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import sys
from typing import Final


REQUIRED_MODULES: Final = {
    name: (module, name) for name, module in (
        ('numpy', 'numpy'), ('scipy', 'scipy'), ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'), ('opencv-python', 'cv2'),
        ('psutil', 'psutil'), ('torch', 'torch'), ('pytest', 'pytest'))
}


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 11),
        "required_packages": {
            name: {
                "available": module_available(module),
                "version": package_version(distribution),
            }
            for name, (module, distribution) in REQUIRED_MODULES.items()
        },
        "video_tools": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
        },
    }

    if module_available("torch"):
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    required_ok = bool(report["python_ok"]) and all(
        item["available"]
        for item in report["required_packages"].values()  # type: ignore[index]
    )
    tools_ok = all(report["video_tools"].values())
    return 0 if required_ok and tools_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
