"""로컬 CUDA/GPU 상태 조회.

서버 process에서 CUDA context를 만들지 않는 것이 이 module의 핵심 제약입니다.
``torch.cuda.get_device_properties()``는 CUDA를 초기화하면서 수백 MB의 VRAM을
영구히 잡아, 정작 학습 child process가 그만큼 쓰지 못하게 됩니다. 그래서 torch는
context를 만들지 않는 ``is_available()``/``device_count()``만 쓰고, 이름·메모리·
사용률·온도는 별도 process인 ``nvidia-smi``에서 가져옵니다.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

from .masking import sanitize_line
from .paths import REPOSITORY_ROOT


__all__ = ["cuda_is_available", "probe", "run_nvidia_smi"]


NVIDIA_SMI_QUERY = (
    "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
)
NVIDIA_SMI_TIMEOUT_SECONDS = 5
_NOT_AVAILABLE = {"[n/a]", "n/a", "[not supported]", "not supported", ""}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cuda_is_available() -> bool:
    """CUDA를 쓸 수 있는지만 확인합니다. CUDA context를 만들지 않습니다."""

    try:
        import torch
    except Exception:  # torch 미설치/로딩 실패가 GUI를 멈추게 하지 않습니다.
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"cuda_available": False, "device_count": 0, "reason": "torch_unavailable"}
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        return {"cuda_available": available, "device_count": count, "reason": None}
    except Exception as error:
        return {
            "cuda_available": False,
            "device_count": 0,
            "reason": f"torch_probe_failed:{type(error).__name__}",
        }


def _resolve_nvidia_smi() -> str | None:
    """PATH에서 nvidia-smi를 찾되 저장소 안의 실행 파일은 거부합니다.

    Windows의 ``CreateProcess``는 PATH보다 현재 directory를 먼저 찾습니다. 학습을
    저장소 root를 cwd로 실행하므로, 저장소에 놓인 ``nvidia-smi.exe``가 대신 실행될
    수 있습니다. 그런 경로는 쓰지 않습니다.
    """

    resolved = shutil.which("nvidia-smi")
    if resolved is None:
        return None
    try:
        from pathlib import Path

        Path(resolved).resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved  # 저장소 밖 -> 정상
    except OSError:
        return None
    return None  # 저장소 안 -> 거부


def run_nvidia_smi(executable: str) -> subprocess.CompletedProcess[str]:
    """nvidia-smi를 argv로 실행합니다. shell을 쓰지 않습니다."""

    return subprocess.run(
        [
            executable,
            f"--query-gpu={NVIDIA_SMI_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
    )


def _parse_number(value: str) -> int | None:
    text = value.strip()
    if text.lower() in _NOT_AVAILABLE:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_rows(stdout: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, utilization, memory_used, memory_total, temperature = fields
        devices.append(
            {
                "index": _parse_number(index),
                "name": name or None,
                "utilization_percent": _parse_number(utilization),
                "memory_used_mb": _parse_number(memory_used),
                "memory_total_mb": _parse_number(memory_total),
                "temperature_c": _parse_number(temperature),
            }
        )
    return devices


def _telemetry() -> dict[str, Any]:
    executable = _resolve_nvidia_smi()
    if executable is None:
        return {
            "source": "unavailable",
            "reason": "nvidia_smi_not_found",
            "message": "nvidia-smi를 찾지 못해 GPU 사용률 정보를 가져올 수 없습니다.",
            "devices": [],
        }

    try:
        completed = run_nvidia_smi(executable)
    except FileNotFoundError:
        return {
            "source": "unavailable",
            "reason": "nvidia_smi_not_found",
            "message": "nvidia-smi를 찾지 못해 GPU 사용률 정보를 가져올 수 없습니다.",
            "devices": [],
        }
    except subprocess.TimeoutExpired:
        return {
            "source": "unavailable",
            "reason": "timeout",
            "message": "nvidia-smi 응답이 없어 GPU 사용률 정보를 가져오지 못했습니다.",
            "devices": [],
        }
    except OSError as error:
        return {
            "source": "unavailable",
            "reason": f"os_error:{type(error).__name__}",
            "message": "nvidia-smi를 실행하지 못해 GPU 사용률 정보를 가져올 수 없습니다.",
            "devices": [],
        }

    if completed.returncode != 0:
        detail = sanitize_line((completed.stderr or "").strip()) or "알 수 없는 오류"
        return {
            "source": "unavailable",
            "reason": "nvidia_smi_failed",
            "message": f"nvidia-smi가 오류를 반환했습니다: {detail}",
            "devices": [],
        }

    return {
        "source": "nvidia-smi",
        "reason": None,
        "message": None,
        "devices": _parse_rows(completed.stdout or ""),
    }


def probe() -> dict[str, Any]:
    """GPU 상태를 조회합니다. 실패해도 예외를 던지지 않습니다."""

    return {
        "torch": _torch_status(),
        "telemetry": _telemetry(),
        "queried_at": _now(),
    }
