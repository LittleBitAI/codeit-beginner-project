"""Subprocess 실행과 종료를 담당하는 얇은 layer.

Test가 patch할 지점을 여기 한곳에 모읍니다(``tools/git_pr.py``의 ``capture``/
``execute``와 같은 방식). 그래서 이 module의 함수 이름은 test 계약의 일부입니다.

**여기서도, 다른 어디에서도 ``shell=True``를 쓰지 않습니다.** argv에는 사용자가 입력한
문자열이 하나도 들어가지 않습니다. ``sys.executable``, 고정 문자열, 그리고 서버가 만든
uuid 기반 경로뿐입니다.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any


__all__ = [
    "IS_WINDOWS",
    "build_argv",
    "child_environment",
    "kill_tree",
    "run_taskkill",
    "signal_group",
    "spawn",
    "terminate_tree",
]


IS_WINDOWS = os.name == "nt"

TERMINATE_GRACE_SECONDS = 10

# Windows에는 SIGKILL이 없습니다. 이름을 그때그때 참조하면 AttributeError가 나면서
# 예외 처리에 조용히 삼켜지므로, 여기서 한 번만 확정합니다.
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def build_argv(config_relative_path: str) -> list[str]:
    """학습을 실행할 argv를 만듭니다. 순수 함수입니다.

    이 저장소에서 학습을 시작하는 유일하게 허용된 방법입니다.
    """

    return [
        sys.executable,
        "-m",
        "src.main_pipeline",
        "--config",
        config_relative_path,
        "--only",
        "train",
    ]


def child_environment() -> dict[str, str]:
    """Child process가 UTF-8로 즉시 출력하도록 환경 변수를 준비합니다.

    한국어 Windows의 기본 console encoding은 cp949입니다. ``PYTHONIOENCODING``이 없으면
    train이 내보내는 한국어 오류 메시지가 깨져서 도착합니다. ``os.environ`` 자체는
    변경하지 않고 사본만 만듭니다.
    """

    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONUTF8"] = "1"
    return environment


def spawn(argv: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    """학습 process를 띄웁니다. ``Popen``을 부르는 유일한 지점입니다."""

    options: dict[str, Any] = {}
    if not IS_WINDOWS:
        # 자식이 process group leader가 되어야 worker까지 한 번에 종료할 수 있습니다.
        options["start_new_session"] = True

    return subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        close_fds=True,
        **options,
    )


def run_taskkill(pid: int) -> subprocess.CompletedProcess[str]:
    """Windows에서 process tree를 종료합니다. shell을 쓰지 않습니다."""

    return subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TERMINATE_GRACE_SECONDS,
    )


def signal_group(pid: int, signal_number: int) -> None:
    """POSIX에서 process group 전체에 신호를 보냅니다."""

    os.killpg(os.getpgid(pid), signal_number)


def terminate_tree(process: subprocess.Popen[str]) -> None:
    """자식과 그 자손까지 정중하게 종료를 요청합니다.

    ``Popen.terminate()``는 직계 자식만 종료하므로, DataLoader worker가 GPU 메모리를
    쥔 채 남습니다. 그래서 platform별로 tree 전체를 다룹니다.
    """

    if process.pid is None:
        return
    if IS_WINDOWS:
        try:
            run_taskkill(process.pid)
        except (OSError, subprocess.SubprocessError):
            process.terminate()
        return
    try:
        signal_group(process.pid, signal.SIGTERM)
    except (OSError, AttributeError):
        process.terminate()


def kill_tree(process: subprocess.Popen[str]) -> None:
    """정중한 종료가 통하지 않을 때 강제로 끝냅니다."""

    if process.pid is None:
        return
    if IS_WINDOWS:
        try:
            run_taskkill(process.pid)
        except (OSError, subprocess.SubprocessError):
            process.kill()
        return
    try:
        signal_group(process.pid, SIGKILL)
    except (OSError, AttributeError):
        process.kill()
