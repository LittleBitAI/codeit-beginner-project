"""GPU 상태 조회.

실제 GPU가 없어도, nvidia-smi가 없어도 통과해야 합니다.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from src.pipelines.web import gpu


SAMPLE_CSV = (
    "0, NVIDIA GeForce RTX 4090, 78, 18227, 24564, 61\n"
    "1, NVIDIA A100-SXM4-40GB, 41, 16384, 40960, 55\n"
)


def completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(["nvidia-smi"], returncode, stdout, stderr)


@pytest.fixture
def smi_available(monkeypatch):
    monkeypatch.setattr(gpu, "_resolve_nvidia_smi", lambda: "/usr/bin/nvidia-smi")


# --- 정상 조회 --------------------------------------------------------------


def test_parses_nvidia_smi_rows(monkeypatch, smi_available):
    monkeypatch.setattr(gpu, "run_nvidia_smi", lambda executable: completed(SAMPLE_CSV))

    telemetry = gpu.probe()["telemetry"]

    assert telemetry["source"] == "nvidia-smi"
    assert len(telemetry["devices"]) == 2
    first = telemetry["devices"][0]
    assert first["name"] == "NVIDIA GeForce RTX 4090"
    assert first["utilization_percent"] == 78
    assert first["memory_used_mb"] == 18227
    assert first["memory_total_mb"] == 24564
    assert first["temperature_c"] == 61


def test_not_available_fields_become_none(monkeypatch, smi_available):
    monkeypatch.setattr(
        gpu, "run_nvidia_smi", lambda executable: completed("0, GPU, [N/A], 100, 200, [N/A]\n")
    )

    device = gpu.probe()["telemetry"]["devices"][0]

    assert device["utilization_percent"] is None
    assert device["temperature_c"] is None
    assert device["memory_used_mb"] == 100


def test_malformed_rows_are_skipped(monkeypatch, smi_available):
    monkeypatch.setattr(
        gpu, "run_nvidia_smi", lambda executable: completed("깨진 줄\n0, GPU, 1, 2, 3, 4\n")
    )

    assert len(gpu.probe()["telemetry"]["devices"]) == 1


def test_nvidia_smi_argv_is_exact_and_shellless(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return completed(SAMPLE_CSV)

    monkeypatch.setattr(subprocess, "run", fake_run)

    gpu.run_nvidia_smi("/usr/bin/nvidia-smi")

    assert captured["args"] == [
        "/usr/bin/nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    assert "shell" not in captured["kwargs"]
    assert captured["kwargs"]["timeout"] == gpu.NVIDIA_SMI_TIMEOUT_SECONDS


# --- 조회 실패 --------------------------------------------------------------


def test_nvidia_smi_absent_reports_unavailable(monkeypatch):
    monkeypatch.setattr(gpu, "_resolve_nvidia_smi", lambda: None)

    telemetry = gpu.probe()["telemetry"]

    assert telemetry["source"] == "unavailable"
    assert telemetry["reason"] == "nvidia_smi_not_found"
    assert telemetry["devices"] == []
    # 0으로 채운 게이지 대신 이유를 설명해야 합니다.
    assert "nvidia-smi" in telemetry["message"]


def test_nvidia_smi_timeout_reports_unavailable(monkeypatch, smi_available):
    def timeout(executable):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr(gpu, "run_nvidia_smi", timeout)

    assert gpu.probe()["telemetry"]["reason"] == "timeout"


def test_nvidia_smi_file_not_found_reports_unavailable(monkeypatch, smi_available):
    def missing(executable):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu, "run_nvidia_smi", missing)

    assert gpu.probe()["telemetry"]["reason"] == "nvidia_smi_not_found"


def test_nonzero_exit_does_not_leak_paths(monkeypatch, smi_available):
    from src.pipelines.web.paths import REPOSITORY_ROOT

    monkeypatch.setattr(
        gpu,
        "run_nvidia_smi",
        lambda executable: completed(stderr=f"열 수 없음 {REPOSITORY_ROOT}/x", returncode=9),
    )

    telemetry = gpu.probe()["telemetry"]

    assert telemetry["reason"] == "nvidia_smi_failed"
    assert str(REPOSITORY_ROOT) not in telemetry["message"]


def test_nvidia_smi_inside_repository_is_refused(monkeypatch, tmp_path):
    """cwd가 저장소 root라 저장소 안의 nvidia-smi.exe가 먼저 잡힐 수 있습니다."""

    from src.pipelines.web import paths

    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(paths, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        gpu.shutil, "which", lambda name: str(repository / "nvidia-smi.exe")
    )

    assert gpu._resolve_nvidia_smi() is None
    assert gpu.probe()["telemetry"]["reason"] == "nvidia_smi_not_found"


def test_nvidia_smi_outside_repository_is_accepted(monkeypatch, tmp_path):
    """저장소와 무관한 위치의 실행 파일이어야 씁니다.

    pytest --basetemp이 저장소 안을 가리켜도 결과가 달라지지 않도록, 저장소 root와
    실행 파일 위치를 test 안에서 명시적으로 갈라 둡니다.
    """

    from src.pipelines.web import paths

    repository = tmp_path / "repo"
    repository.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    executable = elsewhere / "nvidia-smi"
    executable.write_text("", encoding="utf-8")

    monkeypatch.setattr(paths, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(gpu.shutil, "which", lambda name: str(executable))

    assert gpu._resolve_nvidia_smi() == str(executable)


# --- torch 쪽 --------------------------------------------------------------


def test_gpu_probe_does_not_initialize_cuda_context(monkeypatch):
    """get_device_properties는 CUDA context를 만들어 VRAM을 수백 MB 잡아먹습니다.

    그만큼 학습 child process가 쓰지 못하므로 절대 부르면 안 됩니다.
    """

    from unittest.mock import Mock

    properties = Mock()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_properties=properties,
            init=Mock(),
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(gpu, "_resolve_nvidia_smi", lambda: None)

    result = gpu.probe()

    properties.assert_not_called()
    fake_torch.cuda.init.assert_not_called()
    assert result["torch"] == {"cuda_available": True, "device_count": 2, "reason": None}


def test_torch_probe_failure_degrades(monkeypatch):
    broken = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("깨짐")))
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", broken)
    monkeypatch.setattr(gpu, "_resolve_nvidia_smi", lambda: None)

    torch_status = gpu.probe()["torch"]

    assert torch_status["cuda_available"] is False
    assert torch_status["reason"].startswith("torch_probe_failed")


def test_cuda_is_available_never_raises(monkeypatch):
    broken = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("깨짐")))
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", broken)

    assert gpu.cuda_is_available() is False


def test_probe_result_is_json_serializable(monkeypatch, smi_available):
    import json

    monkeypatch.setattr(gpu, "run_nvidia_smi", lambda executable: completed(SAMPLE_CSV))

    json.dumps(gpu.probe(), allow_nan=False, ensure_ascii=False)
