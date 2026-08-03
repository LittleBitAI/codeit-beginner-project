import json
from types import SimpleNamespace

import pytest

from src import main_pipeline


def test_full_dummy_pipeline_completes():
    result = main_pipeline.run({"execution": {"mode": "dummy"}})

    assert result["status"] == "ok"
    assert list(result["summary"]) == [
        "data",
        "train",
        "evaluate",
        "registry",
        "web",
    ]
    assert set(result) == {"status", "artifacts", "summary", "message"}


def test_only_executes_one_pipeline(capsys):
    exit_code = main_pipeline.main(["--only", "train"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert list(output["summary"]) == ["train"]


def test_pipeline_stops_on_failure(monkeypatch):
    calls = []

    def succeed(config):
        calls.append("data")
        return {
            "status": "ok",
            "artifacts": {},
            "summary": {},
            "message": "",
        }

    def fail(config):
        calls.append("train")
        return {
            "status": "error",
            "artifacts": {},
            "summary": {},
            "message": "의도한 실패",
        }

    def must_not_run(config):
        calls.append("evaluate")
        raise AssertionError("실패 이후 pipeline이 실행되었습니다.")

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (
            ("data", SimpleNamespace(run=succeed)),
            ("train", SimpleNamespace(run=fail)),
            ("evaluate", SimpleNamespace(run=must_not_run)),
        ),
    )

    result = main_pipeline.run({})

    assert result["status"] == "error"
    assert result["message"] == "train: 의도한 실패"
    assert calls == ["data", "train"]


def test_pipeline_rejects_invalid_return_schema(monkeypatch):
    def invalid(config):
        return {
            "status": "ok",
            "artifacts": {},
            "summary": {},
        }

    monkeypatch.setattr(
        main_pipeline,
        "_STAGES",
        (("data", SimpleNamespace(run=invalid)),),
    )

    with pytest.raises(ValueError, match="공통 계약과 다릅니다"):
        main_pipeline.run({})
