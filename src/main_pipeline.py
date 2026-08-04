"""Data부터 registry까지 공개 interface만 연결하는 통합 진입점."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

from src.common import load_config, validate_pipeline_result
from src.pipelines import data, evaluate, registry, train


# Pipeline 연결 순서는 이 파일에서만 관리합니다. Web은 registry 결과를 소비하는
# 별도 사용자 interface이므로 실행 stage에 포함하지 않습니다.
_STAGES = (
    ("data", data),
    ("train", train),
    ("evaluate", evaluate),
    ("registry", registry),
)

_REQUIRED_ARTIFACTS = {
    "data": (
        "train_manifest_uri",
        "validation_manifest_uri",
        "class_map_uri",
        "dataset_summary_uri",
    ),
    "train": (
        "run_id",
        "best_checkpoint_uri",
        "last_checkpoint_uri",
        "training_history_uri",
    ),
    "evaluate": ("run_id", "metrics_uri", "predictions_uri"),
    "registry": ("run_id", "experiment_record_uri"),
}


def _error_result(
    stage: str,
    cause: str,
    *,
    artifacts: Mapping[str, dict[str, Any]] | None = None,
    summaries: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """실패 stage와 원인을 공통 반환 형식으로 만듭니다."""

    return {
        "status": "error",
        "artifacts": dict(artifacts or {}),
        "summary": dict(summaries or {}),
        "message": f"{stage}: {cause}",
    }


def _configured_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """단독 stage 실행에 사용할 명시적 artifact 입력을 복사합니다."""

    inputs = config.get("inputs", {})
    if not isinstance(inputs, Mapping):
        raise ValueError("config['inputs']는 object여야 합니다.")
    return _copy_inputs(inputs)


def _copy_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Pipeline이 바깥쪽 artifact mapping을 바꾸지 못하도록 복사합니다."""

    return {
        name: dict(value) if isinstance(value, Mapping) else value
        for name, value in inputs.items()
    }


def _is_dummy_result(result: Mapping[str, Any]) -> bool:
    summary = result.get("summary")
    return isinstance(summary, Mapping) and summary.get("mode") == "dummy"


def _validate_required_artifacts(stage: str, result: Mapping[str, Any]) -> None:
    """성공한 stage가 다음 단계에 필요한 문자열 artifact를 냈는지 확인합니다."""

    if _is_dummy_result(result):
        return

    required = _REQUIRED_ARTIFACTS.get(stage, ())
    artifacts = result["artifacts"]
    missing = [key for key in required if key not in artifacts]
    invalid = [
        key
        for key in required
        if key in artifacts
        and (not isinstance(artifacts[key], str) or not artifacts[key].strip())
    ]
    problems = []
    if missing:
        problems.append(f"필수 artifact 누락: {', '.join(missing)}")
    if invalid:
        problems.append(
            "비어 있지 않은 문자열이어야 하는 artifact: " + ", ".join(invalid)
        )
    if problems:
        raise ValueError("; ".join(problems))


def _dry_run_result(only: str | None) -> dict[str, Any]:
    stages = [name for name, _ in _STAGES if only is None or name == only]
    return {
        "status": "ok",
        "artifacts": {},
        "summary": {"dry_run": True, "stages": stages},
        "message": "pipeline을 실행하지 않고 실행 계획만 확인했습니다.",
    }


def run(
    config: dict[str, Any] | None = None,
    only: str | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """전체 pipeline 또는 선택한 pipeline 하나를 실행합니다.

    전체 실행에서는 data에 명시적으로 설정된 시작 artifact만 주고, 이후
    stage에는 이 실행에서 앞선 stage가 반환한 artifact만 누적해 전달합니다.
    ``only`` 실행은 필요한 선행 artifact를 config의 ``inputs``에서 받습니다.
    """

    if config is None:
        resolved_config: dict[str, Any] = {}
    elif isinstance(config, dict):
        resolved_config = dict(config)
    else:
        return _error_result("config", "최상위 값은 object여야 합니다.")

    stage_names = {name for name, _ in _STAGES}
    if only is not None and only not in stage_names:
        raise ValueError(f"지원하지 않는 pipeline입니다: {only}")

    try:
        configured_inputs = _configured_inputs(resolved_config)
    except ValueError as error:
        return _error_result("config", str(error))
    if dry_run:
        return _dry_run_result(only)

    artifacts: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for name, pipeline in _STAGES:
        if only is not None and name != only:
            continue

        if only is not None:
            stage_inputs = _copy_inputs(configured_inputs)
        elif name == "data":
            stage_inputs = {}
            if "data" in configured_inputs:
                data_input = configured_inputs["data"]
                stage_inputs["data"] = (
                    dict(data_input) if isinstance(data_input, Mapping) else data_input
                )
        else:
            stage_inputs = _copy_inputs(artifacts)

        stage_config = dict(resolved_config)
        stage_config["inputs"] = stage_inputs

        try:
            result = pipeline.run(stage_config)
            validate_pipeline_result(result, pipeline_name=name)
        except Exception as error:
            cause = f"{type(error).__name__}: {error}"
            return _error_result(
                name, cause, artifacts=artifacts, summaries=summaries
            )

        if result["status"] != "ok":
            artifacts[name] = dict(result["artifacts"])
            summaries[name] = dict(result["summary"])
            return _error_result(
                name,
                result["message"],
                artifacts=artifacts,
                summaries=summaries,
            )

        try:
            _validate_required_artifacts(name, result)
        except ValueError as error:
            summaries[name] = dict(result["summary"])
            return _error_result(
                name, str(error), artifacts=artifacts, summaries=summaries
            )

        artifacts[name] = dict(result["artifacts"])
        summaries[name] = dict(result["summary"])

    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": summaries,
        "message": "pipeline 실행을 완료했습니다.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pill object-detection pipeline")
    parser.add_argument("--config", help="읽을 JSON config 경로")
    parser.add_argument("--only", choices=[name for name, _ in _STAGES])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pipeline을 실행하지 않고 실행 순서만 출력",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        result = _error_result("config", f"{type(error).__name__}: {error}")
    else:
        try:
            result = run(config, only=args.only, dry_run=args.dry_run)
        except Exception as error:
            result = _error_result("main", f"{type(error).__name__}: {error}")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
