"""Web pipeline의 공개 experiment 조회 interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.common import ExperimentRegistryError, read_experiment_record


__all__ = ["run"]


def _result_error(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "artifacts": {},
        "summary": {"pipeline": "web", "mode": "experiment_registry"},
        "message": message,
    }


def _dummy_result() -> dict[str, Any]:
    return {
        "status": "ok",
        "artifacts": {},
        "summary": {"pipeline": "web", "mode": "dummy"},
        "message": "web pipeline dummy 실행 완료",
    }


def run(config: dict) -> dict:
    """설정된 exact URI의 experiment metadata를 공용 facade로 조회합니다."""

    if not isinstance(config, Mapping):
        return _result_error("config는 object여야 합니다.")

    web_config = config.get("web")
    if web_config is None:
        return _dummy_result()
    if not isinstance(web_config, Mapping):
        return _result_error("config['web']는 object여야 합니다.")

    if "experiment_record_uri" not in web_config:
        return _dummy_result()
    experiment_record_uri = web_config["experiment_record_uri"]

    try:
        record = read_experiment_record(
            experiment_record_uri,
            config,
            expected_run_id=web_config.get("expected_run_id"),
        )
    except ExperimentRegistryError as error:
        return _result_error(str(error))

    run_id = record["run_id"].strip()
    schema_version = record.get("schema_version")
    return {
        "status": "ok",
        "artifacts": {},
        "summary": {
            "pipeline": "web",
            "mode": "experiment_registry",
            "run_id": run_id,
            "schema_version": schema_version,
            "experiment_record_uri": experiment_record_uri.strip(),
        },
        "message": f"experiment record 조회 완료: {run_id}",
    }
