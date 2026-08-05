"""Registry pipeline의 실제 S3 왕복 smoke test 명령입니다.

작은 experiment record 하나를 실제 S3에 등록하고 다시 조회해서, registry
pipeline이 s3 backend에서도 동작하는지 확인합니다. 실행 방법은 다음과 같습니다.

    python -m src.pipelines.registry.smoke_s3 --config configs/env.aws.json

AWS credential은 받지도, 저장하지도 않고 boto3의 default credential chain에
맡깁니다. `scripts/s3_smoke_test.py`와 같은 정책으로 생성한 object를 자동으로
삭제하지 않으며, 남은 object URI를 결과에 알려 줍니다.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import uuid
from collections.abc import Sequence

from src.common import S3Storage, StorageError, create_storage, load_config

from . import run as run_registry
from .record import RegistryError


__all__ = ["SMOKE_INPUTS", "SMOKE_PREFIX", "main", "run_smoke_test"]

# 실제 dataset이나 실험 artifact와 섞이지 않도록 전용 prefix를 씁니다.
SMOKE_PREFIX = "registry/smoke-tests/"

# 계약 형식을 그대로 따르는 최소 입력입니다. 모두 `s3://` 참조이므로 registry는
# 내용을 읽지 않고 참조만 기록하며, 따라서 실제로 존재하지 않아도 됩니다.
SMOKE_INPUTS: dict[str, dict[str, str]] = {
    "data": {
        "train_manifest_uri": "s3://example-bucket/datasets/train_manifest.json",
        "validation_manifest_uri": "s3://example-bucket/datasets/validation_manifest.json",
        "class_map_uri": "s3://example-bucket/datasets/class_map.json",
        "dataset_summary_uri": "s3://example-bucket/datasets/dataset_summary.json",
    },
    "train": {
        "run_id": "smoke",
        "best_checkpoint_uri": "s3://example-bucket/experiments/completed/smoke/best_checkpoint.pt",
        "last_checkpoint_uri": "s3://example-bucket/experiments/completed/smoke/last_checkpoint.pt",
        "training_history_uri": "s3://example-bucket/experiments/completed/smoke/training_history.json",
    },
    "evaluate": {
        "run_id": "smoke",
        "metrics_uri": "s3://example-bucket/experiments/completed/smoke/metrics.json",
        "predictions_uri": "s3://example-bucket/experiments/completed/smoke/predictions.json",
    },
}


def build_smoke_config(config: dict, *, run_id: str, record_uri: str) -> dict:
    """원본 config를 바꾸지 않고 smoke test용 registry 설정을 얹습니다."""

    smoke_config = copy.deepcopy(config)
    smoke_config["registry"] = {
        "run_id": run_id,
        "record_uri": record_uri,
        "verify_artifacts": False,
    }
    smoke_config["inputs"] = copy.deepcopy(SMOKE_INPUTS)
    smoke_config["purpose"] = "pill-object-detection-registry-smoke-test"
    return smoke_config


def run_smoke_test(config_path: str, *, smoke_id: str | None = None) -> dict:
    """Experiment record를 실제 S3에 등록하고 다시 읽어서 비교합니다."""

    config = load_config(config_path)
    storage = create_storage(config)
    if not isinstance(storage, S3Storage):
        raise StorageError("smoke test에는 s3 storage backend 설정이 필요합니다.")

    resolved_id = smoke_id or uuid.uuid4().hex
    run_id = f"smoke-{resolved_id}"
    record_uri = f"{SMOKE_PREFIX}{resolved_id}/experiment_record.json"

    result = run_registry(build_smoke_config(config, run_id=run_id, record_uri=record_uri))
    if result["status"] != "ok":
        raise RegistryError(f"registry pipeline이 실패했습니다: {result['message']}")

    written_uri = result["artifacts"]["experiment_record_uri"]
    if not written_uri.lower().startswith("s3://"):
        raise StorageError(f"S3 URI가 아닌 경로에 기록했습니다: {written_uri}")

    fetched = storage.read_json(written_uri)
    if fetched.get("run_id") != run_id:
        raise StorageError("등록한 record와 조회한 record의 run_id가 다릅니다.")
    if fetched.get("schema_version") != "1.1":
        raise StorageError("조회한 record의 schema_version이 예상과 다릅니다.")
    if not storage.exists(written_uri):
        raise StorageError("등록한 record를 exists()로 확인하지 못했습니다.")

    listed_objects = storage.list(SMOKE_PREFIX)
    if written_uri not in listed_objects:
        raise StorageError("등록한 record가 smoke-test prefix listing에 없습니다.")

    return {
        "status": "ok",
        "run_id": run_id,
        "experiment_record_uri": written_uri,
        "listed_count": len(listed_objects),
        "artifacts_skipped_remote": result["summary"]["artifacts_skipped_remote"],
        "message": "S3 experiment record 등록, 조회, 내용 비교, prefix listing 확인 완료",
        "cleanup": (
            "자동 삭제하지 않았습니다. 위 experiment_record_uri의 smoke-test object를 "
            "직접 확인하고 정리하세요."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Registry pipeline의 실제 S3 experiment record 등록/조회 smoke test"
    )
    parser.add_argument("--config", required=True, help="AWS storage JSON config 경로")
    args = parser.parse_args(argv)

    try:
        result = run_smoke_test(args.config)
    except (StorageError, RegistryError, OSError, ValueError) as error:
        print(f"registry S3 smoke test 실패: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
