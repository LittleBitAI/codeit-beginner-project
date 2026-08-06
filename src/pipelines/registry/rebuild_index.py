"""빠진 experiment index를 experiment record에서 다시 만드는 명령입니다.

Index는 record에서 언제든 다시 만들 수 있는 cache이고 record가 진실입니다. 실행 중
index 저장이 실패했거나 index prefix를 바꿨을 때 이 명령으로 채워 넣습니다.

    python -m src.pipelines.registry.rebuild_index --config configs/env.local.json

기본은 `overwrite=False`라서 이미 있는 index는 건드리지 않습니다. Record는 어떤
경우에도 읽기만 하며 고치거나 지우지 않습니다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from src.common import StorageError, create_storage, load_config

from . import (
    _DEFAULT_RECORD_NAME,
    _DEFAULT_RECORD_PREFIX,
    _relative_to_repo,
    _resolve_repo_root,
    resolve_index_prefix,
)
from .record import RegistryError
from .smoke_s3 import SMOKE_PREFIX
from .summary import build_summary


__all__ = ["main", "rebuild_index"]


def _is_record_location(location: str) -> bool:
    """Experiment record file인지 확인합니다. smoke test 전용 prefix는 제외합니다."""

    normalized = location.replace("\\", "/")
    if not normalized.endswith(f"/{_DEFAULT_RECORD_NAME}"):
        return False
    return f"/{SMOKE_PREFIX}" not in f"/{normalized}"


def rebuild_index(config: Mapping[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Record를 훑어 빠진 index만 다시 만들고 결과를 요약해 돌려줍니다."""

    registry_config = config.get("registry")
    registry_config = registry_config if isinstance(registry_config, Mapping) else {}
    repo_root = _resolve_repo_root(registry_config)
    index_prefix = resolve_index_prefix(registry_config)

    storage = create_storage(config)
    locations = [
        location
        for location in storage.list(f"{_DEFAULT_RECORD_PREFIX}/")
        if _is_record_location(location)
    ]

    written = 0
    skipped = 0
    failed: list[str] = []

    for location in locations:
        try:
            record = storage.read_json(location)
        except StorageError:
            failed.append(location)
            continue

        run_id = record.get("run_id") if isinstance(record, Mapping) else None
        if not isinstance(run_id, str) or not run_id.strip():
            failed.append(location)
            continue

        destination = f"{index_prefix}/{run_id.strip()}.json"
        if not overwrite and storage.exists(destination):
            skipped += 1
            continue

        summary = build_summary(
            record,
            record_uri=_relative_to_repo(location, repo_root),
            repo_root=repo_root,
        )
        try:
            storage.write_json(destination, summary, overwrite=overwrite)
        except StorageError:
            failed.append(location)
            continue
        written += 1

    return {
        "records_found": len(locations),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "index_prefix": index_prefix,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 진입점입니다. 실패하면 0이 아닌 exit code를 돌려줍니다."""

    parser = argparse.ArgumentParser(
        description="빠진 experiment index를 record에서 다시 만듭니다."
    )
    parser.add_argument("--config", required=True, help="실행 설정 JSON 경로")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 있는 index도 다시 씁니다. 기본은 건드리지 않습니다.",
    )
    arguments = parser.parse_args(argv)

    try:
        report = rebuild_index(load_config(arguments.config), overwrite=arguments.overwrite)
    except (StorageError, RegistryError, OSError, ValueError) as error:
        print(f"index를 다시 만들지 못했습니다: {error}", file=sys.stderr)
        return 1

    print(
        f"record {report['records_found']}개를 확인해 index {report['written']}개를 "
        f"새로 쓰고 {report['skipped']}개를 건너뛰었습니다."
    )
    if report["failed"]:
        print(f"읽지 못한 record {len(report['failed'])}개가 있습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
