"""Experiment record와 summary를 읽는 공용 facade.

`read_experiment_record()`는 지금까지처럼 **정확한 URI 하나만** 읽습니다. prefix
listing이나 최신 record 탐색은 하지 않습니다. 목록·검색·비교는 registry가 따로
발행한 summary index만 읽는 별개 경로이며, 원하는 index가 없다고 해서 record를
추측해 fallback하지 않습니다.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .storage import LocalStorage, ObjectNotFoundError, StorageError, create_storage


__all__ = [
    "ExperimentRegistryError",
    "compare_experiment_summaries",
    "list_experiment_summaries",
    "read_experiment_record",
    "read_experiment_summary",
    "search_experiment_summaries",
]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Registry가 summary sidecar를 남기는 기본 prefix입니다.
DEFAULT_INDEX_PREFIX = "registry/index"

#: 이름에 이 글자가 있으면 storage에 넘기기 전에 거부합니다. 넘기면 backend가 아니라
#: 그 아래 OS가 죽습니다 — NUL이 든 이름은 pathlib이 :class:`ValueError`를 던지고,
#: 그것은 이 module의 오류가 아니라서 호출자가 다루지 못한 채 새어 나갑니다.
#:
#: 경로 구분자는 **막지 않습니다.** registry는 지정된 run_id를 검증 없이 그대로 써서
#: ``폴더/이름``이 ``registry/index/폴더/이름.json``으로 저장되고, 목록 경로는 그것을
#: 찾아냅니다. 읽는 쪽만 거부하면 등록은 됐는데 조회는 안 되는 실험이 생깁니다.
#: index 밖으로 나가는 이름은 아래에서 따로 판정합니다.
_UNSAFE_RUN_ID_CHARACTERS = frozenset("\x7f") | {chr(code) for code in range(0x20)}


#: 비교 화면이 실험 사이에서 실제로 견주는 값들입니다.
_COMPARABLE_FIELDS = (
    "created_at",
    "seed",
    "schema_version",
    "metrics_source",
    "experiment_record_uri",
)


class ExperimentRegistryError(RuntimeError):
    """Experiment record 조회 또는 최소 schema 검증이 실패한 경우입니다."""


def _repo_root(config: Mapping[str, Any] | None) -> Path:
    """Registry와 같은 config 규칙으로 repository root를 결정합니다."""

    registry_config = config.get("registry") if isinstance(config, Mapping) else None
    if registry_config is None:
        return _REPOSITORY_ROOT
    if not isinstance(registry_config, Mapping):
        raise ExperimentRegistryError("config['registry']는 object여야 합니다.")

    configured = registry_config.get("repo_root")
    if configured is None:
        return _REPOSITORY_ROOT
    if not isinstance(configured, str) or not configured.strip():
        raise ExperimentRegistryError(
            "config['registry']['repo_root']는 비어 있지 않은 문자열이어야 합니다."
        )
    return Path(configured).expanduser().resolve()


def _local_read_source(uri: str, storage: LocalStorage, repo_root: Path) -> str:
    """Registry의 repo-relative URI를 LocalStorage root 기준으로 맞춥니다.

    이름이 같은 path segment를 추측해 제거하지 않습니다. URI를 registry의
    repository root 기준 절대 후보로 만든 뒤, 그 후보가 실제 LocalStorage root
    안에 있을 때만 storage 기준 상대 경로로 바꿉니다.
    """

    candidate = Path(uri)
    if candidate.is_absolute():
        return uri

    absolute_candidate = (repo_root / candidate).resolve()
    try:
        return absolute_candidate.relative_to(storage.root).as_posix()
    except ValueError:
        return uri


def read_experiment_record(
    experiment_record_uri: str,
    config: Mapping[str, Any] | None,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """정확한 URI의 experiment record를 읽고 최소 identity를 검증합니다.

    Prefix listing이나 최신 record 탐색은 하지 않습니다. Storage 오류와 record
    schema 오류는 호출자가 backend와 무관하게 처리할 수 있도록
    :class:`ExperimentRegistryError`로 통일합니다.
    """

    if not isinstance(experiment_record_uri, str) or not experiment_record_uri.strip():
        raise ExperimentRegistryError(
            "experiment_record_uri는 비어 있지 않은 문자열이어야 합니다."
        )
    uri = experiment_record_uri.strip()

    if expected_run_id is not None and (
        not isinstance(expected_run_id, str) or not expected_run_id.strip()
    ):
        raise ExperimentRegistryError(
            "expected_run_id는 비어 있지 않은 문자열이어야 합니다."
        )
    expected = expected_run_id.strip() if isinstance(expected_run_id, str) else None

    try:
        storage = create_storage(config)
        source = (
            _local_read_source(uri, storage, _repo_root(config))
            if isinstance(storage, LocalStorage)
            else uri
        )
        record = storage.read_json(source)
    except StorageError as error:
        raise ExperimentRegistryError(
            "experiment record storage 조회에 실패했습니다 "
            f"({type(error).__name__})."
        ) from error

    if not isinstance(record, dict):
        raise ExperimentRegistryError(
            "experiment record는 object(dict)여야 하지만 "
            f"{type(record).__name__}을(를) 받았습니다."
        )

    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ExperimentRegistryError(
            "experiment record의 run_id는 비어 있지 않은 문자열이어야 합니다."
        )
    resolved_run_id = run_id.strip()
    if expected is not None and resolved_run_id != expected:
        raise ExperimentRegistryError(
            "experiment record의 run_id가 요청과 다릅니다: "
            f"expected={expected}, actual={resolved_run_id}"
        )

    return record


def _index_prefix(config: Mapping[str, Any] | None) -> str:
    """Summary index prefix를 registry와 같은 설정 규칙으로 결정합니다."""

    registry_config = config.get("registry") if isinstance(config, Mapping) else None
    if registry_config is None:
        return DEFAULT_INDEX_PREFIX
    if not isinstance(registry_config, Mapping):
        raise ExperimentRegistryError("config['registry']는 object여야 합니다.")

    configured = registry_config.get("index_prefix", DEFAULT_INDEX_PREFIX)
    if not isinstance(configured, str) or not configured.strip():
        raise ExperimentRegistryError(
            "config['registry']['index_prefix']는 비어 있지 않은 문자열이어야 합니다."
        )
    return configured.strip().strip("/")


def read_experiment_summary(
    run_id: str, config: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Index에서 실험 **하나의** summary만 읽습니다. 등록된 적이 없으면 ``None``입니다.

    Registry는 index를 ``<prefix>/<run_id>.json`` 한 파일로 남깁니다. 그래서 이름을
    이미 아는 조회에는 목록이 필요하지 않습니다. :func:`list_experiment_summaries`는
    등록된 실험 수만큼 storage를 왕복하므로, 상세·비교처럼 무엇을 볼지 정해진 화면이
    그 길을 쓰면 기록이 쌓일수록 함께 느려집니다.

    없는 것과 못 읽은 것은 구분합니다. 파일이 없으면 ``None``이고, 읽지 못했거나
    index가 다른 실험을 가리키면 :class:`ExperimentRegistryError`입니다 — 못 읽은
    것을 "없다"로 답하면 화면이 지워지지 않은 실험을 사라졌다고 말하게 됩니다.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ExperimentRegistryError("run_id는 비어 있지 않은 문자열이어야 합니다.")
    wanted = run_id.strip()
    if not _UNSAFE_RUN_ID_CHARACTERS.isdisjoint(wanted):
        raise ExperimentRegistryError("run_id에는 제어문자를 쓸 수 없습니다.")

    prefix = _index_prefix(config)
    # `index_prefix="/"`는 검증을 지나 빈 prefix가 됩니다. 그때 앞에 구분자를 붙이면
    # 절대 경로가 되어, S3에서 bucket root에 저장돼 목록에 잡히는 실험을 못 읽습니다.
    location = f"{prefix}/{wanted}.json" if prefix else f"{wanted}.json"
    # `..`가 섞인 이름은 파일 시스템이 먼저 정규화합니다. 그 결과가 index 밖으로 나가면
    # 목록에 없는 파일을 읽게 되므로, 같은 규칙으로 미리 계산해 봅니다. Windows의
    # LocalStorage는 역슬래시도 구분자로 읽으므로 이름과 prefix 양쪽에서 함께 눕힙니다 —
    # 한쪽만 눕히면 `registry\index` 같은 prefix에서 모든 이름이 밖으로 판정됩니다.
    #
    # segment를 이름만 보고 거부하지 않는 이유는, `a/../b`처럼 index **안에서** 상쇄되는
    # 이름을 registry가 실제로 저장하고 목록도 세기 때문입니다. 읽는 쪽만 거부하면 목록에
    # 보이는 실험을 열지 못합니다. 반대로 `..foo`는 밖으로 나가지 않으므로 그대로 둡니다.
    base = posixpath.normpath(prefix.replace("\\", "/"))
    inside = posixpath.relpath(posixpath.normpath(location.replace("\\", "/")), base)
    if inside == ".." or inside.startswith("../"):
        raise ExperimentRegistryError("run_id는 index 밖을 가리킬 수 없습니다.")

    try:
        summary = create_storage(config).read_json(location)
    except ObjectNotFoundError:
        return None
    except StorageError as error:
        # 깨진 JSON도 여기로 옵니다. 두 backend 모두 그것을 StorageError로 감싸므로
        # `json.JSONDecodeError`를 따로 잡을 자리가 없습니다.
        raise ExperimentRegistryError(
            f"experiment index 조회에 실패했습니다 ({type(error).__name__})."
        ) from error

    if not isinstance(summary, dict) or summary.get("run_id") != wanted:
        raise ExperimentRegistryError(
            f"'{wanted}' index 항목이 다른 실험을 가리킵니다."
        )
    return summary


def _sort_value(summary: Mapping[str, Any], field: str) -> tuple[int, Any]:
    """정렬 key를 만듭니다. 값이 없는 실험은 항상 뒤로 보냅니다."""

    if field in {"mAP", "mAP50", "mAP75", "precision50", "recall50"}:
        metrics = summary.get("metrics")
        value = metrics.get(field) if isinstance(metrics, Mapping) else None
    else:
        value = summary.get(field)
    if value is None:
        return (1, "")
    return (0, value)


def list_experiment_summaries(
    config: Mapping[str, Any] | None,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Index prefix 아래의 experiment summary를 최신순으로 돌려줍니다.

    Record 자체는 읽지 않습니다. 항목 하나를 읽지 못해도 목록 전체를 실패시키지 않고
    그 항목만 건너뜁니다. 실험 하나가 깨졌다고 화면 전체가 비면 더 곤란하기
    때문입니다.
    """

    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ExperimentRegistryError("offset은 0 이상의 정수여야 합니다.")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise ExperimentRegistryError("limit은 0 이상의 정수여야 합니다.")

    prefix = _index_prefix(config)
    try:
        storage = create_storage(config)
        locations = storage.list(f"{prefix}/")
    except StorageError as error:
        raise ExperimentRegistryError(
            f"experiment index 목록 조회에 실패했습니다 ({type(error).__name__})."
        ) from error

    summaries: list[dict[str, Any]] = []
    for location in locations:
        if not location.replace("\\", "/").endswith(".json"):
            continue
        try:
            summary = storage.read_json(location)
        except (StorageError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict):
            continue
        run_id = summary.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        summaries.append(summary)

    summaries.sort(key=lambda item: _sort_value(item, "created_at"), reverse=True)
    end = None if limit is None else offset + limit
    return summaries[offset:end]


def search_experiment_summaries(
    config: Mapping[str, Any] | None,
    *,
    run_id_contains: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    min_map: float | None = None,
    has_submission: bool | None = None,
    sort_by: str = "created_at",
    descending: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """조건에 맞는 experiment summary만 골라 돌려줍니다.

    조건을 하나도 주지 않으면 전체 목록과 같습니다. 값이 없는(None) 실험은 비교할 수
    없으므로 해당 조건이 걸린 검색에서 제외합니다.
    """

    summaries = list_experiment_summaries(config)
    selected: list[dict[str, Any]] = []

    for summary in summaries:
        run_id = summary.get("run_id")
        if run_id_contains and run_id_contains not in str(run_id):
            continue

        created_at = summary.get("created_at")
        if created_after is not None and not (
            isinstance(created_at, str) and created_at >= created_after
        ):
            continue
        if created_before is not None and not (
            isinstance(created_at, str) and created_at <= created_before
        ):
            continue

        if min_map is not None:
            metrics = summary.get("metrics")
            value = metrics.get("mAP") if isinstance(metrics, Mapping) else None
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if value < min_map:
                continue

        if has_submission is not None:
            artifacts = summary.get("artifacts")
            uri = artifacts.get("submission_uri") if isinstance(artifacts, Mapping) else None
            if bool(uri) is not has_submission:
                continue

        selected.append(summary)

    selected.sort(key=lambda item: _sort_value(item, sort_by), reverse=descending)
    return selected if limit is None else selected[:limit]


def compare_experiment_summaries(
    run_ids: Sequence[str],
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """실험 여러 개를 field 단위로 견줍니다.

    각 field마다 실험별 값과 값이 서로 다른지를 함께 돌려주므로, 화면은 다른 값만
    강조하면 됩니다. Index에 없는 run_id는 `missing`에 모아 알립니다.
    """

    if not isinstance(run_ids, Sequence) or isinstance(run_ids, str):
        raise ExperimentRegistryError("run_ids는 문자열 목록이어야 합니다.")
    requested = [str(run_id).strip() for run_id in run_ids]
    if not requested or not all(requested):
        raise ExperimentRegistryError(
            "run_ids에는 비어 있지 않은 run_id가 하나 이상 필요합니다."
        )

    found = {
        summary["run_id"]: summary
        for summary in list_experiment_summaries(config)
        if summary.get("run_id") in requested
    }
    missing = [run_id for run_id in requested if run_id not in found]
    resolved = [run_id for run_id in requested if run_id in found]

    metric_keys: list[str] = []
    for run_id in resolved:
        metrics = found[run_id].get("metrics")
        if isinstance(metrics, Mapping):
            metric_keys.extend(key for key in metrics if key not in metric_keys)

    fields: dict[str, Any] = {}
    for field in (*_COMPARABLE_FIELDS, *metric_keys):
        values = {run_id: _sort_value(found[run_id], field)[1] for run_id in resolved}
        for run_id in resolved:
            if _sort_value(found[run_id], field)[0] == 1:
                values[run_id] = None
        fields[field] = {
            "values": values,
            "differs": len({json.dumps(value, sort_keys=True) for value in values.values()}) > 1,
        }

    return {"run_ids": resolved, "fields": fields, "missing": missing}
