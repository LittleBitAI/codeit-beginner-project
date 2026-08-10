"""실험별 Kaggle 실제 점수를 Web 소유 상태로 저장합니다."""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.common import ObjectAlreadyExistsError, StorageError, create_storage

from .datasets import storage_environment
from .errors import WebStateError
from .paths import repository_root


_LOCAL_PREFIX = "artifacts/web/kaggle-scores"
_S3_PREFIX = "experiments/web-state/kaggle-scores"
_CACHE_SECONDS = 30.0
_READ_WORKERS = 8
_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, ...], tuple[float, dict[str, float]]] = {}


def _scope() -> tuple[dict[str, Any], str, tuple[str, ...]]:
    environment = storage_environment()
    backend = environment["default_backend"]
    if backend == "s3":
        config = {"storage": {"backend": "s3", "s3": {"prefix": ""}}}
        prefix = _S3_PREFIX
        key = (
            "s3",
            str(environment.get("bucket") or ""),
            os.environ.get("PILL_STORAGE_S3_PREFIX", "").strip(),
        )
        return config, prefix, key
    config = {
        "storage": {
            "backend": "local",
            "local": {"root": str(repository_root())},
        }
    }
    return config, _LOCAL_PREFIX, ("local", str(repository_root()))


def _document_path(prefix: str, run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return f"{prefix}/{digest}.json"


def _score(document: Any) -> tuple[str, float] | None:
    if not isinstance(document, Mapping):
        return None
    run_id = document.get("run_id")
    score = document.get("score")
    if not isinstance(run_id, str) or not run_id.strip():
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return run_id, value


def load_scores() -> dict[str, float]:
    """현재 Registry 저장소와 같은 범위의 실제 점수를 읽습니다."""

    config, prefix, cache_key = _scope()
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None and cached[0] > now:
            return dict(cached[1])

    try:
        storage = create_storage(config)
        locations = storage.list(f"{prefix}/")
        workers = max(1, min(_READ_WORKERS, len(locations)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            documents = list(executor.map(storage.read_json, locations))
    except StorageError as error:
        raise WebStateError(
            f"Kaggle 실제 점수 기록을 읽지 못했습니다({type(error).__name__})."
        ) from error

    scores = {}
    for document in documents:
        parsed = _score(document)
        if parsed is not None:
            scores[parsed[0]] = parsed[1]
    with _CACHE_LOCK:
        _CACHE[cache_key] = (now + _CACHE_SECONDS, dict(scores))
    return scores


def save_score(run_id: str, score: float, *, overwrite: bool = False) -> bool:
    """새 실제 점수를 저장합니다. 이미 있으면 보존하고 ``False``를 반환합니다.

    ``overwrite=True``는 사람이 화면에서 수정 버튼을 켜고 기존 기록을 고치겠다고
    밝혔을 때만 옵니다. 손이 미끄러진 저장이 기록을 지우지 못하도록 기본값은
    ``False``로 두고, 덮어쓸지 말지는 부르는 쪽이 매번 정하게 합니다.
    """

    config, prefix, cache_key = _scope()
    document = {"version": 1, "run_id": run_id, "score": score}
    try:
        create_storage(config).write_json(
            _document_path(prefix, run_id), document, overwrite=overwrite
        )
    except ObjectAlreadyExistsError:
        return False
    except StorageError as error:
        raise WebStateError(
            f"Kaggle 실제 점수를 저장하지 못했습니다({type(error).__name__})."
        ) from error
    with _CACHE_LOCK:
        _CACHE.pop(cache_key, None)
    return True


__all__ = ["load_scores", "save_score"]
