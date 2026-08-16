"""여러 실행의 test 예측을 합치기 전에, 합칠 값어치가 있는지 먼저 잽니다.

앙상블의 이득은 **돌려 보기 전에는 모릅니다.** 그리고 확인하는 유일한 방법이 Kaggle
제출이라, 한 번 잘못 고르면 하루치 제출을 버립니다. 실제로 이 저장소에서 이런 일이
있었습니다.

- 일곱 개를 합쳤더니 0.62087로 **단독 최고(0.62437)보다 낮았습니다.** 약한 실행이
  결과를 pool 평균 쪽으로 끌어내린 것입니다.
- 상위 셋만 합치니 0.62645로 뒤집혔습니다. 같은 코드, 고른 것만 달랐습니다.
- 다른 dataset 판으로 만든 예측을 섞으려다 evaluate가 거부해 실행 자체가 실패했습니다.
- 아키텍처가 다른 실행을 넣어 보려다, 먼저 재 보니 96.9%가 같은 예측이라 그만뒀습니다.

**넷 다 합치기 전에 알 수 있는 것들이었습니다.** 이 모듈이 그것을 미리 재서 알려 줍니다.
막지는 않습니다 — 예측이 틀릴 때가 있고, 막아 버리면 반증할 길까지 막힙니다.
"""

from __future__ import annotations

import hashlib
import statistics
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from src.common import (
    ExperimentRegistryError,
    LocalStorage,
    S3Storage,
    StorageError,
    create_storage,
    list_experiment_summaries,
)

from . import kaggle_scores
from .datasets import storage_environment
from .errors import WebError, WebStateError
from .experiments import registry_config
from .paths import repository_root
from .train_config import RUN_ID_PATTERN


__all__ = [
    "build_fusion_config",
    "build_harvest_config",
    "check_selection",
    "diagnose",
    "list_candidates",
    "pending_runs",
]


#: 제출 CSV가 이미지마다 남기는 수입니다. 다양성도 **제출에 남을 것**끼리 재야
#: 의미가 있어서 같은 수로 자릅니다.
_SUBMISSION_LIMIT = 4
#: 이 아래로 붙으면 "거의 같은 예측"입니다. 0.62645를 낸 조합이 97.7%였고 이득이
#: +0.002였습니다. 그보다 붙어 있으면 이득을 기대하기 어렵습니다.
_AGREEMENT_WARN = 0.95
#: 최고 실행보다 이만큼 낮은 실행은 결과를 평균 쪽으로 끌어내립니다. 0.57485짜리를
#: 넣었을 때 실제로 그랬습니다.
_WEAK_MARGIN = 0.02
#: 합치는 수가 늘수록 고유 class가 줄었습니다(84 → 82 → 81 → 78 → 77). 확신도가
#: 동의 수로 깎여 드물게 잡히는 class가 상한 밖으로 밀리기 때문입니다.
_CLASS_DROP_FROM = 4

_LOCAL_PREFIX = "artifacts/web/ensemble-diversity"
_S3_PREFIX = "experiments/web-state/ensemble-diversity"

_LOCK = threading.Lock()
_PREDICTION_CACHE: dict[str, dict[str, Any]] = {}
_PAIR_CACHE: dict[tuple[str, str], dict[str, float]] = {}


def _uses_s3() -> bool:
    """S3에 쓰는가. **자리를 정하는 모든 곳이 이 하나를 봐야** 서로 갈리지 않습니다."""

    return storage_environment()["default_backend"] == "s3"


def _storage_root() -> str:
    """S3일 때의 `s3://<bucket>`입니다."""

    return f"s3://{(storage_environment().get('bucket') or '').strip()}"


def _storage_config() -> dict[str, Any]:
    """pipeline에 넘길 storage 설정입니다. 자리를 정하는 곳과 같은 기준으로 갈립니다."""

    if _uses_s3():
        bucket = (storage_environment().get("bucket") or "").strip()
        return {"backend": "s3", "s3": {"bucket": bucket, "prefix": ""}}
    return {"backend": "local", "local": {"root": str(repository_root())}}


def _checkpoint_identity(uri: str) -> str:
    """checkpoint의 신원입니다. 표기가 달라도 같은 파일이면 같은 값입니다.

    **규칙을 새로 적지 않고 공용 저장 계층에 맡깁니다.** 직접 풀었더니 계층과 반대로
    움직였습니다 — S3에서 `a//x.pt`와 `a/x.pt`는 서로 **다른** key인데 같다고 보았고,
    percent 표기(`%2F`)는 같은 객체인데 다르다고 보았습니다. 규칙을 두 곳에 적으면
    이렇게 갈립니다.

    `s3://`는 **주소에서 bucket을 뽑아** 그 저장소에 물어봅니다. `identity()`는 원격에
    닿지 않으므로 credential도 설정도 필요 없고, bucket이 설정되지 않은 환경(기본
    local, test)에서도 됩니다.

    **오류는 숨기지 않습니다.** 글자로 되돌리면 같은 파일을 두 번 넣은 것을 놓칩니다.
    """

    try:
        if uri.startswith("s3://"):
            bucket = uri[len("s3://"):].split("/", 1)[0]
            return str(S3Storage(bucket=bucket).identity(uri))
        return str(LocalStorage(repository_root()).identity(uri))
    except StorageError as error:
        raise WebError(
            f"{uri}의 저장 신원을 얻지 못했습니다({type(error).__name__}). 같은 "
            "checkpoint를 두 번 넣었는지 가릴 수 없어 멈춥니다."
        ) from error


def _submission_uri(run_id: str) -> str:
    """융합 결과 제출 CSV가 놓일 자리입니다."""

    if _uses_s3():
        return f"{_storage_root()}/submissions/{run_id}/submission.csv"
    return f"artifacts/ensemble/{run_id}/submission.csv"


def _scope() -> tuple[dict[str, Any], str]:
    """kaggle 점수와 같은 자리 규칙을 씁니다."""

    if _uses_s3():
        return {"storage": {"backend": "s3", "s3": {"prefix": ""}}}, _S3_PREFIX
    return (
        {"storage": {"backend": "local", "local": {"root": str(repository_root())}}},
        _LOCAL_PREFIX,
    )


def _text(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _dataset_label(manifest_uri: str) -> str | None:
    """`.../processed/v5-seed42-8020-group/train_manifest.json` -> 폴더 이름입니다."""

    parts = manifest_uri.replace("\\", "/").rstrip("/").split("/")
    return parts[-2] if len(parts) >= 2 else None


# --------------------------------------------------------------------------- 후보

#: 이 화면이 만든 test 예측을 두는 자리입니다. web이 소유한 상태 아래에 둡니다 —
#: 남이 만든 artifact를 덮어쓰지 않으려는 것입니다.
_HARVEST_PREFIX = "experiments/web-state/ensemble-candidates"
#: local backend에서 쓰는 같은 자리입니다. `artifacts/web/` 아래는 gitignore됩니다.
_LOCAL_HARVEST_PREFIX = "artifacts/web/ensemble-candidates"
#: 합치려면 이미지마다 후보가 넉넉해야 합니다. 제출이 남기는 4개만 저장하면 융합이
#: 고를 것이 없습니다. 그래서 이 파일의 CSV는 제출할 수 없고, 그것이 의도입니다.
_HARVEST_DETECTIONS = 20
#: 체크포인트 하나로 842장을 추론하는 데 걸린 실측값입니다(GPU). 고르는 사람이
#: "지금 눌러도 되는가"를 판단할 수 있어야 합니다.
_HARVEST_MINUTES = 9


def _harvest_root() -> str:
    """이 화면이 만든 예측을 두는 자리입니다. backend에 따라 갈립니다.

    S3만 지원하면 기본 local 환경에서 "예측을 먼저 만든다"가 통째로 실패합니다.
    목록에는 후보가 보이는데 누르면 안 되는 셈이라 더 나쁩니다.
    """

    # **`_scope()`와 같은 기준으로 갈립니다.** bucket이 있는지만 보면
    # `PILL_STORAGE_BACKEND=local`에 bucket이 함께 설정된 환경에서 저장은 local로,
    # 경로는 s3://로 갈려 실행이 실패합니다.
    return f"{_storage_root()}/{_HARVEST_PREFIX}" if _uses_s3() else _LOCAL_HARVEST_PREFIX


def _harvest_uri(run_id: str) -> str:
    return f"{_harvest_root()}/{run_id}/test_predictions.json"


def _harvest_match_prefix() -> str:
    """목록 결과와 견줄 앞자리입니다.

    local 저장소는 **절대 경로**를 돌려주므로, 저장소 뿌리를 앞에 붙여야 같은 자리를
    가리킵니다. 앞자리를 안 맞추면 `startswith`가 늘 거짓이 되어 수확한 것을 못 찾거나,
    반대로 느슨하게 견주면 남의 폴더까지 걸립니다.
    """

    root = _harvest_root()
    if root.startswith("s3://"):
        return root
    return str(repository_root() / root).replace("\\", "/")


def list_candidates() -> list[dict[str, Any]]:
    """**체크포인트가 있는 학습은 전부 후보입니다.**

    처음에는 `test_predictions.json`을 이미 남긴 실행만 후보로 삼았는데, 그 key가
    계약에 늦게 들어와서(제안서 017) 47개 중 2개만 목록에 떴습니다. 가장 점수가 높은
    실행들이 전부 빠졌습니다.

    예측이 없는 것은 **만들면 됩니다.** 그래서 목록은 체크포인트로 정하고, 예측이
    있는지는 `ready`로 알려 줍니다. 새로 학습한 실행도 등록되는 순간 후보가 됩니다.
    """

    try:
        summaries = list_experiment_summaries(registry_config())
    except ExperimentRegistryError as error:
        raise WebError(f"실험 목록을 읽지 못했습니다({type(error).__name__}).") from error

    scores = kaggle_scores.load_scores()
    harvested = _harvested_runs()

    candidates: list[dict[str, Any]] = []
    for summary in summaries:
        run_id = _text(summary.get("run_id"))
        artifacts = summary.get("artifacts")
        if run_id is None or not isinstance(artifacts, Mapping):
            continue
        checkpoint = _text(artifacts.get("best_checkpoint_uri"))
        if checkpoint is None:
            # 체크포인트가 없으면 예측을 만들 길이 없습니다.
            continue
        data_inputs = {
            key: _text(artifacts.get(key))
            for key in ("validation_manifest_uri", "test_manifest_uri", "class_map_uri")
        }
        # 기록이 가리키는 것이 먼저입니다. 없으면 이 화면이 만들어 둔 것을 씁니다.
        predictions = _text(artifacts.get("test_predictions_uri"))
        if predictions is None and run_id in harvested:
            predictions = _harvest_uri(run_id)
        candidates.append(
            {
                "run_id": run_id,
                "checkpoint_uri": checkpoint,
                "test_predictions_uri": predictions,
                "ready": predictions is not None,
                "dataset_label": _dataset_label(_text(artifacts.get("train_manifest_uri")) or ""),
                "kaggle_score": scores.get(run_id),
                "created_at": _text(summary.get("created_at")),
                # 융합 config를 만들 때 그대로 씁니다. 고른 실행이 대던 것을 쓰지 않고
                # 새로 정하면 다른 시험지를 채점하게 됩니다.
                "data_inputs": data_inputs,
                # validation 지표는 융합이 만들지 않습니다. 가장 센 실행의 것을 씁니다.
                "predictions_uri": _text(artifacts.get("predictions_uri")),
            }
        )
    # 점수가 있는 것을 위로, 그중 높은 것을 위로 둡니다. 고르는 사람이 보는 순서가
    # 곧 pool 품질 순서여야 약한 실행을 무심코 넣지 않습니다.
    candidates.sort(key=lambda item: (item["kaggle_score"] is None, -(item["kaggle_score"] or 0.0)))
    return candidates


def _harvested_runs() -> set[str]:
    """이 화면이 이미 예측을 만들어 둔 실행입니다. 목록 한 번으로 끝냅니다."""

    config, _ = _scope()
    root = _harvest_root()
    try:
        entries = [str(item) for item in create_storage(config).list(root)]
    except StorageError:
        # 못 읽으면 "아직 없다"와 같게 다룹니다. 다시 만들면 되고, 목록 전체를
        # 못 보게 만들 이유가 없습니다.
        return set()
    # **자리를 정확히 견줍니다.** 파일이 든 폴더 이름만 보면 `<root>/a/b/…` 같은 깊은
    # 경로의 `b`까지 실행 이름으로 잡히고, 구분자 없는 앞자리로 거르면
    # `ensemble-candidates-old/…`까지 걸립니다. **뿌리 바로 아래 한 칸**만 받습니다.
    prefix = _harvest_match_prefix().rstrip("/") + "/"
    found: set[str] = set()
    for entry in entries:
        normalized = str(entry).replace("\\", "/")
        if not normalized.startswith(prefix):
            continue
        parts = normalized[len(prefix):].split("/")
        if len(parts) == 2 and parts[0] and parts[1] == "test_predictions.json":
            found.add(parts[0])
    return found


# ----------------------------------------------------------------------- 예측 읽기

def _read_predictions(uri: str) -> dict[str, Any]:
    with _LOCK:
        cached = _PREDICTION_CACHE.get(uri)
    if cached is not None:
        return cached

    config, _ = _scope()
    try:
        storage = create_storage(config)
        document = storage.read_json(uri)
    except (StorageError, ValueError) as error:
        raise WebError(f"{uri}를 읽지 못했습니다({type(error).__name__}).") from error
    if not isinstance(document, Mapping):
        raise WebError(f"{uri}: 예측 파일이 object가 아닙니다.")

    predictions = document.get("predictions")
    if not isinstance(predictions, Sequence):
        raise WebError(f"{uri}: predictions가 없습니다.")

    # 이미지마다 제출에 남을 것만 남깁니다. 뒤쪽 저확신 후보는 제출에 안 들어가므로
    # 다양성에 세면 실제와 다른 그림이 나옵니다.
    by_image: dict[Any, list[Mapping[str, Any]]] = {}
    for item in predictions:
        if isinstance(item, Mapping):
            by_image.setdefault(item.get("image_id"), []).append(item)
    kept: dict[tuple[Any, Any], tuple[float, ...]] = {}
    for image_id, items in by_image.items():
        items.sort(key=lambda row: -float(row.get("score") or 0.0))
        for row in items[:_SUBMISSION_LIMIT]:
            box = row.get("bbox")
            if isinstance(box, Sequence) and len(box) == 4:
                kept[(image_id, row.get("category_id"))] = tuple(float(v) for v in box)

    loaded = {
        "uri": uri,
        "test_manifest_uri": _text(document.get("test_manifest_uri")),
        "checkpoint_uri": _text(document.get("checkpoint_uri")),
        "prediction_source": _text(document.get("prediction_source")),
        "fused_from": document.get("fused_from"),
        "boxes": kept,
    }
    with _LOCK:
        _PREDICTION_CACHE[uri] = loaded
    return loaded


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - overlap
    return overlap / union if union > 0 else 0.0


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


#: 일치율을 재는 방법이 바뀌면 올립니다. **저장해 둔 값의 자리가 함께 바뀌어야**
#: 옛 공식으로 잰 값이 새 공식인 척 되살아나지 않습니다. 값만 보고는 어느 공식으로
#: 쟀는지 알 수 없으므로 key로 가릅니다.
_AGREEMENT_FORMULA = "dice-v1"


def _stored_pair_path(prefix: str, key: tuple[str, str]) -> str:
    digest = hashlib.sha256(
        " ".join((_AGREEMENT_FORMULA, *key)).encode("utf-8")
    ).hexdigest()
    return f"{prefix}/{digest}.json"


def _load_stored_pair(key: tuple[str, str]) -> dict[str, float] | None:
    config, prefix = _scope()
    try:
        storage = create_storage(config)
        path = _stored_pair_path(prefix, key)
        if not storage.exists(path):
            return None
        document = storage.read_json(path)
    except (StorageError, ValueError):
        # 저장해 둔 값을 못 읽는 것은 다시 재면 되는 일이라 실패로 만들지 않습니다.
        return None
    if not isinstance(document, Mapping):
        return None
    agreement = document.get("agreement")
    box_iou = document.get("box_iou")
    if not isinstance(agreement, (int, float)) or not isinstance(box_iou, (int, float)):
        return None
    return {"agreement": float(agreement), "box_iou": float(box_iou)}


def _store_pair(key: tuple[str, str], value: Mapping[str, float]) -> None:
    config, prefix = _scope()
    try:
        storage = create_storage(config)
        storage.write_json(
            _stored_pair_path(prefix, key),
            {"runs": list(key), **dict(value)},
            overwrite=True,
        )
    except StorageError as error:
        # 저장 실패로 진단까지 막지 않습니다. 다음에 다시 재면 됩니다.
        raise WebStateError(f"다양성 결과를 저장하지 못했습니다({type(error).__name__}).") from error


def _compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    """두 실행이 얼마나 같은 예측을 하는지입니다.

    **쌍 일치**는 (이미지, class)가 겹치는 비율이고, **상자 IoU**는 겹친 쌍의 상자가
    얼마나 포개지는지입니다. 둘 다 높으면 합칠 것이 없습니다 — 서로 다르게 틀려야
    앙상블이 이득을 냅니다.

    한쪽 수로 나누면 어느 쪽을 먼저 골랐는지에 따라 답이 달라집니다 — 94개가 100개에
    통째로 들어 있으면 순서에 따라 100%(경고)와 94%(통과)로 갈립니다. 저장해 두는 값의
    key는 순서를 가리지 않으므로, 먼저 잰 쪽이 영원히 남아 같은 조합이 다르게
    판정됩니다.

    그래서 **Dice**를 씁니다: `2|∩| / (|A| + |B|)`. 대칭이면서, 두 실행의 예측 수가
    같을 때 예전 값과 **정확히 같은 눈금**을 냅니다. 합집합으로 나누면(Jaccard) 대칭은
    되지만 눈금이 바뀌어, 96.9%로 경고하던 조합이 94.0%가 되어 임계값 0.95를 조용히
    빠져나갑니다. 제출은 이미지마다 같은 수(4개)를 남기므로 크기가 대개 같습니다.
    """

    left_boxes: dict[Any, tuple[float, ...]] = left["boxes"]
    right_boxes: dict[Any, tuple[float, ...]] = right["boxes"]
    total = len(left_boxes) + len(right_boxes)
    if not total:
        return {"agreement": 0.0, "box_iou": 0.0}
    shared = set(left_boxes) & set(right_boxes)
    ious = [_iou(left_boxes[key], right_boxes[key]) for key in shared]
    return {
        "agreement": 2 * len(shared) / total,
        "box_iou": statistics.median(ious) if ious else 0.0,
    }


def _pair_diversity(left: Mapping[str, Any], right: Mapping[str, Any], *, run_ids: tuple[str, str]) -> dict[str, float]:
    key = _pair_key(*run_ids)
    with _LOCK:
        cached = _PAIR_CACHE.get(key)
    if cached is not None:
        return cached
    stored = _load_stored_pair(key)
    if stored is not None:
        with _LOCK:
            _PAIR_CACHE[key] = stored
        return stored
    value = _compare(left, right)
    with _LOCK:
        _PAIR_CACHE[key] = value
    try:
        _store_pair(key, value)
    except WebStateError:
        pass
    return value


# ----------------------------------------------------------------------- 진단

def _check(identifier: str, level: str, title: str, detail: str) -> dict[str, str]:
    return {"id": identifier, "level": level, "title": title, "detail": detail}


def _test_set_check(loaded: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """같은 시험지를 본 예측인가.

    **manifest URI 전체**를 견줍니다. 폴더 이름만 보면 bucket이나 앞 경로가 달라도
    같은 판으로 읽혀 "모두 같은 시험지"라는 잘못된 통과가 나오고, 실제 거부는
    evaluate에 가서야 일어납니다. 폴더 이름은 사람이 읽을 이름으로만 씁니다.
    """

    uris = {item["test_manifest_uri"] for item in loaded if item["test_manifest_uri"]}
    labels = {_dataset_label(item["test_manifest_uri"] or "") for item in loaded}
    labels.discard(None)
    if len(uris) <= 1:
        return _check("test_set", "ok", "같은 시험지", "모두 같은 test manifest를 봤습니다.")
    return _check(
        "test_set",
        "warn",
        "시험지가 다릅니다",
        f"{', '.join(sorted(str(label) for label in labels)) or '경로가 다릅니다'} — dataset 판마다 같은 사진을 "
        "자기 prefix에 복사해 두기 때문에, 내용이 같아도 위치가 달라 evaluate가 거부할 수 "
        "있습니다. 같은 사진임을 확인했다면 fusion_allow_copied_images를 켜고 실행하세요.",
    )


def _reuse_check(loaded: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """이미 합친 결과이거나 같은 실행이 두 번 들어왔는가."""

    fused = [
        item["uri"]
        for item in loaded
        if item["prediction_source"] == "fusion" or item["fused_from"] is not None
    ]
    if fused:
        return _check(
            "reuse",
            "warn",
            "합친 결과가 들어 있습니다",
            f"{len(fused)}개가 이미 융합 결과입니다. evaluate는 이것을 거부합니다 — "
            "원본 예측들을 한 번에 고르세요.",
        )
    checkpoints = [item["checkpoint_uri"] for item in loaded if item["checkpoint_uri"]]
    if len(set(checkpoints)) != len(checkpoints):
        return _check(
            "reuse", "warn", "같은 실행이 두 번 있습니다",
            "같은 checkpoint의 예측이 둘 이상입니다. 한 실행이 두 표를 갖게 되어 "
            "확신도가 부풀려집니다.",
        )
    return _check("reuse", "ok", "서로 다른 실행", "합친 결과도, 중복도 없습니다.")


def _diversity_check(
    loaded: Sequence[Mapping[str, Any]], run_ids: Sequence[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    """서로 다르게 예측하는가 — 이 화면의 핵심입니다."""

    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(loaded):
        for offset in range(index + 1, len(loaded)):
            value = _pair_diversity(
                left, loaded[offset], run_ids=(run_ids[index], run_ids[offset])
            )
            pairs.append({"runs": [run_ids[index], run_ids[offset]], **value})
    if not pairs:
        return (
            _check("diversity", "warn", "잴 수 없음", "둘 이상을 골라야 잽니다."),
            {"pairs": []},
        )
    agreement = statistics.mean(item["agreement"] for item in pairs)
    box_iou = statistics.mean(item["box_iou"] for item in pairs)
    summary = {"pairs": pairs, "agreement": agreement, "box_iou": box_iou}
    if agreement >= _AGREEMENT_WARN:
        return (
            _check(
                "diversity",
                "warn",
                f"거의 같은 예측입니다 (일치 {agreement:.1%})",
                f"상자도 IoU {box_iou:.3f}로 포개집니다. 이 정도로 붙어 있으면 합칠 것이 "
                "거의 없습니다 — 실제로 97.7% 조합의 이득이 +0.002였습니다. 다양성은 "
                "아키텍처가 아니라 학습 데이터에서 나옵니다.",
            ),
            summary,
        )
    return (
        _check(
            "diversity",
            "ok",
            f"다르게 예측합니다 (일치 {agreement:.1%})",
            f"상자 IoU 중앙 {box_iou:.3f}. 합칠 여지가 있습니다.",
        ),
        summary,
    )


def _dilution_check(selected: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    """약한 실행이 결과를 평균 쪽으로 끌어내리는가."""

    scored = [item for item in selected if isinstance(item.get("kaggle_score"), (int, float))]
    if len(scored) < 2:
        return (
            _check(
                "dilution", "warn", "점수를 모르는 실행이 있습니다",
                "Kaggle 점수가 없는 실행은 pool 품질을 가늠할 수 없습니다.",
            ),
            {},
        )
    values = [float(item["kaggle_score"]) for item in scored]
    best, mean = max(values), statistics.mean(values)
    # 일곱 개 융합이 평균에서 최고까지의 82% 지점에 떨어졌습니다. 그 관측을 그대로
    # 기대 구간으로 보여 줍니다.
    expected = {"floor": mean, "ceiling": best, "observed_ratio": 0.82}
    weak = [item["run_id"] for item in scored if best - float(item["kaggle_score"]) > _WEAK_MARGIN]
    if weak:
        return (
            _check(
                "dilution",
                "warn",
                f"약한 실행 {len(weak)}개",
                f"{', '.join(weak)} — 최고보다 {_WEAK_MARGIN:.0%}p 넘게 낮습니다. 모든 실행이 "
                f"같은 한 표를 받으므로 결과가 pool 평균({mean:.5f}) 쪽으로 끌려갑니다. "
                f"실제로 이것 때문에 융합이 단독 최고보다 낮게 나온 적이 있습니다.",
            ),
            expected,
        )
    return (
        _check(
            "dilution", "ok", "품질이 고릅니다",
            f"최고 {best:.5f}, 평균 {mean:.5f}. 크게 뒤처지는 실행이 없습니다.",
        ),
        expected,
    )


def _pool_size_check(count: int) -> dict[str, str]:
    if count < 2:
        return _check("pool_size", "warn", "둘 이상 필요", "합칠 것이 없습니다.")
    if count >= _CLASS_DROP_FROM:
        return _check(
            "pool_size",
            "warn",
            f"{count}개는 많을 수 있습니다",
            "합치는 수가 늘수록 고유 class가 줄었습니다(84 → 82 → 81 → 78 → 77). 확신도가 "
            "동의한 실행 수로 깎여, 드물게 잡히는 class가 이미지당 상한 밖으로 밀립니다. "
            "이 저장소에서는 3개가 가장 좋았습니다.",
        )
    return _check("pool_size", "ok", f"{count}개", "합치기 좋은 수입니다.")


def _readiness_check(pending: Sequence[str]) -> dict[str, str]:
    """예측이 아직 없는 실행은 추론이 먼저입니다 — GPU를 쓰는 유일한 단계입니다."""

    if not pending:
        return _check("readiness", "ok", "바로 합칠 수 있음", "고른 실행 모두 예측이 있습니다.")
    return _check(
        "readiness",
        "warn",
        f"{len(pending)}개는 추론이 먼저입니다",
        f"{', '.join(pending)} — 체크포인트는 있지만 test 예측이 없습니다. 합치기를 누르면 "
        f"먼저 만듭니다. 체크포인트당 약 {_HARVEST_MINUTES}분(GPU)이라 "
        f"약 {len(pending) * _HARVEST_MINUTES}분 걸리고, 학습이 도는 중이면 GPU를 나눠 씁니다. "
        "다양성은 예측이 생긴 뒤에야 잴 수 있습니다.",
    )


def diagnose(run_ids: Sequence[str]) -> dict[str, Any]:
    """고른 조합을 합치기 전에 알 수 있는 것을 전부 재서 돌려줍니다.

    예측이 아직 없는 실행은 다양성을 잴 수 없습니다. 그렇다고 진단을 통째로 접지
    않습니다 — 나머지 검사는 그대로 쓸모가 있고, 무엇을 아직 모르는지 말해 주는 것이
    아무 말도 안 하는 것보다 낫습니다.
    """

    wanted = [str(item).strip() for item in run_ids if str(item).strip()]
    if not wanted:
        raise WebError("합칠 실행을 하나도 고르지 않았습니다.")
    by_run = {item["run_id"]: item for item in list_candidates()}
    unknown = [run_id for run_id in wanted if run_id not in by_run]
    if unknown:
        raise WebError(f"기록에 없는 실행입니다: {', '.join(unknown)}")

    selected = [by_run[run_id] for run_id in wanted]
    ready = [item for item in selected if item["ready"]]
    pending = [item["run_id"] for item in selected if not item["ready"]]
    loaded = [_read_predictions(item["test_predictions_uri"]) for item in ready]
    ready_ids = [item["run_id"] for item in ready]

    checks = [_pool_size_check(len(wanted)), _readiness_check(pending)]
    diversity: dict[str, Any] = {"pairs": []}
    if len(loaded) >= 2:
        diversity_check, diversity = _diversity_check(loaded, ready_ids)
        checks += [_test_set_check(loaded), _reuse_check(loaded), diversity_check]
    dilution_check, expected = _dilution_check(selected)
    checks.append(dilution_check)
    return {
        "run_ids": wanted,
        "checks": checks,
        "diversity": diversity,
        "expected": expected,
        "blocking": False,
    }


# --------------------------------------------------------------------- 실행 config

def check_selection(
    run_ids: Sequence[str],
    *,
    run_id: str,
    allow_copied_images: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """추론을 걸기 **전에** 거절할 것을 거절합니다.

    합칠 수 있는지는 융합 단계에서도 확인하지만, 그때는 이미 체크포인트마다 9분씩
    GPU를 쓴 뒤입니다. 알 수 있는 것을 늦게 알리면 그 시간이 통째로 버려집니다.

    이름도 여기서 봅니다. 길이만 재면 local backend에서 `../train/name` 같은 값이
    `artifacts/ensemble/` 밖의 **다른 pipeline 자리**로 풀립니다.
    """

    wanted = [str(item).strip() for item in run_ids if str(item).strip()]
    if len(wanted) < 2:
        raise WebError("합칠 예측이 둘 이상 필요합니다.")
    if len(set(wanted)) != len(wanted):
        raise WebError("같은 실행을 두 번 골랐습니다.")
    name = str(run_id).strip()
    if not RUN_ID_PATTERN.fullmatch(name):
        raise WebError(
            "결과 이름은 영문·숫자로 시작하고 영문·숫자·`.`·`_`·`-`만 쓸 수 있습니다: "
            f"{run_id!r}"
        )
    by_run = {item["run_id"]: item for item in list_candidates()}
    unknown = [item for item in wanted if item not in by_run]
    if unknown:
        raise WebError(f"기록에 없는 실행입니다: {', '.join(unknown)}")
    selected = [by_run[item] for item in wanted]

    # **마지막 실행의 결함도 첫 추론 전에 찾습니다.** 순서대로 확인하면 세 번째가
    # 입력을 빠뜨린 것을 앞의 둘을 18분 추론한 뒤에 알게 됩니다.
    for item in selected:
        missing = [
            key
            for key in ("validation_manifest_uri", "test_manifest_uri", "class_map_uri")
            if not (item.get("data_inputs") or {}).get(key)
        ]
        if missing:
            raise WebError(
                f"{item['run_id']}의 기록에 {', '.join(missing)}가 없어 합칠 수 없습니다."
            )

    # **evaluate가 읽는 값으로 견줍니다.** 기록과 예측 파일이 다른 값을 담을 수 있고,
    # 실제로 거절 판정을 하는 쪽은 파일입니다. 예측이 있으면 파일이 선언한 것을,
    # 아직 없으면 이제 만들 때 쓸 기록의 것을 봅니다.
    declared: list[tuple[str, str, str]] = []
    for item in selected:
        uri = item.get("test_predictions_uri")
        if uri:
            document = _read_predictions(str(uri))
            if document["prediction_source"] == "fusion" or document["fused_from"] is not None:
                raise WebError(
                    f"{item['run_id']}은 이미 합친 결과입니다. 합친 것을 다시 합칠 수는 "
                    "없으니 원본 실행들을 고르세요."
                )
            # 예측 문서가 **스스로** 무엇의 증거인지 말해야 합니다.
            if not document["checkpoint_uri"]:
                raise WebError(
                    f"{item['run_id']}의 예측 파일이 어느 checkpoint의 증거인지 적고 "
                    "있지 않아 합칠 수 없습니다."
                )
            if not document["test_manifest_uri"]:
                raise WebError(
                    f"{item['run_id']}의 예측 파일이 어느 시험지를 본 것인지 적고 "
                    "있지 않아 합칠 수 없습니다."
                )
            declared.append(
                (item["run_id"], str(document["checkpoint_uri"]), str(document["test_manifest_uri"]))
            )
        else:
            declared.append(
                (
                    item["run_id"],
                    str(item.get("checkpoint_uri") or ""),
                    str((item.get("data_inputs") or {})["test_manifest_uri"]),
                )
            )

    # 이름이 달라도 같은 checkpoint면 한 실행이 두 표를 갖습니다. 표기가 아니라
    # **저장 신원**으로 봅니다 — `s3://bucket/a.pt`와 `a.pt`는 같은 파일입니다.
    seen: dict[str, str] = {}
    for name_of, checkpoint, _ in declared:
        identity = _checkpoint_identity(checkpoint)
        if identity in seen:
            raise WebError(
                f"{seen[identity]}와 {name_of}가 같은 checkpoint를 가리킵니다. "
                "한 실행이 두 표를 갖게 되어 확신도가 부풀려집니다."
            )
        seen[identity] = name_of

    # 시험지가 다르면 evaluate가 거부합니다. 사람이 같은 사진임을 확인했다고 말한
    # 경우에만 통과시킵니다 — 그 확인 없이 추론부터 돌리면 전부 버려집니다.
    manifests = {manifest for _, _, manifest in declared}
    if len(manifests) > 1 and not allow_copied_images:
        raise WebError(
            "고른 실행들이 서로 다른 test manifest를 봅니다: "
            f"{', '.join(sorted(manifests))} — 사진이 같은데 위치만 다른 것을 확인했다면 "
            "'사진이 같은데 위치만 다른 것을 확인했습니다'를 켜고 다시 실행하세요."
        )

    # 같은 이름의 결과가 이미 있으면 evaluate가 덮어쓰지 않고 멈춥니다. 그 판단도
    # 추론 뒤가 아니라 지금 합니다.
    if not overwrite:
        existing = _existing_output(name)
        if existing is not None:
            raise WebError(
                f"{name}이라는 결과가 이미 있습니다: {existing} — 다른 이름을 쓰거나 "
                "덮어쓰기를 켜세요."
            )
    return selected


def _existing_output(run_id: str) -> str | None:
    """같은 이름의 융합 결과가 이미 있는지입니다.

    **확인하지 못하면 추측하지 않고 그 자리에서 알립니다.** 없다고 넘기면 추론을 다
    돌린 뒤 출력 충돌로 실패하고, 있다고 넘기면 멀쩡한 이름이 막힙니다. 게다가 저장소를
    못 읽는 상태면 뒤 단계도 안전하게 끝나지 않습니다 — 원인을 지금 말하는 편이 낫습니다.
    """

    uri = _submission_uri(run_id)
    config, _ = _scope()
    try:
        return uri if create_storage(config).exists(uri) else None
    except StorageError as error:
        raise WebError(
            f"{uri}가 이미 있는지 확인하지 못했습니다({type(error).__name__}). "
            "저장소에 닿지 못하면 합치기도 끝까지 갈 수 없습니다."
        ) from error


def pending_runs(run_ids: Sequence[str]) -> list[dict[str, Any]]:
    """고른 것 중 예측이 아직 없어 추론이 필요한 실행입니다. 고른 순서를 지킵니다."""

    by_run = {item["run_id"]: item for item in list_candidates()}
    return [
        by_run[str(item).strip()]
        for item in run_ids
        if str(item).strip() in by_run and not by_run[str(item).strip()]["ready"]
    ]


def build_harvest_config(candidate: Mapping[str, Any], *, device: str | None = None) -> dict[str, Any]:
    """체크포인트 하나로 test 예측을 만드는 evaluate config입니다.

    남의 artifact를 건드리지 않습니다. 결과는 web이 소유한 자리에 두고, 그 실행의
    `experiments/completed/.../evaluate/`는 그대로 둡니다.

    제출이 남기는 4개가 아니라 스무 개를 저장합니다. 합칠 때 고를 것이 있어야 하기
    때문이고, 그래서 이 실행이 만드는 CSV는 제출할 수 없습니다.
    """

    from .evaluation import resolve_device

    run_id = str(candidate.get("run_id") or "").strip()
    checkpoint = str(candidate.get("checkpoint_uri") or "").strip()
    if not run_id or not checkpoint:
        raise WebError("체크포인트가 없는 실행은 예측을 만들 수 없습니다.")
    data_inputs = {key: value for key, value in (candidate.get("data_inputs") or {}).items() if value}
    for key in ("validation_manifest_uri", "test_manifest_uri", "class_map_uri"):
        if key not in data_inputs:
            raise WebError(f"{run_id}에 {key}가 없어 예측을 만들 수 없습니다.")

    # local backend에서도 만들 수 있어야 합니다. S3만 지원하면 목록에는 후보가
    # 보이는데 누르면 언제나 실패합니다.
    storage = _storage_config()
    output_dir = f"{_harvest_root()}/{run_id}"

    return {
        "project": {"name": "pill-object-detection"},
        "execution": {"mode": "real"},
        "storage": storage,
        "evaluate": {
            "run_id": run_id,
            "output_dir": output_dir,
            # 제출할 수 없는 CSV라 submissions/ 아래에 두지 않습니다.
            "submission_uri": f"{output_dir}/candidates.csv",
            "checkpoint_uri": checkpoint,
            "max_detections_per_image": _HARVEST_DETECTIONS,
            "score_threshold": 0.0,
            "device": resolve_device(device),
            "overwrite": False,
            **data_inputs,
        },
    }


def build_fusion_config(
    run_ids: Sequence[str],
    *,
    run_id: str,
    allow_copied_images: bool = False,
    max_detections_per_image: int = _SUBMISSION_LIMIT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """고른 실행들을 합치는 evaluate config를 만듭니다.

    **데이터 입력은 고른 실행이 대던 것을 그대로 씁니다.** 여기서 새로 정하면 학습이
    본 것과 다른 시험지를 채점하게 됩니다.

    validation 지표는 융합이 만들지 않습니다. 가장 센 실행의 예측 파일을 그대로
    넘겨(`predictions_input_uri`) 검증 추론을 건너뜁니다 — 융합은 test 쪽만 바꾸므로
    검증을 다시 도는 것은 GPU만 쓰고 아무것도 알려 주지 않습니다.
    """

    wanted = [str(item).strip() for item in run_ids if str(item).strip()]
    if len(wanted) < 2:
        raise WebError("합칠 예측이 둘 이상 필요합니다.")
    name = str(run_id).strip()
    if not name:
        raise WebError("결과에 붙일 이름이 필요합니다.")

    by_run = {item["run_id"]: item for item in list_candidates()}
    unknown = [item for item in wanted if item not in by_run]
    if unknown:
        raise WebError(f"기록에 없는 실행입니다: {', '.join(unknown)}")
    selected = [by_run[item] for item in wanted]
    pending = [item["run_id"] for item in selected if not item["ready"]]
    if pending:
        # 추론이 먼저입니다. 실행기가 그것을 끝낸 뒤 이 함수를 다시 부릅니다.
        raise WebError(f"아직 예측이 없는 실행입니다: {', '.join(pending)}")

    # 점수가 가장 높은 실행의 입력을 기준으로 삼습니다. 어느 것을 골라도 같아야 하고,
    # 다르면 진단이 이미 경고했습니다.
    anchor = max(
        selected,
        key=lambda item: item["kaggle_score"] if isinstance(item["kaggle_score"], (int, float)) else -1.0,
    )
    data_inputs = {key: value for key, value in anchor["data_inputs"].items() if value}
    for key in ("validation_manifest_uri", "test_manifest_uri", "class_map_uri"):
        if key not in data_inputs:
            raise WebError(f"{anchor['run_id']}에 {key}가 없어 융합 config를 만들 수 없습니다.")

    storage = _storage_config()
    output_dir = (
        f"{_storage_root()}/experiments/ensemble/{name}"
        if _uses_s3()
        else f"artifacts/ensemble/{name}"
    )
    submission_uri = _submission_uri(name)

    settings: dict[str, Any] = {
        "run_id": name,
        "output_dir": output_dir,
        "submission_uri": submission_uri,
        "test_predictions_input_uris": [item["test_predictions_uri"] for item in selected],
        "max_detections_per_image": int(max_detections_per_image),
        "score_threshold": 0.0,
        "overwrite": bool(overwrite),
        # 융합은 CPU로 몇 분입니다. GPU를 잡아 학습을 방해할 이유가 없습니다.
        "device": "cpu",
        **data_inputs,
    }
    if allow_copied_images:
        settings["fusion_allow_copied_images"] = True
    if anchor["predictions_uri"]:
        settings["predictions_input_uri"] = anchor["predictions_uri"]

    return {
        "project": {"name": "pill-object-detection"},
        "execution": {"mode": "real"},
        "storage": storage,
        "evaluate": settings,
    }
