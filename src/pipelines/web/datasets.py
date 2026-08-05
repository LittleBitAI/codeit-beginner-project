"""전처리 결과 폴더 하나에서 train이 요구하는 artifact 4개를 찾아냅니다.

이 저장소에는 아직 data artifact의 표준 위치도, 정해진 file 이름도 없습니다
(``contracts/README.md``는 pipeline별 산출물 schema를 각 담당 directory에서 정한다고
둡니다). 그래서 이름 규약에 기대지 않고 **내용을 보고** 무엇이 무엇인지 판단합니다.

| artifact | 알아보는 방법 |
| --- | --- |
| manifest | 최상위에 ``images``와 ``annotations``가 있는 COCO 형식 |
| class map | 이름과 번호만 담긴 object (``{"pill": 1}`` 또는 ``{"1": "pill"}``) |
| dataset summary | 위 둘이 아닌 나머지 JSON object |

학습용과 검증용 manifest는 형태가 같으므로 file 이름으로 구분합니다. 판단 근거를
그대로 돌려주므로 사용자가 화면에서 확인하고 고칠 수 있습니다.

이 module은 사용자가 직접 지정한 directory를 **읽기만** 합니다. 어떤 파일도 만들거나
바꾸지 않습니다.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .errors import FieldError, WebPathError, WebValidationError
from .jobs import runner
from .masking import sanitize_line

# REPOSITORY_ROOT를 값으로 가져오면 안 됩니다. 그러면 import 시점에 고정돼 test가
# 저장소 root를 바꿔도 실제 저장소에 파일을 쓰게 됩니다. 항상 함수로 물어봅니다.
from .paths import (
    config_dir,
    repository_root,
    resolve_within_repo,
    to_repo_relative_posix,
    web_state_dir,
)
from .train_config import DATA_ARTIFACT_KEYS


__all__ = [
    "build_data_config",
    "classify_document",
    "clear_selection",
    "inspect_directory",
    "load_selection",
    "save_selection",
    "verify_with_pipeline",
]


# 폴더 하나에서 살펴볼 JSON 파일 수의 상한입니다. 실수로 거대한 directory를 지정해도
# 화면이 멈추지 않게 합니다.
MAX_FILES = 200

# 전체를 parse할 최대 크기. manifest는 이미지가 많아지면 수십 MB가 될 수 있는데,
# 무엇인지 알아보는 데는 앞부분만 있으면 충분합니다.
FULL_PARSE_LIMIT = 2 * 1024 * 1024
PEEK_BYTES = 8192

_INTEGER_TEXT = re.compile(r"^\d+$")

# 학습용/검증용을 가르는 file 이름 힌트입니다. 앞쪽이 먼저 매칭됩니다.
_VALIDATION_HINTS = ("validation", "valid", "val", "dev", "eval")
_TRAIN_HINTS = ("train", "training", "trn")

# 클래스 맵과 데이터셋 요약은 둘 다 "이름: 숫자" 형태일 수 있습니다.
# 예: {"pill": 1} 과 {"train_images": 1, "validation_images": 1}
# 내용만으로는 가를 수 없어 file 이름을 함께 봅니다.
_CLASS_MAP_HINTS = ("class", "label", "categor")
_SUMMARY_HINTS = ("summary", "stat", "meta", "info", "overview", "dataset")

_LABELS = {
    "train_manifest_uri": "학습 manifest",
    "validation_manifest_uri": "검증 manifest",
    "class_map_uri": "클래스 맵",
    "dataset_summary_uri": "데이터셋 요약",
}


def _looks_like_class_map(document: Any) -> bool:
    """``load_class_map``(train/dataset.py:73)이 받아들이는 두 형태를 확인합니다."""

    if not isinstance(document, dict) or not document:
        return False
    values = list(document.values())
    # {"pill": 1} 형태
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return True
    # {"7": "pill"} 형태
    if all(isinstance(value, str) and value.strip() for value in values):
        return all(_INTEGER_TEXT.fullmatch(str(key).strip()) for key in document)
    return False


def _looks_like_manifest(document: Any) -> bool:
    return (
        isinstance(document, dict)
        and isinstance(document.get("images"), list)
        and isinstance(document.get("annotations"), list)
    )


def classify_document(document: Any) -> str:
    """읽어 들인 JSON이 어떤 artifact인지 알려 줍니다.

    ``manifest`` / ``class_map`` / ``summary`` / ``unknown`` 중 하나입니다.
    """

    if _looks_like_manifest(document):
        return "manifest"
    if _looks_like_class_map(document):
        return "class_map"
    if isinstance(document, dict) and document:
        return "summary"
    return "unknown"


# 큰 manifest는 "images" 배열만으로도 앞부분이 8KB를 넘깁니다. 그래서 두 key가 모두
# 앞에 있기를 기대하면 안 되고, 배열이 열리는 모양 하나만 봐도 충분합니다.
# ``{"train_images": 1}`` 같은 요약은 값이 배열이 아니라서 걸리지 않습니다.
_MANIFEST_HEAD = re.compile(r'"(?:images|annotations)"\s*:\s*\[')


def _peek_kind(path: Path) -> str | None:
    """큰 파일은 앞부분만 보고 manifest인지 판단합니다."""

    try:
        with path.open("rb") as stream:
            head = stream.read(PEEK_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return None
    if _MANIFEST_HEAD.search(head):
        return "manifest"
    return None


def _read_kind(path: Path) -> tuple[str, str | None]:
    """(종류, 문제) 를 돌려줍니다. 읽지 못해도 예외를 던지지 않습니다."""

    try:
        size = path.stat().st_size
    except OSError as error:
        return "unknown", f"파일 정보를 읽지 못했습니다({type(error).__name__})."

    if size > FULL_PARSE_LIMIT:
        peeked = _peek_kind(path)
        if peeked:
            return peeked, None
        return "unknown", "파일이 너무 커서 종류를 확인하지 못했습니다."

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        return "unknown", f"파일을 읽지 못했습니다({type(error).__name__})."
    except json.JSONDecodeError:
        return "unknown", "올바른 JSON이 아닙니다."

    return classify_document(document), None


def _split_role(name: str) -> str | None:
    """file 이름으로 학습용/검증용을 가릅니다."""

    lowered = name.lower()
    # 검증을 먼저 봅니다. "validation"에는 "val"이 들어 있어 순서가 중요합니다.
    for hint in _VALIDATION_HINTS:
        if hint in lowered:
            return "validation"
    for hint in _TRAIN_HINTS:
        if hint in lowered:
            return "train"
    return None


def _has_hint(entry: dict[str, Any], hints: tuple[str, ...]) -> bool:
    lowered = entry["name"].lower()
    return any(hint in lowered for hint in hints)


def _pick_class_map_and_summary(
    class_shaped: list[dict[str, Any]], objects: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """클래스 맵과 데이터셋 요약을 가릅니다.

    ``{"pill": 1}``과 ``{"train_images": 1}``은 내용 형태가 같습니다. 그래서
    형태가 같은 후보가 여럿이면 file 이름으로 가르고, 그래도 모호하면 사실대로 알립니다.
    """

    problems: list[str] = []
    named_class = [entry for entry in class_shaped if _has_hint(entry, _CLASS_MAP_HINTS)]
    named_summary = [entry for entry in class_shaped if _has_hint(entry, _SUMMARY_HINTS)]

    if len(named_class) == 1:
        class_map = named_class[0]
    elif len(class_shaped) == 1 and not named_summary:
        # 후보가 하나뿐이고 요약처럼 보이지도 않으면 클래스 맵으로 봅니다.
        class_map = class_shaped[0]
    else:
        class_map = None
        if len(named_class) > 1:
            problems.append("클래스 맵으로 보이는 파일이 여러 개입니다.")
        elif class_shaped:
            problems.append(
                "클래스 맵과 데이터셋 요약이 형태가 같아 가릴 수 없습니다. "
                "file 이름에 class 또는 summary를 넣어 주세요."
            )

    # 클래스 맵으로 쓰지 않은 나머지는 요약 후보가 됩니다.
    leftovers = [entry for entry in class_shaped if entry is not class_map]
    candidates = objects + leftovers
    summary = None
    for entry in candidates:
        if _has_hint(entry, _SUMMARY_HINTS):
            summary = entry
            break
    if summary is None and candidates:
        summary = candidates[0]
    return class_map, summary, problems


def _pick_manifests(
    manifests: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    problems: list[str] = []
    train_side = [entry for entry in manifests if _split_role(entry["name"]) == "train"]
    validation_side = [entry for entry in manifests if _split_role(entry["name"]) == "validation"]
    unlabelled = [entry for entry in manifests if _split_role(entry["name"]) is None]

    if not train_side and not validation_side and len(unlabelled) == 2:
        # 이름에 힌트가 없으면 순서로만 가를 수 없습니다. 사용자가 고르게 둡니다.
        problems.append(
            "manifest 2개를 찾았지만 file 이름만으로는 학습용과 검증용을 가릴 수 없습니다."
        )
        return None, None, problems

    if len(train_side) > 1:
        problems.append("학습용으로 보이는 manifest가 여러 개입니다.")
    if len(validation_side) > 1:
        problems.append("검증용으로 보이는 manifest가 여러 개입니다.")

    train_entry = train_side[0] if len(train_side) == 1 else None
    validation_entry = validation_side[0] if len(validation_side) == 1 else None
    return train_entry, validation_entry, problems


def inspect_directory(directory: object) -> dict[str, Any]:
    """위치 하나를 살펴 artifact 4개를 찾습니다. 읽기만 합니다.

    ``s3://bucket/prefix/`` 를 주면 S3에서, 그 밖에는 저장소 안 폴더에서 찾습니다.
    이미 S3에 준비돼 있는 산출물을 그대로 쓸 수 있어야 하기 때문입니다.
    """

    if isinstance(directory, str) and directory.strip().lower().startswith("s3://"):
        return inspect_s3_prefix(directory)

    try:
        resolved = resolve_within_repo(directory, label="전처리 폴더")
    except WebPathError as error:
        raise WebValidationError([FieldError("directory", str(error))]) from error

    if not resolved.is_dir():
        raise WebValidationError(
            [FieldError("directory", "그 위치에 폴더가 없습니다. 경로를 확인해 주세요.")]
        )

    files = sorted(
        (path for path in resolved.iterdir() if path.is_file() and path.suffix.lower() == ".json"),
        key=lambda path: path.name.lower(),
    )[:MAX_FILES]

    examined: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    class_maps: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for path in files:
        kind, problem = _read_kind(path)
        entry = {
            "name": path.name,
            "uri": to_repo_relative_posix(path),
            "kind": kind,
            "problem": problem,
        }
        examined.append(entry)
        if kind == "manifest":
            manifests.append(entry)
        elif kind == "class_map":
            class_maps.append(entry)
        elif kind == "summary":
            summaries.append(entry)

    return _assemble(to_repo_relative_posix(resolved), examined, empty=not files)


def _assemble(location: str, examined: list[dict[str, Any]], *, empty: bool) -> dict[str, Any]:
    """살펴본 파일 목록에서 artifact 4개를 골라 결과를 만듭니다."""

    manifests = [entry for entry in examined if entry["kind"] == "manifest"]
    class_maps = [entry for entry in examined if entry["kind"] == "class_map"]
    summaries = [entry for entry in examined if entry["kind"] == "summary"]

    problems: list[str] = []
    train_entry, validation_entry, manifest_problems = _pick_manifests(manifests)
    problems.extend(manifest_problems)

    class_map_entry, summary_entry, role_problems = _pick_class_map_and_summary(
        class_maps, summaries
    )
    problems.extend(role_problems)

    resolved_uris: dict[str, str] = {}
    matched: dict[str, dict[str, Any] | None] = {
        "train_manifest_uri": train_entry,
        "validation_manifest_uri": validation_entry,
        "class_map_uri": class_map_entry,
        "dataset_summary_uri": summary_entry,
    }
    missing: list[str] = []
    for key in DATA_ARTIFACT_KEYS:
        entry = matched[key]
        if entry is None:
            missing.append(key)
        else:
            resolved_uris[key] = entry["uri"]

    if empty:
        problems.append("이 위치에 JSON 파일이 없습니다.")

    return {
        "directory": location,
        "complete": not missing,
        "data": resolved_uris,
        "matched": {
            key: (
                None
                if matched[key] is None
                else {"name": matched[key]["name"], "uri": matched[key]["uri"]}
            )
            for key in DATA_ARTIFACT_KEYS
        },
        "labels": dict(_LABELS),
        "missing": missing,
        "problems": problems,
        "examined": examined,
    }


# S3 prefix 하나에서 읽어 볼 JSON 개수 상한입니다. 잘못된 prefix를 넣어도 요청이
# 수백 개로 늘어나지 않게 합니다.
MAX_S3_FILES = 40


def inspect_s3_prefix(location: str) -> dict[str, Any]:
    """``s3://bucket/prefix/`` 아래에서 artifact 4개를 찾습니다. 읽기만 합니다."""

    prefix = location.strip()
    if not prefix.lower().startswith("s3://"):
        raise WebValidationError([FieldError("directory", "s3:// 로 시작해야 합니다.")])
    split = urlsplit(prefix)
    if not split.netloc:
        raise WebValidationError(
            [FieldError("directory", "s3://bucket/prefix/ 형식이어야 합니다.")]
        )
    if split.query or split.fragment:
        raise WebValidationError(
            [FieldError("directory", "s3 위치에 query나 fragment를 쓸 수 없습니다.")]
        )
    if not prefix.endswith("/"):
        prefix += "/"

    from src.common import StorageError, create_storage

    try:
        storage = create_storage({"storage": {"backend": "s3", "s3": {"prefix": ""}}})
        entries_found = [str(item) for item in storage.list(prefix)]
    except StorageError as error:
        # 원문에는 bucket 경로나 backend 오류가 섞일 수 있어 type만 전합니다.
        raise WebValidationError(
            [
                FieldError(
                    "directory",
                    f"S3 위치를 읽지 못했습니다({type(error).__name__}). "
                    "bucket 이름과 접근 권한을 확인해 주세요.",
                )
            ]
        ) from error

    json_uris = sorted(uri for uri in entries_found if uri.lower().endswith(".json"))[
        :MAX_S3_FILES
    ]
    examined: list[dict[str, Any]] = []
    for uri in json_uris:
        try:
            document = storage.read_json(uri)
            kind, problem = classify_document(document), None
        except StorageError as error:
            kind, problem = "unknown", f"읽지 못했습니다({type(error).__name__})."
        except (ValueError, TypeError):
            kind, problem = "unknown", "올바른 JSON이 아닙니다."
        examined.append(
            {"name": uri.rsplit("/", 1)[-1], "uri": uri, "kind": kind, "problem": problem}
        )

    return _assemble(prefix, examined, empty=not json_uris)


# --------------------------------------------------------------- 선택 저장

def _selection_path() -> Path:
    return web_state_dir() / "data_source.json"


def _prepared_selection(stored: dict[str, Any]) -> dict[str, Any] | None:
    """data pipeline이 준비해 준 산출물을 고른 경우입니다.

    폴더를 훑어 찾은 것이 아니라 pipeline이 위치를 알려 준 것이므로 다시 살피지 않고
    기록해 둔 URI를 그대로 씁니다. S3 산출물일 수도 있어 폴더 검사가 성립하지 않습니다.
    """

    data = stored.get("data")
    if not isinstance(data, dict) or set(data) != set(DATA_ARTIFACT_KEYS):
        return None
    return {
        "origin": "prepared",
        "directory": stored.get("processed_prefix"),
        "complete": True,
        "data": dict(data),
        "matched": {
            key: {"name": str(data[key]).rsplit("/", 1)[-1], "uri": data[key]}
            for key in DATA_ARTIFACT_KEYS
        },
        "labels": dict(_LABELS),
        "missing": [],
        "problems": [],
        "examined": [],
        "available": True,
        "selected_at": stored.get("selected_at"),
        "preparation": stored.get("preparation"),
    }


def load_selection() -> dict[str, Any] | None:
    """지금 고른 전처리 데이터셋을 돌려줍니다. 없으면 ``None``입니다.

    폴더를 직접 고른 경우에는 그 사이 내용이 바뀌었을 수 있으므로 다시 살펴 최신 상태를
    담습니다. data pipeline이 준비해 준 경우에는 기록해 둔 URI를 그대로 씁니다.
    """

    path = _selection_path()
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(stored, dict):
        return None
    if stored.get("origin") == "prepared":
        return _prepared_selection(stored)
    if not isinstance(stored.get("directory"), str):
        return None

    def unavailable(problems: list[str]) -> dict[str, Any]:
        """선택은 남겨 두되 지금 쓸 수 없다는 사실을 그대로 알립니다."""

        return {
            "origin": "folder",
            "directory": stored["directory"],
            "selected_at": stored.get("selected_at"),
            "available": False,
            "complete": False,
            "data": {},
            "matched": {key: None for key in DATA_ARTIFACT_KEYS},
            "labels": dict(_LABELS),
            "missing": list(DATA_ARTIFACT_KEYS),
            "problems": problems,
            "examined": [],
        }

    try:
        current = inspect_directory(stored["directory"])
    except WebValidationError as error:
        # 폴더가 사라졌거나 접근할 수 없게 된 경우입니다.
        return unavailable([item["message"] for item in error.as_list()])
    except (OSError, ValueError):
        # 선택 조회가 예외로 터지면 화면 전체가 멈춥니다. 어떤 경우에도 값을 돌려줍니다.
        return unavailable(["저장해 둔 폴더 경로를 해석할 수 없습니다."])

    current["origin"] = "folder"
    current["available"] = True
    current["selected_at"] = stored.get("selected_at")
    return current


def _write_selection(payload: dict[str, Any]) -> None:
    target = _selection_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=".data_source-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def save_prepared_selection(
    data: Mapping[str, str], preparation: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """data pipeline이 준비해 준 artifact를 현재 데이터셋으로 고릅니다."""

    missing = [key for key in DATA_ARTIFACT_KEYS if not str(data.get(key, "")).strip()]
    if missing:
        raise WebValidationError(
            [FieldError("data", f"준비 결과에 {', '.join(missing)}이(가) 없습니다.")]
        )

    meta = dict(preparation or {})
    _write_selection(
        {
            "origin": "prepared",
            "data": {key: str(data[key]) for key in DATA_ARTIFACT_KEYS},
            "processed_prefix": meta.get("processed_prefix"),
            "preparation": meta,
            "selected_at": _now_text(),
        }
    )
    selection = load_selection()
    if selection is None:  # 저장 직후에는 항상 읽혀야 합니다.
        raise WebValidationError([FieldError("data", "준비 결과를 저장하지 못했습니다.")])
    return selection


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def save_selection(directory: object) -> dict[str, Any]:
    """전처리 데이터셋을 고릅니다. artifact 4개를 모두 찾은 경우에만 저장합니다."""

    result = inspect_directory(directory)
    if not result["complete"]:
        missing = ", ".join(_LABELS[key] for key in result["missing"])
        raise WebValidationError(
            [FieldError("directory", f"이 폴더에서 {missing}을(를) 찾지 못했습니다.")]
        )

    selected_at = _now_text()
    _write_selection(
        {"origin": "folder", "directory": result["directory"], "selected_at": selected_at}
    )
    result["origin"] = "folder"
    result["available"] = True
    result["selected_at"] = selected_at
    return result


def clear_selection() -> None:
    """고른 전처리 데이터셋을 지웁니다."""

    _selection_path().unlink(missing_ok=True)


# ------------------------------------------------- data pipeline으로 검증

VERIFY_TIMEOUT_SECONDS = 120


def build_data_config(data_inputs: dict[str, str]) -> dict[str, Any]:
    """``--only data``에 넘길 최소 config를 만듭니다.

    data pipeline은 ``config["inputs"]["data"]``만 봅니다. ``execution.mode``가
    ``"dummy"``면 검증을 건너뛰고 dummy 결과를 돌려주므로 반드시 덮어씁니다.
    """

    uses_s3_inputs = any(value.lower().startswith("s3://") for value in data_inputs.values())
    storage: dict[str, Any] = (
        {"backend": "s3", "s3": {"prefix": ""}}
        if uses_s3_inputs
        else {"backend": "local", "local": {"root": "artifacts"}}
    )
    return {
        "project": {"name": "pill-object-detection"},
        "execution": {"mode": "real"},
        "storage": storage,
        "inputs": {"data": dict(data_inputs)},
    }


def verify_with_pipeline(data_inputs: dict[str, str]) -> dict[str, Any]:
    """실제 data pipeline을 공개 CLI로 불러 계약이 성립하는지 확인합니다.

    web이 자체적으로 하는 검사와 달리, 여기서는 ``main_pipeline``이 data의 ``run()``을
    돌리고 필수 artifact 검사까지 통과시킵니다. 즉 전체 실행에서 data → train 연결이
    성립하는지를 학습 전에 같은 경로로 확인합니다.

    data pipeline은 파일을 만들지 않습니다. 넘긴 URI를 검증해 그대로 돌려줄 뿐입니다.
    """

    # train과 같은 곳에 임시 config를 쓰고, 끝나면 지웁니다.
    from .train_config import config_relative_path, write_runtime_config

    config_id = write_runtime_config(build_data_config(data_inputs))
    try:
        completed = runner.run_stage(
            config_relative_path(config_id),
            "data",
            cwd=repository_root(),
            timeout=VERIFY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": None,
            "message": "data pipeline이 시간 안에 끝나지 않았습니다.",
            "artifacts": {},
            "summary": {},
        }
    except OSError as error:
        return {
            "ok": False,
            "exit_code": None,
            "message": f"data pipeline을 실행하지 못했습니다({type(error).__name__}).",
            "artifacts": {},
            "summary": {},
        }
    finally:
        (config_dir() / f"{config_id}.json").unlink(missing_ok=True)

    result = _parse_result(completed.stdout)
    if result is None:
        lines = (completed.stderr or "").strip().splitlines()
        detail = sanitize_line(lines[-1]) if lines else ""
        return {
            "ok": False,
            "exit_code": completed.returncode,
            "message": f"data pipeline 결과를 해석하지 못했습니다. {detail}".strip(),
            "artifacts": {},
            "summary": {},
        }

    stage_artifacts = result.get("artifacts")
    stage_summary = result.get("summary")
    return {
        "ok": completed.returncode == 0 and result.get("status") == "ok",
        "exit_code": completed.returncode,
        "message": sanitize_line(str(result.get("message") or "")),
        "artifacts": _unwrap_stage(stage_artifacts),
        "summary": _unwrap_stage(stage_summary),
    }


# ---------------------------------------------------- 원본에서 준비 실행

# data pipeline이 지원하는 분할 비율. 값은 data 담당이 정한 것을 그대로 씁니다.
SPLIT_RATIOS = ("8:2", "9:1")

# 원본을 다 읽어야 하므로 검증보다 훨씬 오래 걸릴 수 있습니다.
PREPARE_TIMEOUT_SECONDS = 60 * 60


STORAGE_BACKENDS = ("auto", "local", "s3")


def storage_environment() -> dict[str, Any]:
    """어느 storage를 쓰게 되는지 화면에 알려 주기 위한 정보입니다.

    credential 자체는 boto3의 기본 chain이 다루며 여기서 읽지도 보여 주지도 않습니다.
    """

    bucket = os.environ.get("PILL_STORAGE_S3_BUCKET", "").strip()
    forced = os.environ.get("PILL_STORAGE_BACKEND", "").strip().lower()
    return {
        "bucket": bucket or None,
        "bucket_configured": bool(bucket),
        "profile_configured": bool(os.environ.get("AWS_PROFILE", "").strip()),
        "region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None,
        # PILL_STORAGE_BACKEND가 있으면 config보다 우선하므로 선택이 무시됩니다.
        "forced_backend": forced or None,
        "default_backend": forced or ("s3" if bucket else "local"),
    }


def resolve_backend(backend: str) -> str:
    """``auto``는 환경 설정을 보고 정합니다."""

    if backend not in STORAGE_BACKENDS:
        raise WebValidationError(
            [FieldError("backend", f"{', '.join(STORAGE_BACKENDS)} 중 하나여야 합니다.")]
        )
    if backend != "auto":
        return backend
    return "s3" if storage_environment()["bucket_configured"] else "local"


def build_prepare_config(
    split_ratio: str,
    *,
    seed: int = 42,
    overwrite: bool = False,
    backend: str = "auto",
    raw_prefix: str | None = None,
    processed_root: str | None = None,
) -> dict[str, Any]:
    """``--only data``로 원본에서 artifact를 만들게 하는 config를 만듭니다."""

    if split_ratio not in SPLIT_RATIOS:
        allowed = ", ".join(SPLIT_RATIOS)
        raise WebValidationError(
            [FieldError("split_ratio", f"{allowed} 중 하나여야 합니다.")]
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise WebValidationError(
            [FieldError("seed", "0 이상 2**32 미만의 정수여야 합니다.")]
        )
    if not isinstance(overwrite, bool):
        raise WebValidationError([FieldError("overwrite", "true 또는 false여야 합니다.")])

    section: dict[str, Any] = {
        "prepare": True,
        "split_ratio": split_ratio,
        "seed": seed,
        "overwrite": overwrite,
    }
    if raw_prefix:
        section["raw_prefix"] = raw_prefix
    if processed_root:
        section["processed_root"] = processed_root

    resolved = resolve_backend(backend)
    if resolved == "s3":
        environment = storage_environment()
        if not environment["bucket_configured"]:
            raise WebValidationError(
                [
                    FieldError(
                        "backend",
                        "S3를 쓰려면 PILL_STORAGE_S3_BUCKET 환경 변수가 필요합니다.",
                    )
                ]
            )
        # bucket 이름은 환경 변수에서 오므로 config 파일에 적지 않습니다.
        storage: dict[str, Any] = {"backend": "s3", "s3": {"prefix": ""}}
    else:
        storage = {"backend": "local", "local": {"root": "artifacts"}}

    return {
        "project": {"name": "pill-object-detection"},
        "execution": {"mode": "real"},
        "storage": storage,
        "data": section,
    }


def _unsupported_result(result: dict[str, Any]) -> bool:
    """설치된 data pipeline이 준비 기능을 갖고 있지 않은 경우입니다.

    준비를 요청했는데 응답의 ``summary.mode``가 ``"prepare"``가 아니면, 그 pipeline은
    ``data.prepare``를 아예 모르고 기존 pass-through 경로로 떨어진 것입니다.
    """

    summary = result.get("summary")
    return not (isinstance(summary, Mapping) and summary.get("mode") == "prepare")


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """실제 data pipeline을 불러 원본에서 artifact 4개를 만듭니다."""

    from .train_config import config_relative_path, write_runtime_config

    config_id = write_runtime_config(config)
    try:
        completed = runner.run_stage(
            config_relative_path(config_id),
            "data",
            cwd=repository_root(),
            timeout=PREPARE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "supported": True, "exit_code": None, "artifacts": {},
                "summary": {}, "message": "데이터 준비가 시간 안에 끝나지 않았습니다."}
    except OSError as error:
        return {"ok": False, "supported": True, "exit_code": None, "artifacts": {},
                "summary": {},
                "message": f"data pipeline을 실행하지 못했습니다({type(error).__name__})."}
    finally:
        (config_dir() / f"{config_id}.json").unlink(missing_ok=True)

    result = _parse_result(completed.stdout)
    if result is None:
        lines = (completed.stderr or "").strip().splitlines()
        detail = sanitize_line(lines[-1]) if lines else ""
        return {"ok": False, "supported": True, "exit_code": completed.returncode,
                "artifacts": {}, "summary": {},
                "message": f"data pipeline 결과를 해석하지 못했습니다. {detail}".strip()}

    stage_summary = _unwrap_stage(result.get("summary"))
    if _unsupported_result({"summary": stage_summary}):
        return {
            "ok": False,
            "supported": False,
            "exit_code": completed.returncode,
            "artifacts": {},
            "summary": stage_summary,
            "message": (
                "설치된 data pipeline이 아직 원본에서 데이터를 준비하는 기능을 "
                "지원하지 않습니다. 준비 기능이 들어간 뒤 다시 시도해 주세요."
            ),
        }

    return {
        "ok": completed.returncode == 0 and result.get("status") == "ok",
        "supported": True,
        "exit_code": completed.returncode,
        "artifacts": _unwrap_stage(result.get("artifacts")),
        "summary": stage_summary,
        "message": sanitize_line(str(result.get("message") or "")),
    }


def _parse_result(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _unwrap_stage(value: Any) -> dict[str, Any]:
    """``--only data`` 결과는 stage 이름으로 한 겹 감싸여 옵니다."""

    if not isinstance(value, dict):
        return {}
    stage = value.get("data")
    return dict(stage) if isinstance(stage, dict) else dict(value)
