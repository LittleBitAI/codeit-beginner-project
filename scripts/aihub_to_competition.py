"""AI Hub 경구약제 이미지 데이터를 대회 raw 데이터와 같은 형식으로 바꿉니다.

두 데이터는 원래 같은 출처라서 디렉터리 구조와 JSON schema가 같습니다. 다른 점은
**식별자뿐**입니다. AI Hub 원본은 `image_id`, `category_id`, `annotation_id`가 모두
`1`로 채워져 있고, `categories`도 `{"id": 1, "name": "Drug"}` 하나뿐입니다. 이
module은 그 세 식별자를 대회 규약대로 다시 매겨 아래 구조로 내보냅니다.

    <output>/train_annotations/<조합>_json/<알약코드>/<이미지>.json
    <output>/train_images/<이미지>.png          (--copy-images 를 준 경우에만)
    <output>/conversion_report.json

`category_id`는 `dl_idx`가 아니라 `dl_mapping_code`의 숫자부에서 만듭니다. AI Hub의
`dl_idx`는 그 값보다 1 작고(`K-001900` → `1899`) `K-053384`는 `130376`으로 아예
다르기 때문에, 그대로 쓰면 대회 label 공간과 어긋납니다.

`--label-space`는 class 공간을 고릅니다. `full`은 AI Hub 알약 116종을 그대로 쓰고,
`competition-plus-other`는 대회 class만 남기고 나머지를 `기타 알약` 하나로 합칩니다.
대회 밖 알약의 bbox를 지우는 선택지는 없습니다. 한 사진에 알약이 3~4개라서, 지우면
그 알약이 라벨 없이 남아 "알약은 배경"으로 학습되기 때문입니다.

이미지 파일은 기본적으로 복사하지 않습니다. annotation만 먼저 만들어 검증하고,
17 GiB가 넘는 이미지 이동은 따로 결정하기 위한 기본값입니다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SOURCE_IMAGE_DIRECTORY = "원천데이터"
SOURCE_LABEL_DIRECTORY = "라벨링데이터"
INDEX_IMAGE_SUFFIX = "_index.png"
PILL_CODE_PATTERN = re.compile(r"\d{6}")

# 대회 데이터가 이미 쓰는 범위(image_id 14~1499, annotation_id 50~5691) 위에서
# 시작해 두 데이터를 한 prefix에 합쳐도 식별자가 겹치지 않게 합니다.
DEFAULT_IMAGE_ID_BASE = 100_000
DEFAULT_ANNOTATION_ID_BASE = 1_000_000

COPY_MODES = ("none", "hardlink", "copy")
MAX_REPORTED_EXAMPLES = 20

# `full`은 AI Hub의 알약 116종을 그대로 class로 씁니다.
# `competition-plus-other`는 대회 class만 남기고 나머지를 보조 class 하나로 합칩니다.
LABEL_SPACE_FULL = "full"
LABEL_SPACE_COMPETITION = "competition-plus-other"
LABEL_SPACES = (LABEL_SPACE_FULL, LABEL_SPACE_COMPETITION)

# 실제 알약 코드와 겹치지 않는 값이라야 합니다. 알약 코드는 6자리입니다.
OTHER_CATEGORY_ID = 999_999
OTHER_CATEGORY_NAME = "기타 알약"

# 이미지를 통째로 제외하는 이유입니다.
REASON_NO_LABEL = "라벨 json 없음"
REASON_BAD_BBOX = "bbox 없음 또는 형식 오류"
REASON_DUPLICATE_PILL = "같은 알약 문서 중복"
REASON_PILL_COUNT = "조합 알약 수와 문서 수 불일치"
REASON_BAD_ISCROWD = "iscrowd 값 오류"

# 개별 문서를 건너뛰는 이유입니다.
SKIP_JUNK = "AppleDouble junk"
SKIP_UNREADABLE = "json 파싱 실패"
SKIP_SHAPE = "문서 구조 이상"
SKIP_NO_IMAGE = "원천 이미지 없음"


class ConversionError(RuntimeError):
    """변환을 시작할 수 없을 때 올리는 오류입니다."""


def category_id(mapping_code: str) -> int:
    """`dl_mapping_code`에서 대회 규약의 `category_id`를 만듭니다.

    `K-001900` → `1900`. 원본 `dl_idx`를 쓰지 않는 이유는 module docstring에
    적어 두었습니다.
    """

    text = str(mapping_code).strip()
    parts = text.split("-")
    if len(parts) < 2 or not parts[1].isdigit():
        raise ConversionError(f"알약 코드에서 category_id를 만들 수 없습니다: {text!r}")
    return int(parts[1])


def _pill_count(image_name: str) -> int:
    """이미지 이름 앞의 조합 코드에서 알약 개수를 셉니다."""

    return len(PILL_CODE_PATTERN.findall(image_name.split("_")[0]))


def _is_valid_bbox(value: Any) -> bool:
    """대회 안내의 "존재하고 유효한" bbox 조건입니다: 길이 4의 list."""

    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _is_flag(value: Any) -> bool:
    """`iscrowd`로 쓸 수 있는 0 또는 1인지 확인합니다."""

    return isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}


def _source_images(source: Path) -> dict[str, Path]:
    """원천데이터에서 학습용 PNG를 모읍니다. `_index.png`는 제외합니다."""

    images: dict[str, Path] = {}
    root = source / SOURCE_IMAGE_DIRECTORY
    if not root.is_dir():
        raise ConversionError(f"원천데이터 디렉터리를 찾지 못했습니다: {SOURCE_IMAGE_DIRECTORY}")
    for path in root.rglob("*.png"):
        if path.name.endswith(INDEX_IMAGE_SUFFIX):
            continue
        images[path.name] = path
    if not images:
        raise ConversionError("원천데이터에서 학습용 PNG를 찾지 못했습니다.")
    return images


def _read_documents(
    source: Path,
    images: dict[str, Path],
    skipped: Counter[str],
) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    """라벨링데이터를 읽어 이미지 이름별 문서 목록으로 묶습니다."""

    root = source / SOURCE_LABEL_DIRECTORY
    if not root.is_dir():
        raise ConversionError(f"라벨링데이터 디렉터리를 찾지 못했습니다: {SOURCE_LABEL_DIRECTORY}")

    documents: defaultdict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("._"):
            skipped[SKIP_JUNK] += 1
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            skipped[SKIP_UNREADABLE] += 1
            continue
        if not _has_expected_shape(document):
            skipped[SKIP_SHAPE] += 1
            continue
        image_name = document["images"][0]["file_name"]
        if image_name not in images:
            skipped[SKIP_NO_IMAGE] += 1
            continue
        documents[image_name].append((path, document))
    return documents


def _has_expected_shape(document: Any) -> bool:
    """변환에 필요한 field가 모두 있는 문서인지 확인합니다."""

    if not isinstance(document, dict):
        return False
    required = ("images", "annotations", "categories")
    if not all(isinstance(document.get(key), list) for key in required):
        return False
    if len(document["images"]) != 1 or not isinstance(document["images"][0], dict):
        return False
    image = document["images"][0]
    if not isinstance(image.get("file_name"), str) or not image["file_name"].strip():
        return False
    if not all(isinstance(image.get(key), int) and image[key] > 0 for key in ("width", "height")):
        return False
    if not isinstance(image.get("dl_mapping_code"), str) or not image["dl_mapping_code"].strip():
        return False
    return isinstance(image.get("dl_name"), str) and bool(image["dl_name"].strip())


def _excluded_reason(entries: list[tuple[Path, dict[str, Any]]], image_name: str) -> str | None:
    """이미지를 통째로 제외해야 하는 이유를 찾습니다. 쓸 수 있으면 `None`입니다."""

    codes = [document["images"][0]["dl_mapping_code"] for _, document in entries]
    if len(set(codes)) != len(codes):
        return REASON_DUPLICATE_PILL
    expected = _pill_count(image_name)
    if expected and len(codes) != expected:
        return REASON_PILL_COUNT
    for _, document in entries:
        annotations = document["annotations"]
        if not annotations:
            return REASON_BAD_BBOX
        if not all(_is_valid_bbox(annotation.get("bbox")) for annotation in annotations):
            return REASON_BAD_BBOX
        # `coco.py`는 iscrowd가 0/1이 아니면 실행 전체를 중단합니다. 없는 값은 그
        # module이 0으로 보므로 그대로 넘기고, 어긋난 값만 여기서 걸러 냅니다.
        flags = [annotation["iscrowd"] for annotation in annotations if "iscrowd" in annotation]
        if not all(_is_flag(flag) for flag in flags):
            return REASON_BAD_ISCROWD
    return None


def _combo_directory(image_name: str) -> str:
    """이미지 이름에서 대회 layout의 조합 디렉터리 이름을 만듭니다."""

    return f"{image_name.split('_')[0]}_json"


def _guard_output(output: Path, overwrite: bool) -> None:
    """이미 만들어 둔 산출물을 덮지 않도록 먼저 막습니다.

    `overwrite`일 때는 지난 문서를 남겨 두지 않고 지웁니다. 이미지 하나가 제외되면
    그 뒤 `image_id`가 모두 밀리므로, 남은 문서와 새 문서가 섞이면 같은 tree 안에
    어긋난 식별자가 함께 있게 됩니다. 다만 이 tool이 만든 디렉터리라는 증거인
    `conversion_report.json`이 없으면 지우지 않습니다.
    """

    annotations = output / "train_annotations"
    if not annotations.exists() or not any(annotations.rglob("*.json")):
        return
    if not overwrite:
        raise FileExistsError(
            f"이미 변환 결과가 있습니다: {annotations.as_posix()} (덮어쓰려면 --overwrite)"
        )
    if not (output / "conversion_report.json").is_file():
        raise ConversionError(
            f"이 tool이 만든 결과가 아니어서 지우지 않습니다: {annotations.as_posix()} "
            "(conversion_report.json이 없습니다)"
        )
    shutil.rmtree(annotations)


def _competition_category_ids(location: Path | None) -> set[int]:
    """`class_map.json`에서 대회 class id를 읽습니다.

    형식은 data pipeline이 내보내는 `{"<category id>": "<category name>"}`입니다.
    """

    if location is None:
        raise ConversionError(
            f"--label-space {LABEL_SPACE_COMPETITION}에는 --competition-classes로 "
            "대회 class_map.json을 함께 줘야 합니다."
        )
    try:
        document = json.loads(Path(location).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConversionError(f"대회 class 목록을 읽지 못했습니다: {error}") from error
    if not isinstance(document, dict) or not document:
        raise ConversionError(
            '대회 class 목록은 {"<category id>": "<category name>"} 형식이어야 합니다.'
        )
    identifiers = set()
    for key in document:
        text = str(key).strip()
        if not text.lstrip("-").isdigit():
            raise ConversionError(f"대회 class 목록의 key가 정수가 아닙니다: {key!r}")
        identifiers.add(int(text))
    return identifiers


def _place_image(source_path: Path, target: Path, mode: str) -> None:
    """이미지를 hardlink 또는 복사로 배치합니다."""

    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source_path, target)
            return
        except OSError:
            pass
    target.write_bytes(source_path.read_bytes())


def convert(
    source: Path,
    output: Path,
    *,
    image_id_base: int = DEFAULT_IMAGE_ID_BASE,
    annotation_id_base: int = DEFAULT_ANNOTATION_ID_BASE,
    copy_images: str = "none",
    overwrite: bool = False,
    label_space: str = LABEL_SPACE_FULL,
    competition_classes: Path | None = None,
) -> dict[str, Any]:
    """AI Hub tree를 대회 형식으로 변환하고 결과 보고를 돌려줍니다."""

    if copy_images not in COPY_MODES:
        raise ConversionError(f"--copy-images 값은 {', '.join(COPY_MODES)} 중 하나여야 합니다.")
    if label_space not in LABEL_SPACES:
        raise ConversionError(f"--label-space 값은 {', '.join(LABEL_SPACES)} 중 하나여야 합니다.")
    keep_ids: set[int] | None = None
    if label_space == LABEL_SPACE_COMPETITION:
        keep_ids = _competition_category_ids(competition_classes)
    source, output = Path(source), Path(output)
    _guard_output(output, overwrite)

    skipped: Counter[str] = Counter()
    images = _source_images(source)
    documents = _read_documents(source, images, skipped)

    excluded: Counter[str] = Counter()
    excluded_examples: defaultdict[str, list[str]] = defaultdict(list)
    usable: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for image_name in sorted(images):
        entries = documents.get(image_name)
        if not entries:
            excluded[REASON_NO_LABEL] += 1
            if len(excluded_examples[REASON_NO_LABEL]) < MAX_REPORTED_EXAMPLES:
                excluded_examples[REASON_NO_LABEL].append(image_name)
            continue
        reason = _excluded_reason(entries, image_name)
        if reason is not None:
            excluded[reason] += 1
            if len(excluded_examples[reason]) < MAX_REPORTED_EXAMPLES:
                excluded_examples[reason].append(image_name)
            continue
        usable[image_name] = entries

    # image_id는 file_name 순서로 정해 같은 원본이면 언제나 같은 값이 나오게 합니다.
    image_ids = {name: image_id_base + order for order, name in enumerate(sorted(usable), start=1)}

    annotation_id = annotation_id_base
    written = 0
    annotation_count = 0
    other_annotations = 0
    categories: dict[int, str] = {}
    dl_idx_mismatch = 0
    for image_name in sorted(usable):
        image_id = image_ids[image_name]
        combo = _combo_directory(image_name)
        for path, document in sorted(usable[image_name], key=lambda item: item[0].parent.name):
            image = dict(document["images"][0])
            code = image["dl_mapping_code"]
            identifier = category_id(code)
            if str(image.get("dl_idx")) != str(identifier):
                dl_idx_mismatch += 1
            # 대회 밖 알약은 지우지 않고 보조 class로 옮깁니다. bbox만 지우면 그 알약이
            # 라벨 없이 사진에 남아 "알약은 배경"으로 학습됩니다.
            name = image["dl_name"]
            if keep_ids is not None and identifier not in keep_ids:
                identifier, name = OTHER_CATEGORY_ID, OTHER_CATEGORY_NAME
                other_annotations += len(document["annotations"])
            image["id"] = image_id
            annotations = []
            for annotation in document["annotations"]:
                annotation_id += 1
                annotations.append(
                    {
                        **annotation,
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": identifier,
                    }
                )
            categories[identifier] = name
            target = output / "train_annotations" / combo / code / f"{Path(image_name).stem}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "images": [image],
                        "type": document.get("type", "instances"),
                        "annotations": annotations,
                        "categories": [
                            {"supercategory": "pill", "id": identifier, "name": name}
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            written += 1
            annotation_count += len(annotations)
        if copy_images != "none":
            _place_image(images[image_name], output / "train_images" / image_name, copy_images)

    report: dict[str, Any] = {
        "source_images": len(images),
        "source_documents": sum(len(entries) for entries in documents.values()),
        "converted_images": len(usable),
        "converted_documents": written,
        "converted_annotations": annotation_count,
        "categories": len(categories),
        "category_names": {str(key): categories[key] for key in sorted(categories)},
        "groups": len({name.split("_")[0] for name in usable}),
        "image_id_range": [min(image_ids.values()), max(image_ids.values())] if image_ids else [],
        "annotation_id_range": [annotation_id_base + 1, annotation_id] if annotation_count else [],
        "excluded": dict(excluded),
        "excluded_examples": {key: value for key, value in excluded_examples.items()},
        "skipped_documents": dict(skipped),
        "dl_idx_mismatch": dl_idx_mismatch,
        "copy_images": copy_images,
        "label_space": label_space,
        "other_class_annotations": other_annotations,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI Hub 경구약제 데이터를 대회 raw 형식으로 변환합니다."
    )
    parser.add_argument("--source", required=True, help="AI Hub 1.Training 디렉터리")
    parser.add_argument("--output", default="artifacts/aihub-converted", help="변환 결과 위치")
    parser.add_argument("--image-id-base", type=int, default=DEFAULT_IMAGE_ID_BASE)
    parser.add_argument("--annotation-id-base", type=int, default=DEFAULT_ANNOTATION_ID_BASE)
    parser.add_argument("--copy-images", choices=COPY_MODES, default="none")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--label-space",
        choices=LABEL_SPACES,
        default=LABEL_SPACE_FULL,
        help=(
            f"{LABEL_SPACE_FULL}은 AI Hub 알약을 그대로 class로 씁니다. "
            f"{LABEL_SPACE_COMPETITION}은 대회 class만 남기고 나머지를 "
            f"'{OTHER_CATEGORY_NAME}' 하나로 합칩니다."
        ),
    )
    parser.add_argument(
        "--competition-classes",
        help=f"--label-space {LABEL_SPACE_COMPETITION}에 쓸 대회 class_map.json",
    )
    arguments = parser.parse_args(argv)

    try:
        report = convert(
            Path(arguments.source),
            Path(arguments.output),
            image_id_base=arguments.image_id_base,
            annotation_id_base=arguments.annotation_id_base,
            copy_images=arguments.copy_images,
            overwrite=arguments.overwrite,
            label_space=arguments.label_space,
            competition_classes=(
                Path(arguments.competition_classes) if arguments.competition_classes else None
            ),
        )
    except (ConversionError, FileExistsError) as error:
        print(f"변환 실패: {error}")
        return 1

    print(
        f"이미지 {report['converted_images']}/{report['source_images']}장, "
        f"annotation {report['converted_annotations']}개, "
        f"class {report['categories']}개를 만들었습니다."
    )
    for reason, count in sorted(report["excluded"].items()):
        print(f"  제외 - {reason}: {count}")
    for reason, count in sorted(report["skipped_documents"].items()):
        print(f"  건너뜀 - {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
