"""AI Hub 데이터를 대회 형식으로 바꾸는 변환기의 계약을 확인합니다."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.aihub_to_competition import (
    DEFAULT_ANNOTATION_ID_BASE,
    DEFAULT_IMAGE_ID_BASE,
    LABEL_SPACE_COMPETITION,
    OTHER_CATEGORY_ID,
    OTHER_CATEGORY_NAME,
    ConversionError,
    category_id,
    convert,
)
from src.pipelines.data.coco import consolidate


WIDTH, HEIGHT = 976, 1280

# `raw/v1/original/train_annotations`를 읽어 실측한 대회 데이터의 id 상한입니다.
COMPETITION_MAX_IMAGE_ID = 1499
COMPETITION_MAX_ANNOTATION_ID = 5691


def _document(image_name: str, code: str, *, bbox: Any, dl_name: str) -> dict[str, Any]:
    """AI Hub 원본 라벨 문서를 흉내 냅니다. id는 원본처럼 모두 1입니다."""

    image: dict[str, Any] = {
        "file_name": image_name,
        "width": WIDTH,
        "height": HEIGHT,
        "imgfile": image_name,
        "dl_mapping_code": code,
        # 원본 dl_idx는 K코드 숫자보다 1 작습니다. 변환기는 이 값을 쓰지 않아야 합니다.
        "dl_idx": str(int(code.split("-")[1]) - 1),
        "dl_name": dl_name,
        "id": 1,
    }
    annotation: dict[str, Any] = {
        "area": 100,
        "iscrowd": 0,
        "bbox": bbox,
        "category_id": 1,
        "segmentation": [],
        "id": 1,
        "image_id": 1,
    }
    return {
        "images": [image],
        "type": "instances",
        "annotations": [annotation],
        "categories": [{"supercategory": "pill", "id": 1, "name": "Drug"}],
    }


def _write(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def _source_tree(root: Path, *, bboxes: dict[str, Any] | None = None) -> Path:
    """알약 2개 조합 하나에 이미지 2장이 있는 최소 원본 tree를 만듭니다."""

    combo = "K-000250-000573"
    codes = {"K-000250": "마그밀정", "K-000573": "게보린정"}
    images = [f"{combo}_0_2_0_2_70_000_200.png", f"{combo}_0_2_0_2_75_000_200.png"]
    for name in images:
        image_path = root / "원천데이터" / "TS_1_조합" / combo / name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"png")
        for code, dl_name in codes.items():
            bbox = (bboxes or {}).get((name, code), [10, 20, 30, 40])
            _write(
                root / "라벨링데이터" / "TL_1_조합" / f"{combo}_json" / code / f"{Path(name).stem}.json",
                _document(name, code, bbox=bbox, dl_name=dl_name),
            )
    return root


def _annotation_documents(output: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "train_annotations").rglob("*.json"))
    ]


def test_category_id_uses_mapping_code_not_dl_idx() -> None:
    """category_id는 dl_mapping_code의 숫자부입니다.

    원본 `dl_idx`는 이 값보다 1 작아서 그대로 쓰면 대회 label 공간과 어긋납니다.
    """

    assert category_id("K-001900") == 1900
    assert category_id("K-000250") == 250
    # dl_idx가 130376인 K-053384도 코드 숫자부를 씁니다.
    assert category_id("K-053384") == 53384


def test_converted_documents_carry_competition_ids(tmp_path: Path) -> None:
    """같은 이미지의 알약 문서는 같은 image_id를, annotation은 고유 id를 받습니다."""

    report = convert(_source_tree(tmp_path / "src"), tmp_path / "out")

    documents = _annotation_documents(tmp_path / "out")
    assert len(documents) == 4

    ids_by_image: dict[str, set[int]] = {}
    annotation_ids: list[int] = []
    for document in documents:
        image = document["images"][0]
        ids_by_image.setdefault(image["file_name"], set()).add(image["id"])
        code = image["dl_mapping_code"]
        assert document["categories"][0]["id"] == category_id(code)
        assert document["categories"][0]["name"] == image["dl_name"]
        for annotation in document["annotations"]:
            assert annotation["image_id"] == image["id"]
            assert annotation["category_id"] == category_id(code)
            annotation_ids.append(annotation["id"])

    # 이미지 한 장은 image_id 하나만 가집니다(coco.py가 이를 강제합니다).
    assert [len(found) for found in ids_by_image.values()] == [1, 1]
    assert len(set(annotation_ids)) == len(annotation_ids) == 4
    assert report["converted_images"] == 2
    assert report["converted_annotations"] == 4
    assert report["categories"] == 2


def test_ids_never_collide_with_the_competition_dataset(tmp_path: Path) -> None:
    """기존 대회 데이터가 쓰는 id 위에서 시작합니다.

    측정값입니다: `raw/v1`의 image_id는 14~1499, annotation_id는 50~5691입니다. 두
    데이터를 한 prefix에 합쳐도 같은 id가 서로 다른 대상을 가리키면 안 됩니다.
    """

    assert DEFAULT_IMAGE_ID_BASE > COMPETITION_MAX_IMAGE_ID
    assert DEFAULT_ANNOTATION_ID_BASE > COMPETITION_MAX_ANNOTATION_ID

    convert(_source_tree(tmp_path / "src"), tmp_path / "out")

    documents = _annotation_documents(tmp_path / "out")
    image_ids = {document["images"][0]["id"] for document in documents}
    annotation_ids = {
        annotation["id"] for document in documents for annotation in document["annotations"]
    }
    assert min(image_ids) > COMPETITION_MAX_IMAGE_ID
    assert min(annotation_ids) > COMPETITION_MAX_ANNOTATION_ID


def test_image_is_dropped_when_a_pill_bbox_is_missing(tmp_path: Path) -> None:
    """bbox가 없거나 길이가 4가 아니면 그 이미지를 통째로 제외합니다.

    알약 하나가 라벨을 잃은 이미지를 학습에 넣으면 그 알약을 배경으로 가르칩니다.
    """

    source = _source_tree(
        tmp_path / "src",
        bboxes={
            ("K-000250-000573_0_2_0_2_70_000_200.png", "K-000250"): [],
            ("K-000250-000573_0_2_0_2_75_000_200.png", "K-000573"): [1, 2, 3],
        },
    )
    report = convert(source, tmp_path / "out")

    assert report["converted_images"] == 0
    assert report["converted_annotations"] == 0
    assert report["excluded"]["bbox 없음 또는 형식 오류"] == 2
    assert _annotation_documents(tmp_path / "out") == []


def test_image_is_dropped_when_iscrowd_is_not_a_flag(tmp_path: Path) -> None:
    """`iscrowd`가 0/1이 아니면 그 이미지를 제외합니다.

    `coco.py`는 이 값이 어긋나면 실행 전체를 중단시킵니다. 원본에 실제로 `58548`이
    들어간 문서가 있어, 값을 지어내지 않고 그 이미지만 빼는 쪽을 택했습니다.
    """

    source = _source_tree(tmp_path / "src")
    broken = (
        source
        / "라벨링데이터"
        / "TL_1_조합"
        / "K-000250-000573_json"
        / "K-000250"
        / "K-000250-000573_0_2_0_2_70_000_200.json"
    )
    document = json.loads(broken.read_text(encoding="utf-8"))
    document["annotations"][0]["iscrowd"] = 58548
    _write(broken, document)

    report = convert(source, tmp_path / "out")

    assert report["converted_images"] == 1
    assert report["excluded"]["iscrowd 값 오류"] == 1


def test_unusable_source_documents_are_skipped(tmp_path: Path) -> None:
    """macOS junk, 깨진 json, 없는 이미지를 가리키는 문서는 건너뜁니다."""

    source = _source_tree(tmp_path / "src")
    labels = source / "라벨링데이터" / "TL_1_조합" / "K-000250-000573_json" / "K-000250"
    (labels / "._junk.json").write_bytes(b"\x00\x01")
    (labels / "broken.json").write_text("{not json", encoding="utf-8")
    absent = "K-000250-000573_0_2_0_2_99_000_200"
    _write(
        labels / f"{absent}.json",
        _document(f"{absent}.png", "K-000250", bbox=[1, 2, 3, 4], dl_name="마그밀정"),
    )

    report = convert(source, tmp_path / "out")

    assert report["converted_images"] == 2
    assert report["skipped_documents"]["AppleDouble junk"] == 1
    assert report["skipped_documents"]["json 파싱 실패"] == 1
    assert report["skipped_documents"]["원천 이미지 없음"] == 1


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    """overwrite 없이 두 번 실행하면 이전 산출물을 건드리지 않고 실패합니다."""

    source = _source_tree(tmp_path / "src")
    convert(source, tmp_path / "out")
    annotations = tmp_path / "out" / "train_annotations"
    kept = sorted(path.name for path in annotations.rglob("*.json"))

    with pytest.raises(FileExistsError):
        convert(source, tmp_path / "out")

    assert sorted(path.name for path in annotations.rglob("*.json")) == kept


def test_overwrite_leaves_no_stale_documents(tmp_path: Path) -> None:
    """`overwrite`로 다시 만들 때 지난 실행이 남긴 문서가 섞이지 않습니다.

    이미지 하나가 제외되면 그 뒤 image_id가 전부 밀리므로, 지난 문서가 남으면
    같은 tree 안에 어긋난 식별자가 함께 있게 됩니다.
    """

    source = _source_tree(tmp_path / "src")
    convert(source, tmp_path / "out")
    assert len(_annotation_documents(tmp_path / "out")) == 4

    dropped = "K-000250-000573_0_2_0_2_70_000_200.png"
    for code in ("K-000250", "K-000573"):
        path = (
            source
            / "라벨링데이터"
            / "TL_1_조합"
            / "K-000250-000573_json"
            / code
            / f"{Path(dropped).stem}.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        document["annotations"][0]["bbox"] = []
        _write(path, document)

    report = convert(source, tmp_path / "out", overwrite=True)

    documents = _annotation_documents(tmp_path / "out")
    assert report["converted_images"] == 1
    assert len(documents) == report["converted_documents"] == 2
    assert {document["images"][0]["file_name"] for document in documents} == {
        "K-000250-000573_0_2_0_2_75_000_200.png"
    }


def test_overwrite_refuses_a_directory_this_tool_did_not_write(tmp_path: Path) -> None:
    """지난 실행 보고서가 없는 디렉터리는 `overwrite`로도 지우지 않습니다."""

    output = tmp_path / "out"
    (output / "train_annotations").mkdir(parents=True)
    (output / "train_annotations" / "keep.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ConversionError):
        convert(_source_tree(tmp_path / "src"), output, overwrite=True)

    assert (output / "train_annotations" / "keep.json").exists()


def test_competition_label_space_folds_unknown_pills_into_one_class(tmp_path: Path) -> None:
    """대회 밖 알약을 `기타 알약` 하나로 합치고, 이미지는 버리지 않습니다.

    대회 밖 알약의 bbox만 지우면 그 알약이 라벨 없이 사진에 남아 "알약은 배경"으로
    학습됩니다. 그래서 지우지 않고 보조 class로 옮깁니다.
    """

    classes = tmp_path / "class_map.json"
    classes.write_text(json.dumps({"250": "마그밀정"}, ensure_ascii=False), encoding="utf-8")

    report = convert(
        _source_tree(tmp_path / "src"),
        tmp_path / "out",
        label_space=LABEL_SPACE_COMPETITION,
        competition_classes=classes,
    )

    documents = _annotation_documents(tmp_path / "out")
    assert report["converted_images"] == 2
    assert len(documents) == 4
    by_code = {
        document["images"][0]["dl_mapping_code"]: document["categories"][0]
        for document in documents
    }
    assert by_code["K-000250"] == {"supercategory": "pill", "id": 250, "name": "마그밀정"}
    assert by_code["K-000573"]["id"] == OTHER_CATEGORY_ID
    assert by_code["K-000573"]["name"] == OTHER_CATEGORY_NAME
    for document in documents:
        expected = document["categories"][0]["id"]
        assert {annotation["category_id"] for annotation in document["annotations"]} == {expected}
    assert report["categories"] == 2
    assert report["other_class_annotations"] == 2


def test_competition_label_space_requires_the_class_list(tmp_path: Path) -> None:
    """어떤 class가 대회 것인지 모르면 합칠 수 없으므로 시작을 거부합니다."""

    with pytest.raises(ConversionError):
        convert(
            _source_tree(tmp_path / "src"),
            tmp_path / "out",
            label_space=LABEL_SPACE_COMPETITION,
        )


def test_output_passes_project_consolidate(tmp_path: Path) -> None:
    """변환 결과가 data pipeline의 consolidate()를 그대로 통과합니다."""

    convert(_source_tree(tmp_path / "src"), tmp_path / "out")

    root = tmp_path / "out" / "train_annotations"
    documents = [
        (str(path.relative_to(root)), json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.rglob("*.json"))
    ]
    image_names = sorted({document["images"][0]["file_name"] for _, document in documents})

    dataset = consolidate(documents, [f"raw/v2/train_images/{name}" for name in image_names])

    assert len(dataset.images) == 2
    assert len(dataset.annotations) == 4
    assert [category["id"] for category in dataset.categories] == [250, 573]
    assert dataset.excluded_images == []
