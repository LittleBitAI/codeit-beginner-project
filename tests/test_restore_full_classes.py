"""`기타 알약`으로 뭉친 category_id를 되돌리는 도구의 계약을 확인합니다."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.aihub_to_competition import OTHER_CATEGORY_ID, OTHER_CATEGORY_NAME
from scripts.restore_full_classes import RestoreError, restore


WIDTH, HEIGHT = 976, 1280


def _document(
    image_name: str,
    code: str,
    *,
    category_id: int,
    dl_name: str,
    bbox: Any = (10.0, 20.0, 30.0, 40.0),
    image_id: int = 100_007,
    annotation_id: int = 1_000_009,
) -> dict[str, Any]:
    """이미 대회 형식으로 바뀐 문서를 흉내 냅니다."""

    return {
        "images": [
            {
                "id": image_id,
                "width": WIDTH,
                "height": HEIGHT,
                "file_name": image_name,
                "drug_N": code,
                "dl_mapping_code": code,
                # 원본 dl_idx는 K코드 숫자보다 1 작습니다. 이 값을 쓰면 안 됩니다.
                "dl_idx": str(int(code.split("-")[1]) - 1),
                "dl_name": dl_name,
            }
        ],
        "type": "instances",
        "annotations": [
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": list(bbox),
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
        ],
        "categories": [
            {
                "supercategory": "pill",
                "id": category_id,
                "name": OTHER_CATEGORY_NAME if category_id == OTHER_CATEGORY_ID else dl_name,
            }
        ],
    }


def _write(root: Path, combination: str, code: str, document: dict[str, Any]) -> Path:
    directory = root / "train_annotations" / f"{combination}_json" / code
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{combination}_0_2_0_2_70_000_200.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def _source(root: Path) -> Path:
    """대회 class 하나와 `기타 알약`으로 뭉친 것 하나를 담은 입력을 만듭니다."""

    combination = "K-002483-000573"
    _write(
        root,
        combination,
        "K-002483",
        _document("shot.png", "K-002483", category_id=2483, dl_name="뮤테란캡슐 100mg"),
    )
    _write(
        root,
        combination,
        "K-000573",
        _document(
            "shot.png",
            "K-000573",
            category_id=OTHER_CATEGORY_ID,
            dl_name="게보린정 300mg/PTP",
            bbox=(50.0, 60.0, 70.0, 80.0),
            annotation_id=1_000_010,
        ),
    )
    return root


def _documents(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in (root / "train_annotations").rglob("*.json")
    }


def test_restores_the_pill_that_was_collapsed_into_the_other_class(tmp_path: Path) -> None:
    """대회 밖 알약이 자기 코드를 되찾습니다. 이것이 이 도구의 목적입니다."""

    report = restore(_source(tmp_path / "in"), tmp_path / "out")

    restored = _documents(tmp_path / "out")["K-000573"]
    assert restored["annotations"][0]["category_id"] == 573
    assert restored["categories"] == [
        {"supercategory": "pill", "id": 573, "name": "게보린정 300mg/PTP"}
    ]
    assert report["restored_annotations"] == 1
    assert report["category_count"] == 2


def test_keeps_ids_and_boxes_untouched(tmp_path: Path) -> None:
    """image_id·annotation_id·bbox를 건드리면 data pipeline의 split과 어긋납니다."""

    restore(_source(tmp_path / "in"), tmp_path / "out")

    restored = _documents(tmp_path / "out")["K-000573"]["annotations"][0]
    assert restored["id"] == 1_000_010
    assert restored["image_id"] == 100_007
    assert restored["bbox"] == [50.0, 60.0, 70.0, 80.0]


def test_stops_when_an_existing_id_disagrees_with_the_product_code(tmp_path: Path) -> None:
    """뭉친 것도 아닌데 코드와 다른 id는 데이터가 어긋났다는 뜻이라 멈춥니다.

    조용히 덮어쓰면 어느 쪽이 맞는지 모르는 채로 label이 바뀝니다.
    """

    root = _source(tmp_path / "in")
    _write(
        root,
        "K-002483-000573",
        "K-009999",
        _document("shot.png", "K-009999", category_id=1234, dl_name="다른 약"),
    )

    with pytest.raises(RestoreError, match="category_id"):
        restore(root, tmp_path / "out")


def test_stops_when_the_product_code_is_missing(tmp_path: Path) -> None:
    """복원할 근거가 없으면 넘기지 않고 멈춥니다."""

    root = tmp_path / "in"
    document = _document(
        "shot.png", "K-000573", category_id=OTHER_CATEGORY_ID, dl_name="게보린정 300mg/PTP"
    )
    del document["images"][0]["dl_mapping_code"]
    del document["images"][0]["drug_N"]
    _write(root, "K-000573", "K-000573", document)

    with pytest.raises(RestoreError, match="dl_mapping_code"):
        restore(root, tmp_path / "out")


def test_stops_when_only_the_authoritative_code_is_missing(tmp_path: Path) -> None:
    """`drug_N`으로 슬쩍 대신하지 않습니다.

    권위값은 `dl_mapping_code`입니다. 그것이 없을 때 다른 필드로 넘어가면 두 값이
    어긋나는 문서에서 엉뚱한 알약의 label이 조용히 붙습니다.
    """

    root = tmp_path / "in"
    document = _document(
        "shot.png", "K-000573", category_id=OTHER_CATEGORY_ID, dl_name="게보린정 300mg/PTP"
    )
    del document["images"][0]["dl_mapping_code"]
    _write(root, "K-000573", "K-000573", document)

    with pytest.raises(RestoreError, match="dl_mapping_code"):
        restore(root, tmp_path / "out")


def test_stops_when_the_two_codes_disagree(tmp_path: Path) -> None:
    """두 필드가 다르면 어느 쪽이 맞는지 알 수 없으므로 멈춥니다."""

    root = tmp_path / "in"
    document = _document(
        "shot.png", "K-000573", category_id=OTHER_CATEGORY_ID, dl_name="게보린정 300mg/PTP"
    )
    document["images"][0]["drug_N"] = "K-009999"
    _write(root, "K-000573", "K-000573", document)

    with pytest.raises(RestoreError, match="drug_N"):
        restore(root, tmp_path / "out")


def test_writes_nothing_when_a_later_document_fails(tmp_path: Path) -> None:
    """뒤쪽에서 멈춰도 앞쪽 결과가 남지 않습니다.

    반쯤 만들어진 출력이 남으면 그 자리로 dataset을 만들었을 때 일부 알약만 복원된
    상태가 되고, class 수만 보고는 알아채지 못합니다.
    """

    root = _source(tmp_path / "in")
    broken = _document("shot.png", "K-009999", category_id=1234, dl_name="다른 약")
    _write(root, "K-002483-000573", "K-009999", broken)
    output = tmp_path / "out"

    with pytest.raises(RestoreError):
        restore(root, output)

    assert not list(output.rglob("*.json"))


def test_refuses_when_the_output_place_is_a_file(tmp_path: Path) -> None:
    """출력 자리가 파일이어도 이 도구의 오류로 알려 줍니다.

    그냥 두면 NotADirectoryError가 그대로 튀어나와, 쓰는 사람은 도구가 멈춘 이유를
    traceback에서 읽어야 합니다.
    """

    root = _source(tmp_path / "in")
    output = tmp_path / "out"
    output.write_text("파일입니다", encoding="utf-8")

    with pytest.raises(RestoreError, match="directory"):
        restore(root, output)

    assert output.read_text(encoding="utf-8") == "파일입니다"


def test_refuses_when_the_output_place_is_not_empty(tmp_path: Path) -> None:
    """이전 실행의 결과를 말없이 지우거나 위에 덧쓰지 않습니다.

    덧쓰기만 하면 이번에 안 나온 파일이 남아 옛 label이 섞입니다. 지우는 것은 사람이
    결정할 일이라, 비어 있지 않으면 아무것도 하지 않습니다.
    """

    root = _source(tmp_path / "in")
    restore(root, tmp_path / "out")

    with pytest.raises(RestoreError, match="이미"):
        restore(root, tmp_path / "out")

    # report만 남아 있어도 마찬가지입니다. 그 파일도 이 도구가 쓰는 자리입니다.
    other = tmp_path / "other"
    other.mkdir()
    (other / "restore_report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RestoreError, match="이미"):
        restore(root, other)
