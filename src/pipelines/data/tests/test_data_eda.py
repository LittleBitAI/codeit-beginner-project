"""EDA 경로의 테스트.

실제 AWS에는 접속하지 않습니다. 이미지는 PIL로 그려 넣고, 저장소는 준비 경로
테스트가 쓰는 것과 같은 in-memory 대역을 씁니다.
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

from src.common import validate_pipeline_result
from src.pipelines.data import eda, run
from src.pipelines.data.errors import EdaError


IMAGE_SIZE = (160, 200)


# --- 대역 -----------------------------------------------------------------


class FakeStorage:
    """읽기·쓰기·내려받기만 하는 최소 storage 대역입니다."""

    def __init__(self) -> None:
        self.json: dict[str, Any] = {}
        self.blobs: dict[str, bytes] = {}
        self.downloaded: list[str] = []

    def exists(self, location: str) -> bool:
        return location in self.json or location in self.blobs

    def read_json(self, location: str) -> Any:
        return self.json[location]

    def write_json(self, location: str, value: Any, *, overwrite: bool = False) -> str:
        if location in self.json and not overwrite:
            raise AssertionError("덮어쓰기가 허용되지 않았는데 덮어썼습니다.")
        # 계약대로 JSON으로 바뀌는 값만 받습니다.
        self.json[location] = json.loads(json.dumps(value, allow_nan=False))
        return location

    def download_file(self, location: str, destination) -> None:
        self.downloaded.append(location)
        destination.write_bytes(self.blobs[location])


def draw_image(radii: list[int]) -> bytes:
    """밝은 배경 위에 어두운 원 몇 개를 그려 알약 사진을 흉내 냅니다."""

    image = Image.new("RGB", IMAGE_SIZE, (230, 230, 230))
    canvas = ImageDraw.Draw(image)
    for index, radius in enumerate(radii):
        x = 40 + (index % 2) * 80
        y = 50 + (index // 2) * 100
        canvas.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(30, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def manifest(images: list[dict], annotations: list[dict]) -> dict:
    return {"images": images, "annotations": annotations, "categories": []}


def build_storage(*, radii: int = 18, test_radii: int | None = None) -> FakeStorage:
    """train 2장·validation 1장·test 1장짜리 최소 dataset을 만듭니다."""

    storage = FakeStorage()
    root = "datasets/processed/v9-seed42-8020-group"
    train_images, train_annotations = [], []
    for index, group in enumerate(("K-1-2", "K-3-4"), start=1):
        location = f"raw/train_images/{group}_0_2_0_2_70_000_200.png"
        storage.blobs[location] = draw_image([radii, radii])
        train_images.append({"id": index, "file_name": location, "width": 160, "height": 200})
        for slot, category in enumerate((1, 2), start=1):
            train_annotations.append(
                {
                    "id": index * 10 + slot,
                    "image_id": index,
                    "category_id": category,
                    "bbox": [10.0, 10.0, radii * 2.0, radii * 2.0],
                }
            )
    validation_location = "raw/train_images/K-5-6_0_2_0_2_90_000_200.png"
    storage.blobs[validation_location] = draw_image([radii, radii])
    validation_images = [
        {"id": 9, "file_name": validation_location, "width": 160, "height": 200}
    ]
    validation_annotations = [
        {"id": 91, "image_id": 9, "category_id": 1, "bbox": [10.0, 10.0, 30.0, 30.0]},
        {"id": 92, "image_id": 9, "category_id": 2, "bbox": [50.0, 10.0, 30.0, 30.0]},
    ]
    test_location = "raw/test_images/1.png"
    storage.blobs[test_location] = draw_image([test_radii if test_radii else radii] * 2)

    storage.json[f"{root}/train_manifest.json"] = manifest(train_images, train_annotations)
    storage.json[f"{root}/validation_manifest.json"] = manifest(
        validation_images, validation_annotations
    )
    storage.json[f"{root}/class_map.json"] = {"1": "pill_a", "2": "pill_b"}
    storage.json[f"{root}/test_manifest.json"] = manifest(
        [{"id": 1, "file_name": test_location, "width": 160, "height": 200}], []
    )
    return storage


def config_for(storage: FakeStorage, **data: Any) -> dict:
    root = "datasets/processed/v9-seed42-8020-group"
    return {
        "data": {"eda": True, **data},
        "inputs": {
            "data": {
                "train_manifest_uri": f"{root}/train_manifest.json",
                "validation_manifest_uri": f"{root}/validation_manifest.json",
                "class_map_uri": f"{root}/class_map.json",
                "dataset_summary_uri": f"{root}/dataset_summary.json",
                "test_manifest_uri": f"{root}/test_manifest.json",
            }
        },
    }


def run_with(storage: FakeStorage, config: dict) -> dict:
    with patch.object(eda, "create_storage", return_value=storage):
        return run(config)


# --- 픽셀 측정 -------------------------------------------------------------


def test_measure_image_reports_the_area_the_objects_cover_and_both_colours():
    """물체가 차지한 넓이의 비율이라, 원과 넓이가 맞아야 합니다."""

    with Image.open(io.BytesIO(draw_image([18, 18]))) as image:
        measured = eda.measure_image(image, downscale=2)

    expected = 2 * 3.14159 * 18**2 / (IMAGE_SIZE[0] * IMAGE_SIZE[1])
    assert 0.8 * expected < measured["foreground_fraction"] < 1.3 * expected
    assert all(abs(value - 230) < 5 for value in measured["background_color"])
    assert all(value < 90 for value in measured["foreground_color"])


def test_measure_image_does_not_assume_which_side_is_the_background():
    """알약이 배경보다 밝은 판이 오면 전경과 배경이 통째로 뒤집힙니다."""

    image = Image.new("RGB", IMAGE_SIZE, (20, 20, 20))
    ImageDraw.Draw(image).ellipse((40, 40, 76, 76), fill=(240, 240, 240))

    measured = eda.measure_image(image, downscale=2)

    assert 0.0 < measured["foreground_fraction"] < 0.2
    assert all(value < 30 for value in measured["background_color"])


# --- 리포트 ---------------------------------------------------------------


def test_eda_writes_a_report_next_to_the_dataset_and_returns_its_uri():
    storage = build_storage()

    result = run_with(storage, config_for(storage))

    validate_pipeline_result(result, pipeline_name="data")
    assert result["status"] == "ok", result["message"]
    uri = result["artifacts"]["eda_report_uri"]
    assert uri.endswith("/eda/report.json")
    assert uri in storage.json


def test_eda_republishes_the_dataset_uris_main_pipeline_checks_for():
    """성공한 data stage가 그 URI들을 내지 않으면 main_pipeline이 멈춥니다."""

    from src.main_pipeline import _REQUIRED_ARTIFACTS, _validate_required_artifacts

    storage = build_storage()

    result = run_with(storage, config_for(storage))

    _validate_required_artifacts("data", result)
    assert set(_REQUIRED_ARTIFACTS["data"]) <= set(result["artifacts"])


def test_report_records_class_balance_and_split_leakage():
    """"학습량이 적어서"와 "split이 샜다"를 숫자로 반박할 수 있어야 합니다."""

    storage = build_storage()

    result = run_with(storage, config_for(storage))
    report = storage.json[result["artifacts"]["eda_report_uri"]]

    assert report["classes"]["class_count"] == 2
    assert report["classes"]["imbalance_ratio"] == 1.0
    assert report["combinations"]["groups_in_both_splits"] == 0
    assert report["shape"]["train"]["objects_per_image"] == {"2": 2}
    assert report["shape"]["train"]["images_with_a_repeated_class"] == 0


def test_report_flags_a_group_that_lands_in_both_splits():
    """조합 하나가 양쪽에 걸치면 검증 점수가 부풀어 오릅니다."""

    storage = build_storage()
    root = "datasets/processed/v9-seed42-8020-group"
    leaked = dict(storage.json[f"{root}/validation_manifest.json"])
    leaked["images"] = [
        {**leaked["images"][0], "file_name": "raw/train_images/K-1-2_0_2_0_2_90_000_200.png"}
    ]
    storage.json[f"{root}/validation_manifest.json"] = leaked

    result = run_with(storage, config_for(storage))
    report = storage.json[result["artifacts"]["eda_report_uri"]]

    assert report["combinations"]["groups_in_both_splits"] == 1
    assert report["combinations"]["leaked_group_sample"] == ["K-1-2"]


def test_report_compares_test_pixels_with_train_pixels_measured_the_same_way():
    """정답 bbox와 픽셀 측정을 바로 견주면 재는 방법의 차이가 섞입니다."""

    storage = build_storage(radii=20, test_radii=14)

    result = run_with(storage, config_for(storage))
    size = storage.json[result["artifacts"]["eda_report_uri"]]["object_size"]

    assert size["train_foreground_fraction"]["count"] == 2
    assert size["test_foreground_fraction"]["count"] == 1
    # 반지름 14/20이면 길이비가 0.7 근처여야 합니다.
    assert 0.6 < size["test_over_train"]["length_ratio"] < 0.8
    assert size["calibration"]["trustworthy"] is True


def test_a_size_comparison_is_withheld_when_the_ruler_fails_on_train():
    """정답을 못 맞히는 자로 잰 비율을 적어 두면 그 숫자만 인용됩니다."""

    storage = build_storage()
    root = "datasets/processed/v9-seed42-8020-group"
    # 정답 bbox를 이미지 전체로 부풀리면 픽셀 측정과 크게 어긋납니다.
    manifest_json = storage.json[f"{root}/train_manifest.json"]
    for annotation in manifest_json["annotations"]:
        annotation["bbox"] = [0.0, 0.0, 160.0, 200.0]

    result = run_with(storage, config_for(storage))
    size = storage.json[result["artifacts"]["eda_report_uri"]]["object_size"]

    assert size["calibration"]["trustworthy"] is False
    assert size["test_over_train"] is None
    # 잰 값 자체는 남습니다. 못 믿는 것은 두 값을 견준 결론뿐입니다.
    assert size["test_foreground_fraction"]["count"] == 1


def test_report_compares_the_backdrop_and_lighting_on_both_sides():
    """물체 크기가 같아도 촬영 부스가 다르면 model이 보는 그림이 다릅니다."""

    storage = build_storage()
    # test만 붉은 배경으로 찍힌 판을 흉내 냅니다.
    image = Image.new("RGB", IMAGE_SIZE, (220, 60, 60))
    ImageDraw.Draw(image).ellipse((40, 40, 76, 76), fill=(30, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    storage.blobs["raw/test_images/1.png"] = buffer.getvalue()

    result = run_with(storage, config_for(storage))
    appearance = storage.json[result["artifacts"]["eda_report_uri"]]["appearance"]

    assert appearance["train_background_color"][0] == pytest.approx(230, abs=5)
    assert appearance["test_background_color"][1] == pytest.approx(60, abs=10)
    assert appearance["background_color_distance"] > 100


def test_report_says_what_it_read_and_never_reads_test_annotations():
    """대회 규칙에 걸리는 읽기라, 무엇을 읽었는지 리포트가 들고 다녀야 합니다."""

    storage = build_storage()

    result = run_with(storage, config_for(storage))
    sources = storage.json[result["artifacts"]["eda_report_uri"]]["sources"]

    assert sources["test_annotations_read"] is False
    assert sources["test_manifest_uri"].endswith("test_manifest.json")
    assert any("test_images" in location for location in storage.downloaded)


def test_eda_without_a_test_manifest_still_reports_the_rest():
    """test manifest가 없는 예전 dataset도 나머지는 다 볼 수 있어야 합니다."""

    storage = build_storage()
    config = config_for(storage)
    del config["inputs"]["data"]["test_manifest_uri"]

    result = run_with(storage, config)
    report = storage.json[result["artifacts"]["eda_report_uri"]]

    assert result["status"] == "ok", result["message"]
    assert report["object_size"]["test_foreground_fraction"] is None
    assert report["object_size"]["train_foreground_fraction"]["count"] == 2


# --- 실패와 경계 -----------------------------------------------------------


def test_eda_refuses_to_replace_an_existing_report_unless_overwrite_is_on():
    """이 실행 전에 있던 것을 지우지 않습니다."""

    storage = build_storage()
    first = run_with(storage, config_for(storage))

    again = run_with(storage, config_for(storage))

    assert again["status"] == "error"
    assert "overwrite" in again["message"]
    assert storage.json[first["artifacts"]["eda_report_uri"]]["schema_version"]

    replaced = run_with(storage, config_for(storage, overwrite=True))
    assert replaced["status"] == "ok", replaced["message"]


def test_eda_without_a_selected_dataset_returns_an_error_result():
    """`run()` 경계 밖으로 예외가 나가면 main_pipeline이 통째로 멈춥니다."""

    result = run({"data": {"eda": True}, "inputs": {"data": {}}})

    validate_pipeline_result(result, pipeline_name="data")
    assert result["status"] == "error"
    assert "전처리 dataset" in result["message"]


@pytest.mark.parametrize("value", ["true", 1])
def test_a_non_boolean_eda_flag_is_rejected(value):
    with pytest.raises(EdaError):
        eda.eda_requested({"data": {"eda": value}})


def test_eda_off_keeps_the_existing_integration_path():
    """켜지 않은 실행은 지금까지와 완전히 같게 동작해야 합니다."""

    root = "datasets/processed/v9-seed42-8020-group"
    config = {
        "inputs": {
            "data": {
                "train_manifest_uri": f"{root}/train_manifest.json",
                "validation_manifest_uri": f"{root}/validation_manifest.json",
                "class_map_uri": f"{root}/class_map.json",
                "dataset_summary_uri": f"{root}/dataset_summary.json",
            }
        }
    }

    result = run(config)

    assert result["summary"]["mode"] == "integration"
    assert "eda_report_uri" not in result["artifacts"]
