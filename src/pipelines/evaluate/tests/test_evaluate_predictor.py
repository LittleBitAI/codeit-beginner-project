"""예측 문서 검증과 checkpoint 추론 test입니다."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipelines.evaluate.errors import InputArtifactError, PredictionError
from src.pipelines.evaluate.predictor import (
    load_checkpoint_document,
    parse_predictions,
    predict_with_checkpoint,
)
from src.pipelines.evaluate.storage_io import ArtifactStore


KNOWN_KEYS = {"img-1"}
VALID_PREDICTION = {
    "image_id": "img-1",
    "category_id": 1,
    "bbox": [10, 10, 20, 20],
    "score": 0.9,
}


def test_parse_predictions_accepts_list_and_object():
    from_list = parse_predictions([VALID_PREDICTION], source="p.json", known_image_keys=KNOWN_KEYS)
    from_object = parse_predictions(
        {"predictions": [VALID_PREDICTION]}, source="p.json", known_image_keys=KNOWN_KEYS
    )

    assert from_list == from_object
    assert from_list[0]["image_key"] == "img-1"
    assert from_list[0]["bbox"] == [10.0, 10.0, 20.0, 20.0]


def test_parse_predictions_rejects_unknown_image_id():
    entry = dict(VALID_PREDICTION, image_id="img-999")

    with pytest.raises(InputArtifactError, match="manifest에 없는 image_id"):
        parse_predictions([entry], source="p.json", known_image_keys=KNOWN_KEYS)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"score": "0.9"}, "score는 유한한 숫자"),
        ({"category_id": "1"}, "category_id는 정수"),
        ({"bbox": [10, 10, 20]}, "bbox는"),
        ({"bbox": [10, 10, -1, 20]}, "0보다 커야"),
    ],
)
def test_parse_predictions_rejects_invalid_entries(override, message):
    entry = dict(VALID_PREDICTION, **override)

    with pytest.raises(InputArtifactError, match=message):
        parse_predictions([entry], source="p.json", known_image_keys=KNOWN_KEYS)


def test_parse_predictions_rejects_missing_field():
    entry = {key: value for key, value in VALID_PREDICTION.items() if key != "bbox"}

    with pytest.raises(InputArtifactError, match="필수 field가 없습니다"):
        parse_predictions([entry], source="p.json", known_image_keys=KNOWN_KEYS)


def test_parse_predictions_rejects_invalid_document_type():
    with pytest.raises(InputArtifactError, match="list 또는 object"):
        parse_predictions("predictions", source="p.json", known_image_keys=KNOWN_KEYS)


def test_load_checkpoint_document_reports_missing_file(repository_root: Path):
    store = ArtifactStore()

    with pytest.raises(InputArtifactError, match="artifact가 없습니다"):
        load_checkpoint_document(store, "checkpoints/best.pt", device="cpu")


def test_load_checkpoint_document_reports_broken_file(repository_root: Path):
    checkpoint_path = repository_root / "checkpoints/best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"not a torch checkpoint")
    store = ArtifactStore()

    with pytest.raises(PredictionError, match="checkpoint를 읽지 못했습니다"):
        load_checkpoint_document(store, "checkpoints/best.pt", device="cpu")


def test_predict_with_checkpoint_reports_unknown_architecture(repository_root: Path):
    torch = pytest.importorskip("torch")
    checkpoint_path = repository_root / "checkpoints/best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"architecture": "no_such_detector", "num_classes": 3, "state_dict": {}},
        checkpoint_path,
    )
    store = ArtifactStore()

    with pytest.raises(PredictionError, match="제공하지 않는 architecture"):
        predict_with_checkpoint(store, [], checkpoint_uri="checkpoints/best.pt")


def test_predict_with_checkpoint_reports_missing_state_dict(repository_root: Path):
    torch = pytest.importorskip("torch")
    checkpoint_path = repository_root / "checkpoints/best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": "fasterrcnn_resnet50_fpn", "num_classes": 3}, checkpoint_path)
    store = ArtifactStore()

    with pytest.raises(PredictionError, match="state_dict가 필요합니다"):
        predict_with_checkpoint(store, [], checkpoint_uri="checkpoints/best.pt")


def test_predict_with_checkpoint_runs_on_cpu(repository_root: Path):
    """작은 torchvision detector로 CPU smoke test를 수행합니다."""
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    Image = pytest.importorskip("PIL.Image")

    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None, num_classes=3
    )
    checkpoint_path = repository_root / "checkpoints/best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
            "num_classes": 3,
            "category_ids": [0, 1, 2],
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    image_path = repository_root / "data/val/img-1.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(image_path)

    records = [
        {
            "image_id": "img-1",
            "image_key": "img-1",
            "image_uri": "data/val/img-1.jpg",
            "width": 64,
            "height": 64,
            "annotations": [],
        }
    ]

    predictions = predict_with_checkpoint(
        ArtifactStore(), records, checkpoint_uri="checkpoints/best.pt", device="cpu", seed=7
    )

    assert isinstance(predictions, list)
    for prediction in predictions:
        assert prediction["image_id"] == "img-1"
        assert prediction["category_id"] in {0, 1, 2}
        assert len(prediction["bbox"]) == 4
        assert 0.0 <= prediction["score"] <= 1.0
