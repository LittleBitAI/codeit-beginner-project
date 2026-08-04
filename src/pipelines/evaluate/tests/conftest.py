"""Evaluate pipeline test에서 사용하는 계약 형식 fixture입니다.

이전 pipeline(data, train)이 없어도 평가를 실행할 수 있도록, 계약과 같은 모양의
작은 manifest, class map, 예측 file을 임시 directory에 만듭니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.pipelines.evaluate import pipeline, storage_io  # noqa: E402


IMAGE_RECORDS = [
    {
        "image_id": "img-1",
        "image_uri": "data/val/img-1.jpg",
        "width": 100,
        "height": 100,
        "annotations": [
            {"category_id": 1, "bbox": [10, 10, 20, 20]},
            {"category_id": 2, "bbox": [50, 50, 20, 20]},
        ],
    },
    {
        "image_id": "img-2",
        "image_uri": "data/val/img-2.jpg",
        "width": 100,
        "height": 100,
        "annotations": [{"category_id": 1, "bbox": [30, 30, 10, 10]}],
    },
]

PERFECT_PREDICTIONS = [
    {"image_id": "img-1", "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.95},
    {"image_id": "img-1", "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.90},
    {"image_id": "img-2", "category_id": 1, "bbox": [30, 30, 10, 10], "score": 0.80},
]


def write_text(path: Path, text: str) -> Path:
    """UTF-8 without BOM, LF로 test file을 만듭니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_jsonl(path: Path, records: list[dict]) -> Path:
    lines = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    return write_text(path, lines)


def write_json(path: Path, document: object) -> Path:
    return write_text(path, json.dumps(document, ensure_ascii=False, indent=2) + "\n")


@pytest.fixture
def repository_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Local artifact 경로가 임시 directory 기준 상대 경로가 되도록 root를 바꿉니다."""
    monkeypatch.setattr(storage_io, "REPOSITORY_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def manifest_uri(repository_root: Path) -> str:
    write_jsonl(repository_root / "data/val/manifest.jsonl", IMAGE_RECORDS)
    return "data/val/manifest.jsonl"


@pytest.fixture
def class_map_uri(repository_root: Path) -> str:
    write_json(
        repository_root / "data/val/class_map.json",
        {"classes": [{"id": 1, "name": "tylenol"}, {"id": 2, "name": "aspirin"}]},
    )
    return "data/val/class_map.json"


@pytest.fixture
def predictions_uri(repository_root: Path) -> str:
    write_json(repository_root / "data/val/predictions.json", PERFECT_PREDICTIONS)
    return "data/val/predictions.json"


@pytest.fixture
def base_config(manifest_uri: str, class_map_uri: str, predictions_uri: str) -> dict:
    """계약 형식의 inputs와 evaluate 설정을 가진 기본 config입니다."""
    return {
        "inputs": {
            "data": {
                "train_manifest_uri": "data/train/manifest.jsonl",
                "validation_manifest_uri": manifest_uri,
                "class_map_uri": class_map_uri,
                "dataset_summary_uri": "data/dataset_summary.json",
            },
            "train": {
                "run_id": "train-0001",
                "best_checkpoint_uri": "checkpoints/best.pt",
                "last_checkpoint_uri": "checkpoints/last.pt",
                "training_history_uri": "checkpoints/history.json",
            },
        },
        "evaluate": {
            "run_id": "evaluate-0001",
            "predictions_input_uri": predictions_uri,
            "seed": 7,
        },
    }


@pytest.fixture
def run_evaluate():
    """Evaluate pipeline의 공개 함수를 그대로 호출합니다."""
    return pipeline.run
