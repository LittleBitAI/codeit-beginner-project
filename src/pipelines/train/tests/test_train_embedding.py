"""임베딩 학습: 잘라 낸 알약 crop 하나가 어떤 class인지 재는 자를 만듭니다.

검출기와 달리 manifest가 아니라 crop 은행을 읽고, 설정 칸도 다릅니다. 그 둘이
섞이지 않는지와, 만든 checkpoint 하나로 model을 되살릴 수 있는지를 봅니다.
CPU에서 아주 작은 model을 돌리므로 GPU도 AWS도 필요 없습니다.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from src.common import train_contract
from src.pipelines.train import run
from src.pipelines.train import embedding as embedding_module
from src.pipelines.train.embedding import (
    CROP_SIZE,
    EmbeddingTrainingError,
    read_crop_bank,
    settings,
)


def crop_bytes(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (CROP_SIZE, CROP_SIZE), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def build_bank(path: Path, *, categories: tuple[int, ...] = (11, 22), per_class: int = 3):
    """data가 만드는 것과 같은 모양의 crop 은행 tar를 만듭니다."""

    records = []
    with tarfile.open(path, "w") as archive:
        for index, category_id in enumerate(categories):
            for number in range(per_class):
                name = f"crops/{category_id}/{index}_{number}.jpg"
                payload = crop_bytes((40 * (index + 1), 60, 90))
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
                records.append(
                    {
                        "path": name,
                        "category_id": category_id,
                        "image_id": index * 10 + number,
                        "group": f"g{index}",
                    }
                )
        payload = json.dumps({"version": 1, "records": records}, ensure_ascii=False).encode()
        info = tarfile.TarInfo("index.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return records


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    build_bank(tmp_path / "crop_bank.tar")
    return tmp_path


def artifact_path(uri: str) -> Path:
    """계약이 정한 저장소 기준 상대 경로를 실제 file로 바꿉니다.

    읽는 쪽(`evaluate/storage_io.py`)도 같은 기준으로 풉니다.
    """

    return embedding_module.REPOSITORY_ROOT / uri


def embedding_config(root: Path, **extra: Any) -> dict[str, Any]:
    train_section: dict[str, Any] = {
        "task": "embedding",
        "run_id": "embed-test",
        "epochs": 1,
        "batch_size": 2,
        "pretrained": False,
        "output_dir": str(root / "working"),
        "output_prefix": "embeddings",
    }
    train_section.update(extra)
    return {
        "execution": {"mode": "real"},
        "storage": {"backend": "local", "local": {"root": str(root)}},
        "train": train_section,
        "inputs": {
            "data": {
                "crop_bank_uri": "crop_bank.tar",
                "class_map_uri": "class_map.json",
            }
        },
    }


# --- 계약 ------------------------------------------------------------------


def test_embedding_reads_exactly_the_setting_names_in_the_shared_contract():
    """GUI가 그 이름으로 값을 실어 보냅니다. 여기가 그것을 정말 읽는 쪽입니다.

    detector 쪽과 같은 이유입니다. web은 이 파일을 import할 수 없어 이름을 옮겨 적을
    뿐이라, 한쪽이 이름을 바꾸며 자기 test까지 고치면 양쪽 다 초록인 채로 그 값이
    조용히 버려집니다.
    """

    sent = {
        "task": "embedding",
        "run_id": "keys",
        "backbone": "resnet34",
        "batch_size": 8,
        "checkpoint_every": 2,
        "device": "cpu",
        "epochs": 3,
        "learning_rate": 1e-3,
        "num_workers": 0,
        "output_dir": "artifacts/other",
        "output_prefix": "other",
        "pretrained": False,
        "seed": 7,
        "weight_decay": 5e-4,
    }
    assert set(sent) == set(train_contract.EMBEDDING_SETTING_KEYS)

    read = settings({"train": sent})

    for name, value in sent.items():
        if name == "task":
            continue
        assert getattr(read, name) == value, f"{name}이 그대로 오지 않았습니다."


def test_detector_settings_are_refused_not_ignored():
    """쓰지 않는 칸을 조용히 버리면 사람은 그 값이 반영된 줄 압니다."""

    with pytest.raises(EmbeddingTrainingError, match="architecture"):
        settings({"train": {"task": "embedding", "run_id": "x", "architecture": "retinanet_resnet50_fpn_v2"}})


def test_detector_training_is_untouched_without_a_task():
    """`task`를 보내지 않던 기존 실행은 지금까지처럼 detector로 갑니다."""

    result = run({"execution": {"mode": "real"}, "train": {"run_id": "x"}, "inputs": {}})

    assert result["status"] == "error"
    # detector 경로가 자기 입력을 요구하며 거절한 것이어야 합니다.
    assert "crop_bank_uri" not in result["message"]


# --- 학습 ------------------------------------------------------------------


def test_embedding_training_writes_two_checkpoints_and_a_history(workspace: Path):
    result = run(embedding_config(workspace))

    assert result["status"] == "ok", result["message"]
    assert set(result["artifacts"]) == {
        "run_id",
        "best_checkpoint_uri",
        "last_checkpoint_uri",
        "training_history_uri",
    }
    history = json.loads(
        artifact_path(result["artifacts"]["training_history_uri"]).read_text(encoding="utf-8")
    )
    assert history["task"] == "embedding"
    assert len(history["epochs"]) == 1
    assert result["summary"]["class_count"] == 2


def test_local_artifact_uris_are_repository_relative(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    """절대 경로를 내보내면 세 가지가 어긋납니다.

    다른 컴퓨터에서 열 수 없고, 저장소 규칙을 깨고, 화면까지 흘러가면 OS 사용자
    이름이 드러납니다. detector도 같은 규칙으로 상대 경로를 냅니다.
    """

    monkeypatch.setattr(embedding_module, "REPOSITORY_ROOT", workspace)

    result = run(embedding_config(workspace))

    assert result["status"] == "ok", result["message"]
    for key in ("best_checkpoint_uri", "last_checkpoint_uri", "training_history_uri"):
        uri = result["artifacts"][key]
        assert not Path(uri).is_absolute(), uri
        assert "\\" not in uri, uri
        assert artifact_path(uri).is_file(), uri


def test_the_checkpoint_alone_can_rebuild_the_model(workspace: Path):
    """쓰는 쪽은 이 파일 하나만 받습니다. backbone과 class 순서가 없으면 못 되살립니다."""

    result = run(embedding_config(workspace, backbone="resnet18"))

    payload = torch.load(
        artifact_path(result["artifacts"]["best_checkpoint_uri"]), map_location="cpu"
    )
    assert payload["backbone"] == "resnet18"
    assert payload["category_ids"] == [11, 22]
    assert payload["crop_size"] == CROP_SIZE
    assert payload["normalisation"]["mean"] and payload["normalisation"]["std"]
    from src.pipelines.train.embedding import build_model

    rebuilt = build_model(payload["backbone"], len(payload["category_ids"]), pretrained=False)
    rebuilt.load_state_dict(payload["state_dict"])


def test_published_files_are_never_overwritten(workspace: Path):
    """같은 run_id로 다시 돌리면 앞선 결과를 덮지 않고 멈춰야 합니다."""

    assert run(embedding_config(workspace))["status"] == "ok"

    again = run(embedding_config(workspace))

    assert again["status"] == "error"


# --- crop 은행 읽기 ---------------------------------------------------------


def test_a_bank_that_escapes_its_directory_is_refused(tmp_path: Path):
    """남이 만든 tar를 그대로 푸는 습관을 남기지 않습니다."""

    archive = tmp_path / "crop_bank.tar"
    with tarfile.open(archive, "w") as opened:
        payload = b"x"
        info = tarfile.TarInfo("../escaped.jpg")
        info.size = len(payload)
        opened.addfile(info, io.BytesIO(payload))

    class Local:
        def download_file(self, source, destination, **_):
            Path(destination).write_bytes(Path(source).read_bytes())

    with pytest.raises(EmbeddingTrainingError, match="위험한 경로"):
        read_crop_bank(Local(), str(archive), tmp_path / "out")
