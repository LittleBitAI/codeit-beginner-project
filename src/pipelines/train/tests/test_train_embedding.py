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


class LocalReader:
    """crop 은행 tar을 그 자리에서 복사해 주는 최소 storage입니다."""

    def download_file(self, source, destination, **_):
        Path(destination).write_bytes(Path(source).read_bytes())


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """저장소 root를 임시 directory로 옮긴 작업 공간입니다.

    `output_dir`은 저장소 기준 **상대 경로**만 받습니다(절대 경로는 저장소 밖에
    checkpoint를 쓰고 사용자 이름이 든 URI를 냅니다). 그래서 test도 실제 실행과
    같은 모양으로, root를 옮기고 상대 경로를 줍니다.
    """

    build_bank(tmp_path / "crop_bank.tar")
    (tmp_path / "class_map.json").write_text(
        json.dumps({"11": "pill-a", "22": "pill-b"}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(embedding_module, "REPOSITORY_ROOT", tmp_path)
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
        "output_dir": "working",
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


def test_local_artifact_uris_are_repository_relative(workspace: Path):
    """절대 경로를 내보내면 세 가지가 어긋납니다.

    다른 컴퓨터에서 열 수 없고, 저장소 규칙을 깨고, 화면까지 흘러가면 OS 사용자
    이름이 드러납니다. detector도 같은 규칙으로 상대 경로를 냅니다.
    """

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


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.jpg",
        # 푸는 자리가 `out`일 때 `out-evil`은 **문자열 앞자리가 같습니다.**
        # startswith로 경계를 보면 이것이 통과합니다.
        "../out-evil/escaped.jpg",
    ],
)
def test_a_bank_that_escapes_its_directory_is_refused(tmp_path: Path, name: str):
    """남이 만든 tar를 그대로 푸는 습관을 남기지 않습니다."""

    archive = tmp_path / "crop_bank.tar"
    with tarfile.open(archive, "w") as opened:
        payload = b"x"
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        opened.addfile(info, io.BytesIO(payload))

    with pytest.raises(EmbeddingTrainingError, match="폴더 밖을 가리키는 경로"):
        read_crop_bank(LocalReader(), str(archive), tmp_path / "out")


def test_a_bank_member_that_is_a_link_is_refused(tmp_path: Path):
    """이름만 봐서는 모자랍니다. symlink는 **가리키는 곳**으로 나갑니다.

    이름은 폴더 안이라 앞자리 검사와 relative_to를 모두 통과하지만, 푸는 순간
    link가 되어 그 뒤 쓰기가 밖으로 나갑니다.
    """

    archive = tmp_path / "crop_bank.tar"
    with tarfile.open(archive, "w") as opened:
        info = tarfile.TarInfo("crops/looks-fine.jpg")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside.jpg"
        opened.addfile(info)

    with pytest.raises(EmbeddingTrainingError, match="보통 파일이 아닌 항목"):
        read_crop_bank(LocalReader(), str(archive), tmp_path / "out")


def test_a_bank_listing_that_points_outside_is_refused(tmp_path: Path):
    """tar을 안전하게 풀어도 **목록**이 다른 곳을 가리킬 수 있습니다.

    여는 것은 학습 중이라, 여기서 안 막으면 저장소 밖 파일을 batch마다 읽습니다.
    """

    archive = tmp_path / "crop_bank.tar"
    with tarfile.open(archive, "w") as opened:
        payload = crop_bytes((10, 20, 30))
        info = tarfile.TarInfo("crops/11/0.jpg")
        info.size = len(payload)
        opened.addfile(info, io.BytesIO(payload))
        listing = json.dumps(
            {"records": [{"path": "../../secret.jpg", "category_id": 11}]},
            ensure_ascii=False,
        ).encode()
        info = tarfile.TarInfo("index.json")
        info.size = len(listing)
        opened.addfile(info, io.BytesIO(listing))

    with pytest.raises(EmbeddingTrainingError, match="폴더 밖을 가리키는 경로"):
        read_crop_bank(LocalReader(), str(archive), tmp_path / "out")


# --- 안전장치 ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["C:/Windows/Temp", "/tmp/embeddings", "../outside", "artifacts/../../outside"]
)
def test_an_output_dir_outside_the_repository_is_refused(workspace: Path, value: str):
    """저장소 밖에 checkpoint를 쓰면 그 경로가 그대로 artifact URI가 됩니다."""

    result = run(embedding_config(workspace, output_dir=value))

    assert result["status"] == "error"
    assert "output_dir" in result["message"]


def test_an_interrupted_run_of_the_same_name_is_not_overwritten(workspace: Path):
    """중단된 학습의 checkpoint는 그 학습의 **유일한 사본**입니다.

    같은 run_id로 다시 돌릴 때 그 자리를 이어 쓰면, 다시 돌린 사람이 앞선 밤을
    지웁니다. 지울지 말지는 사람이 정합니다.
    """

    partial = workspace / "working" / ".embed-test.partial"
    partial.mkdir(parents=True)
    survivor = partial / "best_checkpoint.pt"
    only_copy = "앞선 학습의 유일한 사본".encode("utf-8")
    survivor.write_bytes(only_copy)

    result = run(embedding_config(workspace))

    assert result["status"] == "error"
    assert "중단된 학습" in result["message"]
    assert survivor.read_bytes() == only_copy


def test_the_best_checkpoint_is_the_best_epoch_not_a_kept_one(workspace: Path):
    """`best`를 저장 주기 안에 두면 주기 사이의 가장 좋은 epoch을 놓칩니다.

    그러면 더 나쁜 model이 `best_checkpoint.pt`라는 이름으로 나가고, 쓰는 쪽은
    그 이름을 믿습니다.
    """

    result = run(embedding_config(workspace, epochs=3, checkpoint_every=3))

    assert result["status"] == "ok", result["message"]
    history = json.loads(
        artifact_path(result["artifacts"]["training_history_uri"]).read_text(encoding="utf-8")
    )
    accuracies = [entry["train_accuracy"] for entry in history["epochs"]]
    best = torch.load(
        artifact_path(result["artifacts"]["best_checkpoint_uri"]), map_location="cpu"
    )
    # history는 6자리로 반올림해 적고 checkpoint는 그대로 담습니다.
    assert round(best["train_accuracy"], 6) == max(accuracies)
    assert best["epoch"] == history["epochs"][accuracies.index(max(accuracies))]["epoch"]
    # 주기(3)에 걸리지 않는 epoch이 가장 좋아야 이 test가 무언가를 지킵니다.
    assert best["epoch"] % 3 != 0, "주기와 겹쳐 이 test가 아무것도 구별하지 못합니다"


@pytest.mark.parametrize("value", ["embed-test\n", "../escape", "-starts-with-dash", ""])
def test_a_run_id_off_the_name_rule_is_refused(value: str):
    """이름은 그대로 경로가 됩니다.

    `settings()`를 직접 부릅니다. 전체 실행으로 재면 OS가 대신 막아 주는 경우와
    구별되지 않습니다 — Windows는 파일 이름의 줄바꿈을 거부하지만 Linux는 받습니다.
    그러면 이 저장소의 CI에서만 통과하는 test가 됩니다.
    """

    with pytest.raises(EmbeddingTrainingError, match="run_id"):
        settings({"train": {"task": "embedding", "run_id": value}})


def test_a_class_map_without_the_bank_classes_is_refused(workspace: Path):
    """필수 입력을 읽지 않으면 다른 dataset의 class map을 붙여도 그냥 성공합니다."""

    (workspace / "class_map.json").write_text(
        json.dumps({"77": "다른 dataset의 알약"}, ensure_ascii=False), encoding="utf-8"
    )

    result = run(embedding_config(workspace))

    assert result["status"] == "error"
    assert "class map" in result["message"]


def test_the_same_seed_gives_the_same_weights(workspace: Path):
    """seed가 같으면 결과가 같아야 합니다. 아니면 어떤 비교도 근거가 없습니다."""

    first = run(embedding_config(workspace, run_id="seed-a"))
    second = run(embedding_config(workspace, run_id="seed-b"))

    assert first["status"] == "ok", first["message"]
    assert second["status"] == "ok", second["message"]
    left = torch.load(
        artifact_path(first["artifacts"]["best_checkpoint_uri"]), map_location="cpu"
    )["state_dict"]
    right = torch.load(
        artifact_path(second["artifacts"]["best_checkpoint_uri"]), map_location="cpu"
    )["state_dict"]
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key
