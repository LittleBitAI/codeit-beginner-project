"""crop embedding 재순위 test입니다.

model을 학습시키지 않아도 결과를 단정할 수 있게, 시험 이미지와 참조 crop을
**같은 단색**으로 만듭니다. 같은 그림이면 어떤 가중치를 써도 특징이 똑같아
코사인 유사도가 정확히 1이므로, "자기 class 쪽이 더 가까운가"를 학습 없이 그대로
잴 수 있습니다.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from src.pipelines.evaluate.pipeline import run

from conftest import write_json


torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")


#: 시험 crop과 참조 crop을 224px로 다루면 test 하나가 몇 초씩 걸립니다. crop 크기를
#: 은행에서 읽는지도 함께 확인하려고 일부러 작은 값을 씁니다.
CROP_SIZE = 32
RED = (220, 30, 30)
BLUE = (30, 30, 220)


def _write_image(path: Path, colour: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), colour).save(path)


def _crop_bytes(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (CROP_SIZE, CROP_SIZE), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_crop_bank(
    path: Path,
    colours: dict[int, tuple[int, int, int]],
    *,
    per_class: int = 2,
    escaping: bool = False,
) -> str:
    """data가 만드는 것과 같은 모양의 crop 은행 tar을 만듭니다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    payloads: dict[str, bytes] = {}
    for category_id, colour in colours.items():
        for index in range(per_class):
            name = f"crops/{category_id}/{index}.png"
            payloads[name] = _crop_bytes(colour)
            records.append({"path": name, "category_id": category_id})
    document = {
        "version": 1,
        "crop_size": CROP_SIZE,
        "crop_margin": 0.08,
        "per_class": per_class,
        "seed": 42,
        "records": records,
    }
    with tarfile.open(path, "w") as archive:
        index = json.dumps(document, ensure_ascii=False).encode("utf-8")
        info = tarfile.TarInfo("index.json")
        info.size = len(index)
        archive.addfile(info, io.BytesIO(index))
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if escaping:
            escape = b"stolen"
            info = tarfile.TarInfo("../escaped.txt")
            info.size = len(escape)
            archive.addfile(info, io.BytesIO(escape))
    return str(path)


def _write_embedding_checkpoint(path: Path, category_ids: list[int], *, seed: int = 0) -> None:
    torch.manual_seed(seed)
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(category_ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": "embedding",
            "backbone": "resnet18",
            "category_ids": list(category_ids),
            "crop_size": CROP_SIZE,
            "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "state_dict": model.state_dict(),
        },
        path,
    )


def _prepare(base_config: dict, repository_root: Path, *, category_ids: tuple[int, int]) -> None:
    """시험지, 사진, 합칠 예측 둘을 갖춘 test 경로 실행을 만듭니다.

    checkpoint 없이 합친 예측으로 제출을 만드는 길을 씁니다. detector 추론을 하지
    않아도 재순위가 무엇을 하는지 그대로 보입니다.
    """

    write_json(
        repository_root / "data/test/instances.json",
        {
            "images": [
                {"id": 10, "file_name": "0010.png", "width": 100, "height": 100},
                {"id": 20, "file_name": "0020.png", "width": 100, "height": 100},
            ],
            "annotations": [],
            "categories": [{"id": 3, "name": "pill-a"}, {"id": 7, "name": "pill-b"}],
        },
    )
    base_config["inputs"]["data"]["test_manifest_uri"] = "data/test/instances.json"
    base_config["inputs"]["train"].pop("best_checkpoint_uri")
    _write_image(repository_root / "data/test/0010.png", RED)
    _write_image(repository_root / "data/test/0020.png", BLUE)

    for name in ("a", "b"):
        write_json(
            repository_root / f"data/test/{name}.json",
            {
                "test_manifest_uri": "data/test/instances.json",
                "checkpoint_uri": f"checkpoints/{name}.pt",
                "predictions": [
                    {
                        "image_id": 10,
                        "category_id": category_ids[0],
                        "bbox": [10.0, 10.0, 40.0, 40.0],
                        "score": 0.9,
                    },
                    {
                        "image_id": 20,
                        "category_id": category_ids[1],
                        "bbox": [10.0, 10.0, 40.0, 40.0],
                        "score": 0.9,
                    },
                ],
            },
        )
    base_config["evaluate"]["test_predictions_input_uris"] = [
        "data/test/a.json",
        "data/test/b.json",
    ]


def _submission_scores(repository_root: Path, result: dict) -> dict[int, float]:
    rows = (
        (repository_root / result["artifacts"]["submission_uri"])
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    )
    return {int(row.split(",")[1]): float(row.split(",")[7]) for row in rows}


def test_submission_keeps_detector_scores_unless_rerank_is_asked(
    base_config: dict, repository_root: Path
):
    """설정하지 않으면 지금까지와 똑같습니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    assert _submission_scores(repository_root, result) == {10: 0.9, 20: 0.9}
    document = json.loads(
        (repository_root / result["artifacts"]["test_predictions_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert document["rerank"] is None


def test_rerank_pushes_down_the_row_whose_class_looks_wrong(
    base_config: dict, repository_root: Path
):
    """같은 class 안에서 순서를 바꿉니다. 이것이 AP를 움직이는 유일한 길입니다.

    두 행 모두 category 7이라고 나왔지만 사진은 하나만 7의 색입니다. 잘못 붙은
    쪽은 자기 class보다 다른 class 쪽이 더 가까우므로 margin이 음수가 되어 점수가
    절반 아래로 내려가고, 맞은 쪽은 절반 위에 남습니다.
    """

    _prepare(base_config, repository_root, category_ids=(7, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE})
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    scores = _submission_scores(repository_root, result)
    # image 10은 빨강인데 7(파랑)이라고 나왔습니다. image 20은 제 색입니다.
    assert scores[10] < scores[20]
    assert scores[10] <= 0.45 <= scores[20]

    document = json.loads(
        (repository_root / result["artifacts"]["test_predictions_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert document["rerank"]["reranked_rows"] == 2
    assert document["rerank"]["reference_crops"] == 4
    assert document["rerank"]["negative_margin_rows"] == 1
    assert document["rerank"]["checkpoints"] == ["checkpoints/embedding.pt"]


def test_rerank_leaves_a_class_the_bank_never_saw(base_config: dict, repository_root: Path):
    """참조 crop이 없는 class는 재지 못합니다. 못 잰 것을 0으로 두면 점수가 반이 됩니다."""

    # 은행은 3과 5를 압니다. 7은 한 장도 없습니다.
    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 5: BLUE})
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 5])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    result = run(base_config)

    assert result["status"] == "ok", result["message"]
    scores = _submission_scores(repository_root, result)
    # 은행에 없는 category 7은 detector 점수 그대로입니다.
    assert scores[20] == pytest.approx(0.9)
    assert scores[10] != pytest.approx(0.9)
    document = json.loads(
        (repository_root / result["artifacts"]["test_predictions_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert document["rerank"]["reranked_rows"] == 1
    assert document["rerank"]["rows"] == 2


def test_rerank_refuses_a_detector_checkpoint(base_config: dict, repository_root: Path):
    """detector checkpoint를 주면 조용히 건너뛰지 않고 멈춥니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE})
    path = repository_root / "checkpoints/detector.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"architecture": "fasterrcnn_resnet50_fpn", "num_classes": 3, "state_dict": {}}, path
    )
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/detector.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    result = run(base_config)

    assert result["status"] == "error"
    assert "train.task=embedding" in result["message"]


def test_rerank_needs_reference_crops(base_config: dict, repository_root: Path):
    _prepare(base_config, repository_root, category_ids=(3, 7))
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]

    result = run(base_config)

    assert result["status"] == "error"
    assert "재순위에는 참조 crop이 필요합니다" in result["message"]


def test_rerank_refuses_the_same_checkpoint_twice(base_config: dict, repository_root: Path):
    """같은 model을 두 번 넣으면 평균이 조용히 그쪽으로 기웁니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))
    base_config["evaluate"]["rerank_checkpoint_uris"] = [
        "checkpoints/embedding.pt",
        "checkpoints/embedding.pt",
    ]
    base_config["evaluate"]["rerank_crop_bank_uri"] = "data/test/crop_bank.tar"

    result = run(base_config)

    assert result["status"] == "error"
    assert "같은 checkpoint를 두 번" in result["message"]


def test_rerank_refuses_a_crop_bank_that_reaches_outside_its_folder(
    base_config: dict, repository_root: Path
):
    """tar 안의 `../`는 푸는 쪽 file을 덮어씁니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(
        repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE}, escaping=True
    )
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    result = run(base_config)

    assert result["status"] == "error"
    assert "폴더 밖을 가리키는 항목" in result["message"]


def test_fusion_refuses_predictions_whose_scores_were_already_reranked(
    base_config: dict, repository_root: Path
):
    """재순위한 예측을 다시 합치면 한 파일만 다른 자로 잰 점수가 됩니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))
    reranked = json.loads(
        (repository_root / "data/test/a.json").read_text(encoding="utf-8")
    )
    reranked["rerank"] = {"rows": 2, "reranked_rows": 2, "checkpoints": ["checkpoints/e.pt"]}
    write_json(repository_root / "data/test/a.json", reranked)

    result = run(base_config)

    assert result["status"] == "error"
    assert "점수를 다시 매긴 예측은 합칠 수 없습니다" in result["message"]
