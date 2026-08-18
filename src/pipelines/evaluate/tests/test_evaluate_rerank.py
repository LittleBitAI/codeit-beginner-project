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

from src.pipelines.evaluate import rerank as rerank_module
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
    escaping: str | None = None,
    linking: bool = False,
    listing_escapes: bool = False,
    overrides: dict[str, object] | None = None,
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
    if listing_escapes:
        # tar은 멀쩡한데 **목록만** 밖을 가리키는 경우입니다.
        records.append({"path": "../../secret.png", "category_id": 3})
    document = {
        "version": 1,
        "crop_size": CROP_SIZE,
        "crop_margin": 0.08,
        "per_class": per_class,
        "seed": 42,
        "records": records,
    }
    document.update(overrides or {})
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
            info = tarfile.TarInfo(escaping)
            info.size = len(escape)
            archive.addfile(info, io.BytesIO(escape))
        if linking:
            # 이름은 폴더 안이라 경계 검사를 통과하지만, 푸는 순간 link가 되어
            # 그 뒤 쓰기가 밖으로 나갑니다.
            info = tarfile.TarInfo("crops/looks-fine.png")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside.png"
            archive.addfile(info)
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


@pytest.mark.parametrize(
    ("margin", "expected"),
    [
        # margin이 1을 넘으면 점수는 **올라가야** 합니다. 1에서 자르면 가장 확신이
        # 센 행이 확신이 보통인 행과 같은 대우를 받습니다.
        (1.5, 0.9 * 1.25),
        (0.5, 0.9 * 0.75),
        # 아래로도 자르지 않습니다. 0에서 막으면 "확실히 다른 class"인 행들이 전부
        # 같은 점수로 뭉쳐 그 사이 순서가 사라지는데, 채점은 순서로 합니다.
        (-1.5, 0.9 * -0.25),
    ],
)
def test_the_multiplier_follows_the_formula(
    base_config: dict, repository_root: Path, margin: float, expected: float
):
    """식은 `(1 + margin) / 2`입니다. 위로 자르지 않습니다.

    margin은 코사인 유사도의 차이라 2까지 갈 수 있습니다. 실제 은행으로는 1을 넘는
    값을 만들기 어려워, 여기서는 margin 계산만 대신하고 **곱하는 규칙만** 잽니다.
    """

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE})
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank
    original = rerank_module._margins
    rerank_module._margins = lambda similarity, **kwargs: (
        torch.full((similarity.shape[0],), margin),
        torch.ones(similarity.shape[0], dtype=torch.bool),
    )
    try:
        result = run(base_config)
    finally:
        rerank_module._margins = original

    assert result["status"] == "ok", result["message"]
    scores = _submission_scores(repository_root, result)
    assert scores[10] == pytest.approx(expected, rel=1e-6)
    assert scores[20] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        ({"crop_size": None}, "crop 크기를 적어 두지 않았습니다"),
        ({"crop_size": "224"}, "crop 크기를 적어 두지 않았습니다"),
        ({"normalisation": None}, "normalisation이 필요합니다"),
        ({"normalisation": {"mean": [0.5, 0.5, 0.5], "std": [0.0, 0.2, 0.2]}}, "0보다 커야"),
        # 음수 std는 그 channel의 방향을 뒤집어, 오류 없이 특징만 틀어집니다.
        ({"normalisation": {"mean": [0.5, 0.5, 0.5], "std": [-0.2, 0.2, 0.2]}}, "0보다 커야"),
        # python의 정수는 크기 제한이 없습니다. 검사가 스스로 OverflowError를 내면
        # 거절하려던 값 때문에 `run()` 밖으로 예외가 나갑니다.
        (
            {"normalisation": {"mean": [10**400, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}},
            "유한한 숫자 셋",
        ),
        (
            {"normalisation": {"mean": [0.5, 0.5], "std": [0.2, 0.2, 0.2]}},
            "유한한 숫자 셋",
        ),
    ],
)
def test_a_checkpoint_that_cannot_say_how_it_was_trained_is_refused(
    base_config: dict, repository_root: Path, broken: dict, expected: str
):
    """못 쓰는 값을 그냥 지나가면 재순위가 한 행도 못 바꾸고 성공으로 끝납니다.

    std에 0이 있으면 특징이 통째로 `nan`이 되고, 못 잰 margin은 건너뛰도록 되어
    있으므로 **아무 일도 일어나지 않은 채** 성공합니다. crop 크기를 확인하지 않으면
    다른 크기로 학습한 model에 이 은행을 먹이고도 아무 말이 없습니다.
    """

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE})
    path = repository_root / "checkpoints/embedding.pt"
    _write_embedding_checkpoint(path, [3, 7])
    payload = torch.load(path, map_location="cpu")
    for key, value in broken.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    torch.save(payload, path)
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    result = run(base_config)

    assert result["status"] == "error"
    assert expected in result["message"]


def test_the_reported_median_is_a_real_median(base_config: dict, repository_root: Path):
    """보고 숫자가 틀리면 그것을 근거로 다음 판단을 합니다.

    행이 짝수 개일 때 `values[len // 2]`는 위쪽 값이라 median이 아닙니다. 여기서는
    margin이 -0.5와 0.5라 median은 0이어야 합니다.
    """

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE})
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank
    original = rerank_module._margins
    rerank_module._margins = lambda similarity, **kwargs: (
        torch.tensor([-0.5, 0.5][: similarity.shape[0]]),
        torch.ones(similarity.shape[0], dtype=torch.bool),
    )
    try:
        result = run(base_config)
    finally:
        rerank_module._margins = original

    assert result["status"] == "ok", result["message"]
    document = json.loads(
        (repository_root / result["artifacts"]["test_predictions_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert document["rerank"]["reranked_rows"] == 2
    assert document["rerank"]["median_margin"] == pytest.approx(0.0)


def test_a_margin_that_is_not_a_number_stops_the_run(
    base_config: dict, repository_root: Path
):
    """"재지 못했다"와 "재려다 망가졌다"를 같게 다루면 조용한 무동작이 됩니다.

    model이 발산해 특징이 전부 `nan`이 되면 margin도 `nan`이 되는데, 그것을 참조 없는
    class와 같게 건너뛰면 **한 행도 바꾸지 않은 채 성공으로** 끝납니다.
    """

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE})
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank
    original = rerank_module._margins
    rerank_module._margins = lambda similarity, **kwargs: (
        torch.full((similarity.shape[0],), float("nan")),
        torch.ones(similarity.shape[0], dtype=torch.bool),
    )
    try:
        result = run(base_config)
    finally:
        rerank_module._margins = original

    assert result["status"] == "error"
    assert "숫자가 아닙니다" in result["message"]


def test_an_inference_failure_is_reported_not_raised(
    base_config: dict, repository_root: Path
):
    """GPU가 모자라거나 kernel이 터지면 `RuntimeError`가 납니다.

    그대로 두면 `run()` 경계를 넘어 나갑니다. 이 pipeline은 실패를 status=error로
    돌려주지, 예외를 내보내지 않습니다.
    """

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE})
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    def exploding(*args, **kwargs):
        def model(_batch):
            raise RuntimeError("CUDA out of memory")

        return model

    original = rerank_module._embedding_model
    rerank_module._embedding_model = exploding
    try:
        result = run(base_config)
    finally:
        rerank_module._embedding_model = original

    assert result["status"] == "error"
    assert "재순위 추론에 실패했습니다" in result["message"]


def test_a_missing_reference_crop_is_reported_not_raised(
    base_config: dict, repository_root: Path
):
    """이 pipeline은 `run()` 밖으로 예외를 내보내지 않습니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE})
    # 목록에는 있는데 tar에는 없는 crop을 만듭니다.
    with tarfile.open(repository_root / "data/test/crop_bank.tar") as archive:
        names = [name for name in archive.getnames() if name != "index.json"]
        members = {name: archive.extractfile(name).read() for name in names}
        index = json.loads(archive.extractfile("index.json").read().decode("utf-8"))
    index["records"].append({"path": "crops/3/missing.png", "category_id": 3})
    with tarfile.open(repository_root / "data/test/crop_bank.tar", "w") as archive:
        payload = json.dumps(index, ensure_ascii=False).encode("utf-8")
        info = tarfile.TarInfo("index.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        for name, blob in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            archive.addfile(info, io.BytesIO(blob))
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    result = run(base_config)

    assert result["status"] == "error"
    assert "참조 crop을 열지 못했습니다" in result["message"]


def test_rerank_needs_reference_crops(base_config: dict, repository_root: Path):
    _prepare(base_config, repository_root, category_ids=(3, 7))
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]

    result = run(base_config)

    assert result["status"] == "error"
    assert "재순위에는 참조 crop이 필요합니다" in result["message"]


@pytest.mark.parametrize(
    "second",
    [
        "checkpoints/embedding.pt",
        # 글자는 다르지만 같은 파일입니다. 표기로만 보면 통과해, 그 model이
        # 평균에서 두 표를 갖습니다.
        "./checkpoints/embedding.pt",
        "checkpoints/../checkpoints/embedding.pt",
    ],
)
def test_rerank_refuses_the_same_checkpoint_twice(
    base_config: dict, repository_root: Path, second: str
):
    """같은 model을 두 번 넣으면 평균이 조용히 그쪽으로 기웁니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = [
        "checkpoints/embedding.pt",
        second,
    ]
    base_config["evaluate"]["rerank_crop_bank_uri"] = "data/test/crop_bank.tar"

    result = run(base_config)

    assert result["status"] == "error"
    assert "같은 checkpoint를 가리킵니다" in result["message"]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ({"escaping": "../escaped.txt"}, "폴더 밖을 가리키는 항목"),
        # 푸는 자리가 `bank`일 때 `bank-evil`은 **문자열 앞자리가 같습니다.**
        # startswith로 경계를 보면 이것이 통과합니다.
        ({"escaping": "../bank-evil/escaped.txt"}, "폴더 밖을 가리키는 항목"),
        ({"linking": True}, "보통 파일이 아닌 항목"),
        # tar은 멀쩡한데 목록만 밖을 가리키는 경우입니다. 그 경로를 여는 것은
        # 재순위 도중이라, 여기서 안 막으면 batch마다 저장소 밖 파일을 읽습니다.
        ({"listing_escapes": True}, "폴더 밖을 가리키는 항목"),
    ],
)
def test_rerank_refuses_a_crop_bank_that_reaches_outside_its_folder(
    base_config: dict, repository_root: Path, kind: dict, expected: str
):
    """남이 만든 tar을 그대로 푸는 습관을 남기지 않습니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(
        repository_root / "data/test/crop_bank.tar", {3: RED, 7: BLUE}, **kind
    )
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    result = run(base_config)

    assert result["status"] == "error"
    assert expected in result["message"]


@pytest.mark.parametrize(
    "margin",
    [
        # JSON에는 자릿수 제한이 없어 이런 정수가 그대로 들어옵니다. 맨
        # `math.isfinite`로 보면 **거절하려던 검사가 그 자리에서 터집니다.**
        10**400,
        -0.1,
    ],
    ids=["overflow", "negative"],
)
def test_a_crop_bank_whose_margin_is_not_a_ratio_is_refused(
    base_config: dict, repository_root: Path, margin: object
):
    """crop을 얼마나 넓게 잘랐는지는 은행이 말합니다. 그 값이 이상하면 멈춥니다."""

    _prepare(base_config, repository_root, category_ids=(3, 7))
    bank = _write_crop_bank(
        repository_root / "data/test/crop_bank.tar",
        {3: RED, 7: BLUE},
        overrides={"crop_margin": margin},
    )
    _write_embedding_checkpoint(repository_root / "checkpoints/embedding.pt", [3, 7])
    base_config["evaluate"]["rerank_checkpoint_uris"] = ["checkpoints/embedding.pt"]
    base_config["evaluate"]["rerank_crop_bank_uri"] = bank

    result = run(base_config)

    assert result["status"] == "error"
    assert "crop_margin는 0 이상의 유한한 값이어야 합니다" in result["message"]


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
