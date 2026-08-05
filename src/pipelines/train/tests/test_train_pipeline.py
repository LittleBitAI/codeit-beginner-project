import copy
import io
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from botocore.exceptions import ClientError
from PIL import Image
from torch import nn
from torchvision.models.detection import FasterRCNN

from src.common import LocalStorage, S3Storage, StorageError
from src.pipelines import train
from src.pipelines.train import pipeline
from src.pipelines.train.dataset import REPOSITORY_ROOT, load_class_map
from src.pipelines.train.model import build_model
from src.pipelines.train import progress as progress_module
from src.pipelines.train.progress import SCHEMA, ProgressEmitter
from src.pipelines.train.trainer import train_model


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(()))

    def forward(self, images, targets):
        image_term = sum(image.mean() * 0 for image in images)
        return {"loss_classifier": (self.weight - 0.25).square() + image_term}


class BatchNormDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets):
        normalized = self.batch_norm(torch.stack(images))
        return {"loss_classifier": self.weight.square() + normalized.mean() * 0}


class InMemoryDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, value: float) -> None:
        self.image = torch.full((3, 2, 2), value)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.image, {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _write_image(path: Path, *, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color=color).save(path)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _manifest(image_name: str) -> dict:
    return {
        "images": [
            {"id": 1, "file_name": image_name, "width": 16, "height": 12}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [2, 3, 5, 4],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 7, "name": "pill"}],
    }


def _absolute_storage_config(tmp_path: Path, monkeypatch) -> tuple[dict, Path]:
    monkeypatch.setattr(pipeline, "build_model", lambda *args, **kwargs: TinyDetector())
    storage_root = tmp_path / "absolute-storage"
    storage = LocalStorage(storage_root)
    source_train = tmp_path / "source-train.png"
    source_validation = tmp_path / "source-validation.png"
    _write_image(source_train, color="red")
    _write_image(source_validation, color="blue")
    storage.upload_file(source_train, "datasets/train.png")
    storage.upload_file(source_validation, "datasets/validation.png")
    inputs = {
        "train_manifest_uri": storage.write_json(
            "datasets/train.json", _manifest("train.png")
        ),
        "validation_manifest_uri": storage.write_json(
            "datasets/validation.json", _manifest("validation.png")
        ),
        "class_map_uri": storage.write_json("datasets/class_map.json", {"pill": 1}),
        "dataset_summary_uri": storage.write_json(
            "datasets/summary.json", {"train_images": 1, "validation_images": 1}
        ),
    }
    config = {
        "storage": {"backend": "local", "local": {"root": _relative(storage_root)}},
        "inputs": {"data": inputs},
        "train": {
            "run_id": "absolute-inputs",
            "epochs": 1,
            "output_dir": _relative(tmp_path / "absolute-outputs"),
        },
    }
    return config, storage_root


@pytest.fixture
def local_config(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "build_model", lambda *args, **kwargs: TinyDetector())
    fixture_directory = tmp_path / "fixtures"
    train_image = fixture_directory / "train.png"
    validation_image = fixture_directory / "validation.png"
    _write_image(train_image, color="red")
    _write_image(validation_image, color="blue")
    train_manifest = fixture_directory / "train.json"
    validation_manifest = fixture_directory / "validation.json"
    class_map = fixture_directory / "class_map.json"
    dataset_summary = fixture_directory / "summary.json"
    _write_json(train_manifest, _manifest(train_image.name))
    _write_json(validation_manifest, _manifest(validation_image.name))
    _write_json(class_map, {"pill": 1})
    _write_json(dataset_summary, {"train_images": 1, "validation_images": 1})
    return {
        "storage": {"backend": "local", "local": {"root": _relative(tmp_path / "storage")}},
        "inputs": {
            "data": {
                "train_manifest_uri": _relative(train_manifest),
                "validation_manifest_uri": _relative(validation_manifest),
                "class_map_uri": _relative(class_map),
                "dataset_summary_uri": _relative(dataset_summary),
            }
        },
        "train": {
            "run_id": "cpu-smoke",
            "seed": 17,
            "epochs": 2,
            "batch_size": 1,
            "device": "cpu",
            "output_dir": _relative(tmp_path / "outputs"),
        },
    }


def test_run_trains_and_writes_contract_artifacts_without_mutating_inputs(local_config):
    original_inputs = copy.deepcopy(local_config["inputs"])

    result = train.run(local_config)

    assert set(result) == {"status", "artifacts", "summary", "message"}
    assert result["status"] == "ok"
    assert set(result["artifacts"]) == {
        "run_id",
        "best_checkpoint_uri",
        "last_checkpoint_uri",
        "training_history_uri",
    }
    assert local_config["inputs"] == original_inputs
    assert result["summary"]["train_images"] == 1
    assert result["summary"]["validation_images"] == 1
    assert result["summary"]["class_count"] == 1
    assert result["summary"]["epochs"] == 2

    for name in ("best_checkpoint_uri", "last_checkpoint_uri", "training_history_uri"):
        uri = result["artifacts"][name]
        assert not Path(uri).is_absolute()
        assert (REPOSITORY_ROOT / uri).is_file()
    history = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["training_history_uri"]).read_text(encoding="utf-8")
    )
    assert [entry["epoch"] for entry in history] == [1, 2]
    for artifact_name in ("best_checkpoint_uri", "last_checkpoint_uri"):
        checkpoint = torch.load(
            REPOSITORY_ROOT / result["artifacts"][artifact_name],
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["architecture"] == "fasterrcnn_mobilenet_v3_large_320_fpn"
        assert checkpoint["class_map"] == {"pill": 1}
        assert checkpoint["category_ids"] == [0, 7]
        assert checkpoint["num_classes"] == 2
        assert checkpoint["seed"] == 17


def _progress_events(captured_stderr: str) -> list[dict]:
    """torch 경고처럼 JSON이 아닌 줄은 건너뛰고 진행 event만 모읍니다."""

    events = []
    for line in captured_stderr.splitlines():
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == SCHEMA:
            events.append(payload)
    return events


def test_run_emits_progress_events_on_stderr_and_leaves_stdout_empty(
    local_config, capsys
):
    result = train.run(local_config)
    captured = capsys.readouterr()

    assert result["status"] == "ok"
    assert captured.out == ""

    events = _progress_events(captured.err)
    assert [event["event"] for event in events] == [
        "run_started",
        "epoch_started",
        "epoch_completed",
        "epoch_started",
        "epoch_completed",
    ]
    assert all(event["run_id"] == "cpu-smoke" for event in events)
    for event in events:
        datetime.strptime(event["ts"], "%Y-%m-%dT%H:%M:%S.%fZ")

    started, _, first, _, last = events
    assert started == {
        "schema": SCHEMA,
        "event": "run_started",
        "run_id": "cpu-smoke",
        "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "device": "cpu",
        "epochs": 2,
        "train_images": 1,
        "validation_images": 1,
        "class_count": 1,
        "ts": started["ts"],
    }
    assert set(first) == {
        "schema",
        "event",
        "run_id",
        "epoch",
        "epochs",
        "train_loss",
        "validation_loss",
        "best_validation_loss",
        "best_epoch",
        "is_best",
        "epoch_seconds",
        "ts",
    }
    assert (first["epoch"], last["epoch"]) == (1, 2)
    assert first["is_best"] is True
    assert first["best_epoch"] == 1
    assert first["best_validation_loss"] == first["validation_loss"]
    assert isinstance(first["epoch_seconds"], float)
    assert first["epoch_seconds"] >= 0.0

    history = json.loads(
        (REPOSITORY_ROOT / result["artifacts"]["training_history_uri"]).read_text(
            encoding="utf-8"
        )
    )
    assert [entry["epoch"] for entry in history] == [1, 2]
    assert [entry["train_loss"] for entry in history] == [
        first["train_loss"],
        last["train_loss"],
    ]
    assert result["summary"]["best_epoch"] == last["best_epoch"]
    assert result["summary"]["best_validation_loss"] == last["best_validation_loss"]


def test_progress_stream_stays_silent_for_the_dummy_execution(capsys):
    result = train.run({"execution": {"mode": "dummy"}})
    captured = capsys.readouterr()

    assert result["status"] == "ok"
    assert captured.out == ""
    assert _progress_events(captured.err) == []


def test_progress_emitter_does_not_raise_when_the_reader_closed_the_pipe():
    class ClosedPipe:
        def write(self, line):
            raise BrokenPipeError("reader is gone")

        def flush(self):
            raise BrokenPipeError("reader is gone")

    emitter = ProgressEmitter("cancelled-run", ClosedPipe())

    assert emitter.emit("epoch_started", epoch=1, epochs=2) is None


def test_progress_emitter_stays_quiet_outside_the_creating_process(monkeypatch):
    stream = io.StringIO()
    emitter = ProgressEmitter("worker-run", stream)
    monkeypatch.setattr(progress_module.os, "getpid", lambda: -1)

    emitter.emit("epoch_started", epoch=1, epochs=2)

    assert stream.getvalue() == ""


def test_progress_emitter_writes_null_instead_of_invalid_json_numbers():
    stream = io.StringIO()

    ProgressEmitter("nan-run", stream).emit(
        "epoch_completed", train_loss=float("nan"), validation_loss=float("inf")
    )

    line = stream.getvalue()
    assert line.endswith("\n")
    assert "NaN" not in line and "Infinity" not in line
    payload = json.loads(line)
    assert payload["train_loss"] is None
    assert payload["validation_loss"] is None


def test_load_class_map_converts_category_ids_in_sorted_order_without_mutating_input():
    document = {"11": " capsule ", "7": "pill"}
    original = copy.deepcopy(document)
    storage = Mock()
    storage.read_json.return_value = document

    class_map = load_class_map("s3://bucket/class-map.json", storage)

    assert class_map == {"pill": 1, "capsule": 2}
    assert document == original


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"7": "pill", "11": " pill "}, "names must be unique"),
        ({"category-seven": "pill"}, "category ids must be non-negative integers"),
        ({"-7": "pill"}, "category ids must be non-negative integers"),
        ({"7": "   "}, "names must be non-empty strings"),
        ({"1": "pill", "01": "capsule"}, "category ids must be unique"),
    ],
)
def test_load_class_map_rejects_invalid_category_id_format(document, message):
    storage = Mock()
    storage.read_json.return_value = document

    with pytest.raises(ValueError, match=message):
        load_class_map("s3://bucket/class-map.json", storage)


def test_run_accepts_category_id_to_name_class_map_and_keeps_checkpoint_contract(
    local_config,
):
    class_map_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["class_map_uri"]
    _write_json(class_map_path, {"7": "pill"})
    original_inputs = copy.deepcopy(local_config["inputs"])

    result = train.run(local_config)

    assert result["status"] == "ok"
    assert local_config["inputs"] == original_inputs
    checkpoint = torch.load(
        REPOSITORY_ROOT / result["artifacts"]["best_checkpoint_uri"],
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["class_map"] == {"pill": 1}
    assert checkpoint["category_ids"] == [0, 7]


def test_checkpoint_category_ids_are_indexed_by_model_label():
    checkpoint = {"model_state_dict": {"weight": torch.tensor(1.0)}}

    payload = pipeline._checkpoint_payload(
        checkpoint,
        {"seed": 17},
        {"capsule": 2, "pill": 1},
        {2: 11, 1: 7},
    )

    assert payload["category_ids"] == [0, 7, 11]
    assert payload["model_state_dict"] is checkpoint["model_state_dict"]
    assert payload["num_classes"] == 3


def test_run_accepts_absolute_paths_returned_by_local_storage(tmp_path, monkeypatch):
    config, _ = _absolute_storage_config(tmp_path, monkeypatch)

    result = train.run(config)

    assert result["status"] == "ok"


def test_run_rejects_absolute_input_outside_local_storage_root(tmp_path, monkeypatch):
    config, storage_root = _absolute_storage_config(tmp_path, monkeypatch)
    outside_class_map = storage_root.parent / "outside-class-map.json"
    _write_json(outside_class_map, {"pill": 1})
    config["inputs"]["data"]["class_map_uri"] = str(outside_class_map.resolve())

    result = train.run(config)

    assert result["status"] == "error"
    assert "leaves the storage root" in result["message"]


def test_run_refuses_to_overwrite_an_existing_run(local_config):
    first = train.run(local_config)
    second = train.run(local_config)

    assert first["status"] == "ok"
    assert second == {
        "status": "error",
        "artifacts": {},
        "summary": {},
        "message": "training failed: training run artifact already exists: cpu-smoke",
    }


def test_seed_reproduces_training_history(local_config):
    first_config = copy.deepcopy(local_config)
    second_config = copy.deepcopy(local_config)
    first_config["train"]["run_id"] = "seed-first"
    second_config["train"]["run_id"] = "seed-second"

    first = train.run(first_config)
    second = train.run(second_config)

    first_history = json.loads(
        (REPOSITORY_ROOT / first["artifacts"]["training_history_uri"]).read_text(encoding="utf-8")
    )
    second_history = json.loads(
        (REPOSITORY_ROOT / second["artifacts"]["training_history_uri"]).read_text(encoding="utf-8")
    )
    assert first_history == second_history


def test_run_reports_missing_input_artifacts_without_partial_success():
    result = train.run({"inputs": {"data": {}}, "train": {}})

    assert set(result) == {"status", "artifacts", "summary", "message"}
    assert result["status"] == "error"
    assert result["artifacts"] == {}
    assert result["summary"] == {}
    assert "missing artifacts" in result["message"]


def test_run_rejects_invalid_class_map(local_config):
    class_map = REPOSITORY_ROOT / local_config["inputs"]["data"]["class_map_uri"]
    _write_json(class_map, {"pill": 0})

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "0 is background" in result["message"]


def test_run_rejects_invalid_bbox_schema(local_config):
    manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["train_manifest_uri"]
    manifest = _manifest("train.png")
    manifest["annotations"][0]["bbox"] = [2, 3, -1, 4]
    _write_json(manifest_path, manifest)

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "positive size" in result["message"]


def test_run_rejects_corrupt_image(local_config):
    manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["train_manifest_uri"]
    image_path = manifest_path.parent / "train.png"
    image_path.write_bytes(b"not-an-image")

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "missing or corrupt" in result["message"]


def test_run_rejects_train_validation_image_overlap(local_config):
    train_manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["train_manifest_uri"]
    validation_manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["validation_manifest_uri"]
    _write_json(validation_manifest_path, _manifest("train.png"))

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "overlapping images" in result["message"]


def test_run_rejects_different_category_ids_between_splits(local_config):
    validation_manifest_path = REPOSITORY_ROOT / local_config["inputs"]["data"]["validation_manifest_uri"]
    manifest = _manifest("validation.png")
    manifest["categories"][0]["id"] = 9
    manifest["annotations"][0]["category_id"] = 9
    _write_json(validation_manifest_path, manifest)

    result = train.run(local_config)

    assert result["status"] == "error"
    assert "category ids must match" in result["message"]


def test_model_builder_creates_faster_rcnn_without_downloading_weights():
    model = build_model(3, pretrained=False)

    assert isinstance(model, FasterRCNN)
    assert model.roi_heads.box_predictor.cls_score.out_features == 3


def test_faster_rcnn_cpu_forward_and_backward_smoke():
    torch.manual_seed(17)
    model = build_model(2, pretrained=False).cpu().train()
    image = torch.rand(3, 32, 32)
    target = {
        "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor([0], dtype=torch.int64),
        "area": torch.tensor([256.0]),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }

    losses = model([image], [target])
    total = sum(losses.values())
    total.backward()

    assert set(losses) == {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    }
    assert torch.isfinite(total)


def test_validation_does_not_update_batch_norm_running_statistics():
    model = BatchNormDetector()
    settings = {
        "seed": 17,
        "device": "cpu",
        "batch_size": 1,
        "num_workers": 0,
        "learning_rate": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "epochs": 1,
    }

    train_model(
        model,
        InMemoryDetectionDataset(1.0),
        InMemoryDetectionDataset(0.0),
        settings,
    )

    assert torch.allclose(model.batch_norm.running_mean, torch.full((3,), 0.1))


def test_s3_publisher_returns_attempt_uris_and_writes_completion_marker(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pipeline,
        "uuid4",
        lambda: SimpleNamespace(hex="fixed-attempt"),
    )
    client = Mock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "missing"}}, "HeadObject"
    )
    uploaded = {}

    def put_object(**request):
        body = request["Body"]
        uploaded[request["Key"]] = body.read() if hasattr(body, "read") else body

    client.put_object.side_effect = put_object
    storage = S3Storage("bucket", client=client)
    checkpoint = {"model_state_dict": {"weight": torch.tensor(1.0)}}
    settings = {"run_id": "s3-smoke", "output_prefix": "experiments/completed"}

    artifacts = pipeline._publish_s3(
        storage,
        checkpoint,
        checkpoint,
        [{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.0}],
        settings,
    )

    assert artifacts == {
        "best_checkpoint_uri": "s3://bucket/experiments/completed/s3-smoke/attempts/fixed-attempt/best_checkpoint.pt",
        "last_checkpoint_uri": "s3://bucket/experiments/completed/s3-smoke/attempts/fixed-attempt/last_checkpoint.pt",
        "training_history_uri": "s3://bucket/experiments/completed/s3-smoke/attempts/fixed-attempt/training_history.json",
    }
    assert set(uploaded) == {
        "experiments/completed/s3-smoke/attempts/fixed-attempt/best_checkpoint.pt",
        "experiments/completed/s3-smoke/attempts/fixed-attempt/last_checkpoint.pt",
        "experiments/completed/s3-smoke/attempts/fixed-attempt/training_history.json",
        "experiments/completed/s3-smoke/completed.json",
    }
    assert all(call.kwargs["IfNoneMatch"] == "*" for call in client.put_object.call_args_list)


def test_s3_publisher_can_retry_same_run_id_after_intermediate_failure(monkeypatch):
    attempt_ids = iter(("failed-attempt", "successful-attempt"))
    monkeypatch.setattr(
        pipeline,
        "uuid4",
        lambda: SimpleNamespace(hex=next(attempt_ids)),
    )

    class RetryStorage:
        def __init__(self):
            self.objects = {}
            self.fail_once = True

        def exists(self, location):
            return location in self.objects

        def upload_file(self, source, destination):
            if self.fail_once and destination.endswith("last_checkpoint.pt"):
                self.fail_once = False
                raise StorageError("simulated intermediate upload failure")
            self.objects[destination] = Path(source).read_bytes()
            return f"s3://bucket/{destination}"

        def write_json(self, destination, value):
            self.objects[destination] = value
            return f"s3://bucket/{destination}"

    storage = RetryStorage()
    checkpoint = {"model_state_dict": {"weight": torch.tensor(1.0)}}
    history = [{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.0}]
    settings = {"run_id": "retry-smoke", "output_prefix": "experiments/completed"}

    with pytest.raises(StorageError, match="intermediate upload failure"):
        pipeline._publish_s3(storage, checkpoint, checkpoint, history, settings)

    completion = "experiments/completed/retry-smoke/completed.json"
    assert completion not in storage.objects
    artifacts = pipeline._publish_s3(storage, checkpoint, checkpoint, history, settings)

    assert "/attempts/successful-attempt/" in artifacts["best_checkpoint_uri"]
    assert storage.objects[completion] == {
        "run_id": "retry-smoke",
        "artifacts": artifacts,
    }
    with pytest.raises(FileExistsError, match="already exists"):
        pipeline._publish_s3(storage, checkpoint, checkpoint, history, settings)
