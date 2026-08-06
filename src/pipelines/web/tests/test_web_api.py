"""HTTP route.

실제 학습은 돌리지 않습니다. subprocess는 ``jobs/runner.py``를 patch해 대신합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipelines.web.api.app import ALLOWED_ORIGINS
from src.pipelines.web.api.routes_train import ARCHITECTURE
from src.pipelines.web.jobs import runner
from src.pipelines.web.paths import REPOSITORY_ROOT
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS  # noqa: F401  (fixture에서 사용)


TRAVERSAL_IDS = (
    "../../etc/passwd",
    "..%2f..%2fetc%2fpasswd",
    "....//....//x",
    "0" * 31,
    "A" * 32,
    "nope",
)


def create_config(client, payload) -> str:
    response = client.post("/api/train/configs", json=payload)
    assert response.status_code == 201, response.text
    return response.json()["config_id"]


# --- 기본 ------------------------------------------------------------------


def test_health_returns_ok(client):
    assert client.get("/api/health").json() == {"status": "ok", "version": "1"}


def test_defaults_expose_every_train_field(client):
    body = client.get("/api/train/defaults").json()

    names = {field["name"] for field in body["fields"]}
    assert names == {
        "run_id",
        "architecture",
        "optimizer",
        "augmentation",
        "seed",
        "epochs",
        "batch_size",
        "num_workers",
        "learning_rate",
        "momentum",
        "weight_decay",
        "beta1",
        "beta2",
        "epsilon",
        "device",
        "pretrained",
        "output_dir",
        "output_prefix",
    }
    assert {field["name"] for field in body["data_fields"]} == set(DATA_ARTIFACT_KEYS)
    learning_rate = next(
        field for field in body["fields"] if field["name"] == "learning_rate"
    )
    assert learning_rate["defaults_by_optimizer"] == {
        "AdamW": 0.0001,
        "SGD": 0.005,
        "Adam": 0.0001,
    }
    # 값 자체가 train과 같은지는 test_web_train_contract.py가 train source를 읽어 확인합니다.
    assert body["architecture"] == ARCHITECTURE


def test_defaults_report_cuda_availability(client, monkeypatch):
    from src.pipelines.web.api import routes_train

    monkeypatch.setattr(routes_train, "cuda_is_available", lambda: False)

    devices = {item["value"]: item for item in client.get("/api/train/defaults").json()["devices"]}

    assert devices["cpu"]["available"] is True
    assert devices["cuda"]["available"] is False
    assert devices["cuda"]["reason"]


# --- 검증 ------------------------------------------------------------------


def test_validate_returns_field_errors(client, data_inputs):
    body = client.post(
        "/api/train/validate", json={"train": {"epochs": 0, "seed": True}, "inputs": {"data": data_inputs}}
    ).json()

    assert body["valid"] is False
    fields = {item["field"] for item in body["errors"]}
    assert {"train.epochs", "train.seed"} <= fields


def test_validate_writes_nothing(client, valid_payload, isolated_repo):
    client.post("/api/train/validate", json=valid_payload)

    assert not (isolated_repo / "artifacts" / "web" / "configs").exists()


def test_validate_accepts_valid_payload(client, valid_payload):
    body = client.post("/api/train/validate", json=valid_payload).json()

    assert body["valid"] is True
    assert body["normalized"]["execution"] == {"mode": "real"}


@pytest.mark.parametrize("optimizer", ("AdamW", "Adam"))
def test_validate_applies_adam_profile_and_omits_momentum(
    client, data_inputs, optimizer
):
    body = client.post(
        "/api/train/validate",
        json={"train": {"run_id": "adam-run", "optimizer": optimizer}, "inputs": {"data": data_inputs}},
    ).json()

    assert body["valid"] is True
    train = body["normalized"]["train"]
    assert train["optimizer"] == optimizer
    assert train["learning_rate"] == 0.0001
    assert train["beta1"] == 0.9
    assert train["beta2"] == 0.999
    assert train["epsilon"] == 1e-8
    assert "momentum" not in train


@pytest.mark.parametrize("field", ("beta1", "beta2"))
def test_validate_rejects_adam_beta_equal_to_one(client, data_inputs, field):
    body = client.post(
        "/api/train/validate",
        json={
            "train": {"run_id": "bad-beta", "optimizer": "AdamW", field: 1.0},
            "inputs": {"data": data_inputs},
        },
    ).json()

    assert body["valid"] is False
    assert body["errors"][0]["field"] == f"train.{field}"


def test_validate_keeps_legacy_sgd_when_optimizer_is_missing(client, valid_payload):
    body = client.post("/api/train/validate", json=valid_payload).json()

    train = body["normalized"]["train"]
    assert train["optimizer"] == "SGD"
    assert train["momentum"] == 0.9
    assert "beta1" not in train


@pytest.mark.parametrize(
    ("train", "field"),
    (
        ({"optimizer": "AdamW", "momentum": 0.5}, "train.momentum"),
        ({"optimizer": "SGD", "beta1": 0.8}, "train.beta1"),
        ({"optimizer": "SGD", "epsilon": 1e-7}, "train.epsilon"),
    ),
)
def test_validate_rejects_irrelevant_optimizer_fields(client, data_inputs, train, field):
    body = client.post(
        "/api/train/validate",
        json={"train": {"run_id": "bad-profile", **train}, "inputs": {"data": data_inputs}},
    ).json()

    assert body["valid"] is False
    assert field in {item["field"] for item in body["errors"]}


# --- 설정 저장 --------------------------------------------------------------


def test_create_config_returns_id_without_absolute_path(client, valid_payload):
    body = client.post("/api/train/configs", json=valid_payload).json()

    assert len(body["config_id"]) == 32
    assert body["run_id"] == "test-run"
    serialized = json.dumps(body, ensure_ascii=False)
    assert str(REPOSITORY_ROOT) not in serialized
    assert str(Path.home()) not in serialized


def test_create_config_rejects_invalid_settings(client, data_inputs):
    response = client.post(
        "/api/train/configs", json={"train": {"epochs": 0}, "inputs": {"data": data_inputs}}
    )

    assert response.status_code == 400
    assert response.json()["errors"]


# --- job 실행 ---------------------------------------------------------------


def test_start_job_rejects_second_job_with_409(
    client, valid_payload, monkeypatch, fake_process_factory
):
    process = fake_process_factory(block_until_signalled=True)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    config_id = create_config(client, valid_payload)

    first = client.post("/api/train/jobs", json={"config_id": config_id})
    second = client.post("/api/train/jobs", json={"config_id": config_id})

    assert first.status_code == 201
    assert second.status_code == 409
    assert "한 번에 하나만 실행할 수 있습니다" in second.json()["message"]
    process.release()


def test_start_job_with_unknown_config_returns_404(client):
    response = client.post("/api/train/jobs", json={"config_id": "0" * 32})

    assert response.status_code == 404


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_start_job_rejects_traversal_config_id(client, bad, monkeypatch):
    spawned = []
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: spawned.append(a))

    response = client.post("/api/train/jobs", json={"config_id": bad})

    assert response.status_code in (404, 422)
    assert spawned == []  # process를 띄우지 않습니다


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_job_id_traversal_returns_404(client, bad, isolated_repo):
    """id는 디스크 경로가 되기 전에 ``^[0-9a-f]{32}$``로 걸러집니다.

    경로를 만들기 전에 막는다는 것 자체는 ``test_store_rejects_bad_job_id``가 확인합니다.
    """

    response = client.get(f"/api/train/jobs/{bad}")

    assert response.status_code == 404
    # 어떤 파일도 만들어지지 않아야 합니다.
    assert not (isolated_repo / "artifacts" / "web" / "jobs").exists()


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_job_logs_traversal_returns_404(client, bad):
    assert client.get(f"/api/train/jobs/{bad}/logs").status_code == 404


def test_job_lifecycle_through_api(client, valid_payload, monkeypatch, fake_process_factory):
    import time

    stdout = json.dumps(
        {
            "status": "ok",
            "artifacts": {"train": {"run_id": "test-run", "best_checkpoint_uri": "artifacts/x/best.pt"}},
            "summary": {"train": {"epochs": 2, "best_epoch": 1}},
            "message": "pipeline 실행을 완료했습니다.",
        },
        ensure_ascii=False,
        indent=2,
    )
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stdout=stdout))
    config_id = create_config(client, valid_payload)

    started = client.post("/api/train/jobs", json={"config_id": config_id}).json()
    job_id = started["job_id"]

    for _ in range(500):
        body = client.get(f"/api/train/jobs/{job_id}").json()
        if body["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)

    assert body["status"] == "succeeded"
    assert body["artifacts"]["best_checkpoint_uri"] == "artifacts/x/best.pt"
    assert body["summary"]["best_epoch"] == 1
    assert body["elapsed_seconds"] is not None
    # train이 아직 진행 로그를 내보내지 않으므로 지어내지 않고 그대로 보고합니다.
    assert body["progress"]["available"] is False

    listing = client.get("/api/train/jobs").json()
    assert len(listing["jobs"]) == 1
    assert listing["active_job_id"] is None

    logs = client.get(f"/api/train/jobs/{job_id}/logs").json()
    assert logs["complete"] is True

    config_body = client.get(f"/api/train/jobs/{job_id}/config").json()
    assert config_body["config"]["execution"] == {"mode": "real"}


def test_jobs_can_be_filtered_by_status(client, valid_payload, monkeypatch, fake_process_factory):
    import time

    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stdout="깨짐", exit_code=1))
    config_id = create_config(client, valid_payload)
    job_id = client.post("/api/train/jobs", json={"config_id": config_id}).json()["job_id"]
    for _ in range(500):
        if client.get(f"/api/train/jobs/{job_id}").json()["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)

    assert len(client.get("/api/train/jobs?status=failed").json()["jobs"]) == 1
    assert client.get("/api/train/jobs?status=succeeded").json()["jobs"] == []


def test_cancel_returns_409_for_finished_job(
    client, valid_payload, monkeypatch, fake_process_factory
):
    import time

    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stdout="깨짐", exit_code=1))
    config_id = create_config(client, valid_payload)
    job_id = client.post("/api/train/jobs", json={"config_id": config_id}).json()["job_id"]
    for _ in range(500):
        if client.get(f"/api/train/jobs/{job_id}").json()["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)

    assert client.post(f"/api/train/jobs/{job_id}/cancel").status_code == 409


def test_cancel_accepts_running_job(client, valid_payload, monkeypatch, fake_process_factory):
    process = fake_process_factory(block_until_signalled=True, exit_code=1)
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: process)
    monkeypatch.setattr(runner, "terminate_tree", lambda proc: proc.release(1))
    config_id = create_config(client, valid_payload)
    job_id = client.post("/api/train/jobs", json={"config_id": config_id}).json()["job_id"]

    assert client.post(f"/api/train/jobs/{job_id}/cancel").status_code == 202


# --- 보안 ------------------------------------------------------------------


def test_gpu_status_returns_200_even_when_unavailable(client, monkeypatch):
    from src.pipelines.web import gpu

    monkeypatch.setattr(gpu, "_resolve_nvidia_smi", lambda: None)

    response = client.get("/api/gpu/status")

    assert response.status_code == 200
    assert response.json()["telemetry"]["devices"] == []


def test_no_route_returns_absolute_paths(client, valid_payload, monkeypatch, fake_process_factory):
    """향후 route가 늘어나도 개인 절대 경로가 새지 않도록 하는 범용 가드입니다."""

    import time

    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout="깨짐", exit_code=1)
    )
    config_id = create_config(client, valid_payload)
    job_id = client.post("/api/train/jobs", json={"config_id": config_id}).json()["job_id"]
    for _ in range(500):
        if client.get(f"/api/train/jobs/{job_id}").json()["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)

    forbidden = (str(REPOSITORY_ROOT), str(Path.home()), Path.home().name)
    for path in (
        "/api/health",
        "/api/train/defaults",
        "/api/gpu/status",
        "/api/train/jobs",
        f"/api/train/jobs/{job_id}",
        f"/api/train/jobs/{job_id}/logs",
        f"/api/train/jobs/{job_id}/config",
    ):
        body = client.get(path).text
        for secret in forbidden:
            assert secret not in body, f"{path}가 {secret}를 노출했습니다"


@pytest.mark.parametrize(
    "origin",
    (
        *ALLOWED_ORIGINS,
        "http://127.0.0.1:8000",  # 서버가 스스로를 부르는 경우
        "http://localhost:8010",  # --port로 바꿔 띄운 경우
    ),
)
def test_cors_allows_this_computer(client, origin):
    """Vite가 만든 module script에는 crossorigin이 붙어 같은 origin도 CORS 검사를 받습니다.

    서버 자신의 origin이 빠지면 응답에 Access-Control-Allow-Origin이 없어 script가
    실행되지 않고 화면이 빈 채로 뜹니다. port는 --port로 바뀔 수 있습니다.
    """

    response = client.get("/api/health", headers={"Origin": origin})

    assert response.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize(
    "origin",
    (
        "https://evil.example.com",
        "http://localhost.evil.com",
        "http://127.0.0.1.evil.com",
        "https://127.0.0.1",
    ),
)
def test_cors_denies_everything_else(client, origin):
    response = client.get("/api/health", headers={"Origin": origin})

    assert response.headers.get("access-control-allow-origin") is None


# --- 전처리 데이터셋 ---------------------------------------------------------


@pytest.fixture
def demo_dataset(isolated_repo):
    directory = isolated_repo / "artifacts" / "demo"
    directory.mkdir(parents=True)
    manifest = {
        "images": [{"id": 1, "file_name": "a.png", "width": 16, "height": 12}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]}],
        "categories": [{"id": 7, "name": "pill"}],
    }
    for name, document in (
        ("train.json", manifest),
        ("validation.json", manifest),
        ("class_map.json", {"pill": 1}),
        ("summary.json", {"train_images": 1, "validation_images": 1}),
    ):
        (directory / name).write_text(
            json.dumps(document, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
    return "artifacts/demo"


def test_inspect_reports_matched_artifacts(client, demo_dataset):
    body = client.post("/api/data/inspect", json={"directory": demo_dataset}).json()

    assert body["complete"] is True
    assert body["matched"]["class_map_uri"]["name"] == "class_map.json"


def test_inspect_does_not_save_the_selection(client, demo_dataset):
    client.post("/api/data/inspect", json={"directory": demo_dataset})

    assert client.get("/api/data/source").json()["source"] is None


def test_source_can_be_selected_and_read_back(client, demo_dataset):
    created = client.post("/api/data/source", json={"directory": demo_dataset})

    assert created.status_code == 200
    assert created.json()["source"]["complete"] is True

    body = client.get("/api/data/source").json()["source"]
    assert body["directory"] == "artifacts/demo"
    assert set(body["data"]) == set(DATA_ARTIFACT_KEYS)


def test_selected_source_can_fill_a_config(client, demo_dataset):
    """전처리에서 나온 4개가 그대로 설정 저장까지 통과해야 합니다."""

    source = client.post("/api/data/source", json={"directory": demo_dataset}).json()["source"]

    response = client.post(
        "/api/train/configs",
        json={"train": {"run_id": "from-source"}, "inputs": {"data": source["data"]}},
    )

    assert response.status_code == 201
    assert response.json()["config"]["inputs"]["data"] == source["data"]


def test_incomplete_directory_cannot_be_selected(client, isolated_repo):
    directory = isolated_repo / "artifacts" / "partial"
    directory.mkdir(parents=True)
    (directory / "class_map.json").write_text('{"pill": 1}', encoding="utf-8", newline="\n")

    response = client.post("/api/data/source", json={"directory": "artifacts/partial"})

    assert response.status_code == 400
    assert client.get("/api/data/source").json()["source"] is None


def test_source_can_be_cleared(client, demo_dataset):
    client.post("/api/data/source", json={"directory": demo_dataset})

    client.delete("/api/data/source")

    assert client.get("/api/data/source").json()["source"] is None


@pytest.mark.parametrize(
    "bad", ("../outside", "/etc", "C:/Windows", "\\\\server\\share", "artifacts/../../x")
)
def test_directory_traversal_is_rejected(client, bad):
    for path in ("/api/data/inspect", "/api/data/source"):
        response = client.post(path, json={"directory": bad})
        assert response.status_code == 400, f"{path} 가 {bad} 를 받아들였습니다"


def test_evaluate_requires_a_succeeded_training(client, valid_payload, monkeypatch, fake_process_factory):
    """checkpoint가 없으면 평가할 것이 없습니다."""

    import time

    monkeypatch.setattr(
        runner, "spawn", lambda *a, **k: fake_process_factory(stdout="깨짐", exit_code=1)
    )
    config_id = create_config(client, valid_payload)
    job_id = client.post("/api/train/jobs", json={"config_id": config_id}).json()["job_id"]
    for _ in range(500):
        if client.get(f"/api/train/jobs/{job_id}").json()["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)

    response = client.post(f"/api/train/jobs/{job_id}/evaluate", json={})

    assert response.status_code == 409
    assert "성공으로 끝난 학습만" in response.json()["message"]


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_evaluate_rejects_traversal_job_id(client, bad):
    assert client.get(f"/api/train/jobs/{bad}/evaluate").status_code == 404
    assert client.post(f"/api/train/jobs/{bad}/evaluate", json={}).status_code == 404


def test_evaluate_status_starts_idle(client, valid_payload, monkeypatch, fake_process_factory):
    import time

    stdout = json.dumps(
        {
            "status": "ok",
            "artifacts": {"train": {"run_id": "r", "best_checkpoint_uri": "artifacts/b.pt"}},
            "summary": {"train": {}},
            "message": "완료",
        },
        ensure_ascii=False,
        indent=2,
    )
    monkeypatch.setattr(runner, "spawn", lambda *a, **k: fake_process_factory(stdout=stdout))
    config_id = create_config(client, valid_payload)
    job_id = client.post("/api/train/jobs", json={"config_id": config_id}).json()["job_id"]
    for _ in range(500):
        if client.get(f"/api/train/jobs/{job_id}").json()["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)

    body = client.get(f"/api/train/jobs/{job_id}/evaluate").json()

    assert body["evaluation"]["status"] == "idle"


def test_evaluate_route_forwards_an_attached_test_manifest(client, monkeypatch):
    """과거 학습에 없던 test manifest도 이번 평가 요청으로 전달합니다."""

    from src.pipelines.web.api import routes_train
    from src.pipelines.web.jobs.model import JobRecord

    record = JobRecord(
        job_id="a" * 32,
        config_id="b" * 32,
        run_id="run-1",
        status="succeeded",
    )
    captured = {}

    class Manager:
        def get(self, job_id):
            assert job_id == record.job_id
            return record

    class Evaluation:
        def start(self, received_record, options):
            assert received_record is record
            captured.update(options)
            return {"status": "running", "submission_requested": True}

    monkeypatch.setattr(routes_train, "get_manager", lambda: Manager())
    monkeypatch.setattr(routes_train, "get_evaluation_runner", lambda: Evaluation())

    response = client.post(
        f"/api/train/jobs/{record.job_id}/evaluate",
        json={"test_manifest_uri": "s3://bucket/test/test_manifest.json"},
    )

    assert response.status_code == 202
    assert captured["test_manifest_uri"] == "s3://bucket/test/test_manifest.json"


def test_verify_accepts_artifact_uris_directly(client, monkeypatch):
    """준비 결과는 S3에 있을 수 있어 위치를 다시 훑을 수 없습니다."""

    from src.pipelines.web import datasets

    seen = {}

    def fake_verify(data_inputs):
        seen.update(data_inputs)
        return {"ok": True, "supported": True, "exit_code": 0, "artifacts": {},
                "summary": {}, "message": "완료"}

    monkeypatch.setattr(datasets, "verify_with_pipeline", fake_verify)
    data = {key: f"s3://bucket/p/{key}.json" for key in DATA_ARTIFACT_KEYS}

    body = client.post("/api/data/verify", json={"data": data}).json()

    assert body["verification"]["ok"] is True
    assert body["inspected"] is None  # 위치를 훑지 않습니다
    assert seen == data


def test_verify_rejects_incomplete_artifact_uris(client):
    response = client.post(
        "/api/data/verify", json={"data": {"train_manifest_uri": "artifacts/a.json"}}
    )

    assert response.status_code == 400
    fields = {item["field"] for item in response.json()["errors"]}
    assert "inputs.data.class_map_uri" in fields


def test_verify_without_a_target_is_rejected(client):
    response = client.post("/api/data/verify", json={})

    assert response.status_code == 400
    assert response.json()["errors"]


def test_prepare_status_starts_idle_and_lists_ratios(client):
    body = client.get("/api/data/prepare").json()

    assert body["split_ratios"] == ["8:2", "9:1"]
    assert body["preparation"]["status"] == "idle"


@pytest.mark.parametrize("ratio", ("8:2", "9:1"))
def test_prepare_can_be_started_for_each_ratio(client, monkeypatch, ratio):
    from src.pipelines.web import datasets

    monkeypatch.setattr(
        datasets,
        "prepare_dataset",
        lambda config: {
            "ok": True,
            "supported": True,
            "exit_code": 0,
            "artifacts": {key: f"artifacts/p/{key}.json" for key in DATA_ARTIFACT_KEYS},
            "summary": {"mode": "prepare", "split_ratio": ratio},
            "message": "준비 완료",
        },
    )

    response = client.post("/api/data/prepare", json={"split_ratio": ratio})

    assert response.status_code == 202
    assert response.json()["preparation"]["split_ratio"] == ratio


@pytest.mark.parametrize("bad", ("7:3", "80:20", "", "0.2"))
def test_prepare_rejects_other_ratios(client, bad):
    response = client.post("/api/data/prepare", json={"split_ratio": bad})

    assert response.status_code == 400
    assert any(item["field"] == "split_ratio" for item in response.json()["errors"])


def test_prepare_rejects_out_of_range_seed(client):
    assert client.post(
        "/api/data/prepare", json={"split_ratio": "8:2", "seed": -1}
    ).status_code == 422


def test_app_does_not_start_jobs_on_import(client, isolated_repo):
    assert client.get("/api/train/jobs").json()["jobs"] == []
    assert client.get("/api/train/jobs").json()["active_job_id"] is None
