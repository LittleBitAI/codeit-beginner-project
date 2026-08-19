"""MMDetection backend checkpoint를 읽어 추론하는 경로의 test입니다.

계약은 `contracts/proposals/012-mmdetection-checkpoint-inference.md`입니다. train을
import하지 않으므로, 이 test가 두 pipeline 사이 약속을 지키는 유일한 그물입니다.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.pipelines.evaluate import mmdetection_backend, predictor
from src.pipelines.evaluate.errors import PredictionError


def _model_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "schema_version": 1,
        "input_size": 640,
        "resize": "longest_edge",
        "pad_multiple": 32,
    }
    config.update(overrides)
    return config


def _checkpoint(**overrides: Any) -> dict[str, Any]:
    document = {
        "backend": "mmdetection",
        "architecture": "dino_r50_4scale",
        "num_classes": 4,
        "model_state_dict": {},
        "model_config": _model_config(),
        "category_ids": [0, 10, 20, 30],
    }
    document.update(overrides)
    return document


class _FakeInstances:
    def __init__(self, bboxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor):
        self.bboxes = bboxes
        self.labels = labels
        self.scores = scores


class _FakeDataSample:
    def __init__(self) -> None:
        self.metainfo: dict[str, Any] = {}
        self.pred_instances: _FakeInstances | None = None

    def set_metainfo(self, metainfo: dict[str, Any]) -> None:
        self.metainfo.update(metainfo)


class _StubDetector(torch.nn.Module):
    """registry가 돌려주는 detector 대역입니다.

    진짜 `nn.Module`이라 `load_state_dict`가 실제로 strict하게 동작합니다. 그래야
    접두사를 벗긴 state가 정말 그대로 실리는지, 맞지 않는 state가 정말 걸리는지
    확인할 수 있습니다. mmcv의 컴파일된 확장이 없어 실제 detector는 만들지 못합니다.
    """

    def __init__(self, config: dict[str, Any], *, predicted_label: int = 2) -> None:
        super().__init__()
        self.config = config
        self.predicted_label = predicted_label
        self.seen_samples: list[_FakeDataSample] = []
        # 실제 detector처럼 중첩된 이름을 만들어 `detector.` 접두사를 벗긴 key가
        # 그대로 맞아야 하도록 합니다.
        self.backbone = torch.nn.Module()
        self.backbone.conv1 = torch.nn.Conv2d(1, 1, 1, bias=False)

    def predict(self, batch, samples, rescale=False):
        self.seen_samples.extend(samples)
        sample = samples[0]
        sample.pred_instances = _FakeInstances(
            bboxes=torch.tensor([[10.0, 20.0, 30.0, 60.0]]),
            labels=torch.tensor([self.predicted_label]),
            scores=torch.tensor([0.75]),
        )
        return [sample]


def _state(value: float = 0.0) -> dict[str, Any]:
    """`_StubDetector`에 strict하게 실리는 최소 state입니다."""

    return {"detector.backbone.conv1.weight": torch.full((1, 1, 1, 1), value)}


def _fake_dependencies(
    built: list[_StubDetector],
    *,
    predicted_label: int = 2,
    build_error: Exception | None = None,
    register_error: Exception | None = None,
) -> SimpleNamespace:
    def build(config):
        if build_error is not None:
            raise build_error
        detector = _StubDetector(config, predicted_label=predicted_label)
        built.append(detector)
        return detector

    def register():
        if register_error is not None:
            raise register_error

    return SimpleNamespace(
        models=SimpleNamespace(build=build),
        data_sample_type=_FakeDataSample,
        # 진짜 의존성은 mmengine의 ConfigDict를 줍니다. 여기서는 설정이 그대로
        # 전달되는지만 보면 되므로 dict로 충분합니다. ConfigDict가 정말 필요하다는
        # 것은 실제 model을 만드는 test가 확인합니다.
        config_type=dict,
        register=register,
    )


def _install(monkeypatch, dependencies: SimpleNamespace) -> None:
    monkeypatch.setattr(
        mmdetection_backend, "_import_mmdetection", lambda: dependencies
    )


MMDETECTION_PACKAGES = ("mmcv", "mmdet", "mmengine")


def _require_mmdetection():
    """package가 **아예 없을 때만** 건너뜁니다. 그 밖의 실패는 실패로 둡니다.

    가짜 registry는 설정 값이 mmdet에 받아들여지는지를 영영 확인하지 못하므로, 있는
    곳에서는 실제 model을 만들어 계약을 확인합니다.

    설치 실패까지 싸잡아 건너뛰면 안 됩니다. 그러면 잘못된 wheel, `mmcv._ext` 로딩
    실패, 맞지 않는 버전 조합, import 경로 회귀가 모두 **초록색 CI**로 보입니다.
    requirements가 mmdet을 설치하기 시작한 뒤에는 그 구분이 특히 중요합니다. 설치가
    깨진 것과 애초에 설치 대상이 아닌 것은 다릅니다.

    `find_spec`은 module을 실행하지 않으므로, 설치 여부만 보고 import 실패는
    그대로 드러납니다. 예를 들어 CUDA 연산자가 없는 `mmcv-lite`가 깔려 있으면
    여기서는 통과하고 뒤에서 실패합니다. 그것이 맞습니다.
    """

    for name in MMDETECTION_PACKAGES:
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"{name}이(가) 설치돼 있지 않습니다. requirements 밖의 선택 사항입니다.")
    return mmdetection_backend._import_mmdetection()


def test_unknown_backend_is_reported_instead_of_falling_back():
    """모르는 backend를 torchvision으로 추측해 읽으면 조용히 틀린 점수가 나옵니다."""

    with pytest.raises(PredictionError, match="backend"):
        predictor._build_model(_checkpoint(backend="onnx"), source="best.pt")


def test_mmdetection_checkpoint_needs_an_allowlisted_architecture():
    with pytest.raises(PredictionError, match="architecture"):
        predictor._build_model(
            _checkpoint(architecture="yolox_s"), source="best.pt"
        )


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        (None, "model_config"),
        (_model_config(schema_version=2), "schema_version"),
        (_model_config(resize="shortest_edge"), "resize"),
        (_model_config(input_size=0), "input_size"),
    ],
)
def test_unreadable_model_config_is_reported(model_config, message):
    with pytest.raises(PredictionError, match=message):
        mmdetection_backend.read_model_config(
            _checkpoint(model_config=model_config), source="best.pt"
        )


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(ImportError("No module named 'mmdet'"), id="not_installed"),
        # mmcv는 컴파일된 확장을 함께 싣습니다. 설치는 됐는데 확장이 깨졌거나 torch
        # 버전과 어긋나면 ImportError가 아니라 이쪽이 납니다.
        pytest.param(OSError("DLL load failed"), id="broken_compiled_extension"),
    ],
)
def test_real_dependency_failure_is_reported_as_an_install_problem(failure):
    """실제 `_import_mmdetection`이 어떤 설치 실패든 계약 오류로 바꾸는지 봅니다."""

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith(("mmdet", "mmengine")):
            raise failure
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked
    try:
        with pytest.raises(PredictionError, match="mmengine"):
            mmdetection_backend._import_mmdetection()
    finally:
        builtins.__import__ = real_import


@pytest.mark.parametrize(
    ("architecture", "detector_type"),
    [
        ("dino_r50_4scale", "DINO"),
        ("dino_swin_b_4scale", "DINO"),
        ("cascade_rcnn_swin_t_fpn", "CascadeRCNN"),
    ],
)
def test_each_architecture_is_rebuilt_and_receives_the_checkpoint_state(
    monkeypatch, architecture, detector_type
):
    """계약이 요구한 architecture 각각의 재생성·state 적용을 확인합니다.

    한쪽만 확인하면 다른 architecture는 registry 생성 경로를 한 번도 지나지 않은 채
    merge됩니다. 설정의 **모양**만 볼 수 있고 train과 값이 같은지는 확인하지 못합니다.
    threshold나 정규화 상수처럼 값만 다른 drift는 state_dict도 그대로 맞아서 여기서
    잡히지 않습니다. 그것은 `model_config.schema_version`이 맡습니다.
    """

    built: list[_StubDetector] = []
    _install(monkeypatch, _fake_dependencies(built))

    model = predictor._build_model(
        _checkpoint(architecture=architecture, model_state_dict=_state(2.0)),
        source="best.pt",
    )

    detector = built[0]
    config = detector.config
    assert config["type"] == detector_type
    # checkpoint의 num_classes 4에서 background를 뺀 3이 MMDetection으로 갑니다.
    heads = (
        [config["bbox_head"]]
        if detector_type == "DINO"
        else config["roi_head"]["bbox_head"]
    )
    assert [head["num_classes"] for head in heads] == [3] * len(heads)
    # 접두사를 벗긴 state가 strict하게 실렸는지 값으로 확인합니다.
    assert torch.equal(
        detector.backbone.conv1.weight.detach(), torch.full((1, 1, 1, 1), 2.0)
    )
    assert detector.training is False
    assert isinstance(model, mmdetection_backend.MMDetectionPredictor)


def test_image_is_resized_to_the_long_edge_and_padded_to_a_multiple_of_32():
    image = torch.ones((3, 4, 8), dtype=torch.float32)

    batch, metainfo = mmdetection_backend.prepare_image(image, input_size=6)

    assert batch.shape == (1, 3, 32, 32)
    assert metainfo["ori_shape"] == (4, 8)
    assert metainfo["img_shape"] == (3, 6)
    assert metainfo["pad_shape"] == (32, 32)
    # DINO의 pre_transformer가 batch 전체 크기를 읽습니다.
    assert metainfo["batch_input_shape"] == (32, 32)
    assert metainfo["scale_factor"] == (0.75, 0.75)


def test_predictions_return_original_coordinates_and_one_based_labels():
    """MMDetection은 foreground를 0부터 셉니다. category_ids는 1부터 씁니다."""

    metainfo = {
        "ori_shape": (100, 200),
        "img_shape": (50, 100),
        "scale_factor": (0.5, 0.5),
    }
    instances = _FakeInstances(
        bboxes=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        labels=torch.tensor([0]),
        scores=torch.tensor([0.9]),
    )

    output = mmdetection_backend.to_output(instances, metainfo=metainfo)

    assert torch.equal(output["labels"], torch.tensor([1]))
    assert torch.allclose(output["boxes"], torch.tensor([[20.0, 40.0, 60.0, 80.0]]))
    assert torch.allclose(output["scores"], torch.tensor([0.9]))


@pytest.mark.parametrize(
    ("predicted_label", "category_id"),
    [(0, 10), (2, 30)],
)
def test_first_and_last_labels_reach_the_right_category_id(
    monkeypatch, predicted_label, category_id
):
    """MMDetection의 첫 foreground 0과 마지막 N-1이 양끝 category로 가야 합니다.

    가운데 label만 확인하면 1을 더하는 자리와 빼는 자리가 함께 틀려도 통과합니다.
    """

    built: list[_StubDetector] = []
    _install(monkeypatch, _fake_dependencies(built, predicted_label=predicted_label))
    checkpoint = _checkpoint(
        model_state_dict=_state(),
        model_config=_model_config(input_size=64),
    )
    model = predictor._build_model(checkpoint, source="best.pt")

    outputs = model([torch.ones((3, 32, 64))])

    detector = built[0]
    assert detector.seen_samples[0].metainfo["batch_input_shape"] == (32, 64)
    predictions = predictor._outputs_to_predictions(
        outputs[0],
        record={"image_id": "img-1", "image_key": "img-1"},
        category_ids=checkpoint["category_ids"],
    )
    assert [entry["category_id"] for entry in predictions] == [category_id]


@pytest.mark.parametrize(
    ("category_ids", "expected"),
    [
        pytest.param(None, "category_ids", id="missing"),
        pytest.param([0, 10, 20], "num_classes", id="shorter_than_num_classes"),
        pytest.param([0, 10, 20, 30, 40], "num_classes", id="longer_than_num_classes"),
        pytest.param([0, 10, 20, "30"], "category_ids", id="not_integers"),
    ],
)
def test_mmdetection_checkpoint_needs_one_category_id_per_class(
    monkeypatch, category_ids, expected
):
    """category_ids가 없거나 길이가 다르면 조용히 다른 약으로 채점됩니다.

    없으면 model label을 그대로 COCO category id로 씁니다. 짧으면 예측 label이 우연히
    범위 안일 때 다른 약의 category id가 나옵니다. 둘 다 오류처럼 보이지 않고 점수만
    틀리기 때문에 checkpoint를 읽는 자리에서 멈춰야 합니다.
    """

    built: list[_StubDetector] = []
    _install(monkeypatch, _fake_dependencies(built))

    with pytest.raises(PredictionError, match=expected):
        predictor._build_model(
            _checkpoint(category_ids=category_ids, model_state_dict=_state()),
            source="best.pt",
        )

    # 모델을 만든 뒤에 걸러 내면 무거운 생성이 끝난 다음에야 알게 됩니다.
    assert built == []


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        pytest.param("2.1.0", None, id="아래는_손대지_않는다"),
        pytest.param("2.2.0", "2.1.999", id="직접_확인한_그_버전"),
        pytest.param("2.2.0+a8073c7pt2.12.0cu126", "2.1.999", id="local_꼬리표는_떼고_본다"),
        pytest.param("2.2.1", None, id="확인하지_않은_patch는_열지_않는다"),
        pytest.param("2.3.0", None, id="확인하지_않은_minor는_열지_않는다"),
        pytest.param("3.0.0", None, id="다음_major는_열지_않는다"),
    ],
)
def test_mmcv_version_shim_only_covers_the_verified_range(version, expected):
    """mmdet 3.3.0의 상한이 mmcv 2.2.0을 막지만 그 버전은 실제로는 맞습니다.

    범위로 열면 아직 나오지도 않은 2.2.1까지 함께 통과해, 정말로 맞지 않는 조합이
    설치 문제로 보고되는 대신 알 수 없는 자리에서 깨집니다. 직접 확인한 그 버전
    하나만 통과시켜야 합니다.
    """

    assert mmdetection_backend._shimmed_mmcv_version(version) == expected


def test_real_dependencies_expose_the_config_type():
    """two-stage detector는 train_cfg를 속성으로 읽어 평범한 dict로는 못 만듭니다."""

    dependencies = _require_mmdetection()

    assert dependencies.config_type is not None
    # 중첩된 dict까지 함께 바뀌어야 roi_head 안쪽도 속성으로 읽힙니다.
    converted = dependencies.config_type({"a": {"b": 1}})
    assert converted.a.b == 1


def test_this_pipeline_can_build_every_mmdetection_model_the_contract_names():
    """계약이 이름을 정하고, 그 이름으로 무엇을 만들지는 각 pipeline이 정합니다.

    허용 목록은 계약 전체를 그대로 받는데(`SUPPORTED_ARCHITECTURES`) 만들 줄 아는
    것은 여기 적힌 둘뿐입니다. 계약에 세 번째 이름이 생기면 GUI는 그것을 내놓고
    train은 학습까지 마치는데, 평가만 마지막에 거부합니다 — 밤새 학습한 뒤에.
    """

    buildable = set()
    for architecture in mmdetection_backend.SUPPORTED_ARCHITECTURES:
        try:
            mmdetection_backend.build_detector_config(architecture, foreground_classes=3)
        except PredictionError:
            continue
        buildable.add(architecture)

    assert buildable == set(mmdetection_backend.SUPPORTED_ARCHITECTURES)


@pytest.mark.parametrize(
    ("architecture", "detector_type"),
    [
        ("dino_r50_4scale", "DINO"),
        ("cascade_rcnn_swin_t_fpn", "CascadeRCNN"),
    ],
)
def test_real_model_is_rebuilt_and_predicts_in_original_coordinates(
    architecture, detector_type
):
    """진짜 mmdet model로 계약을 끝까지 확인합니다.

    가짜 registry는 설정이 mmdet에 **받아들여지는지**를 확인하지 못합니다. 실제로
    `cascade_rcnn_swin_t_fpn`은 평범한 dict를 넘기던 동안 만들어지지도 않았는데,
    가짜를 쓰던 test는 모두 통과했습니다.
    """

    dependencies = _require_mmdetection()
    dependencies.register()
    num_classes = 4
    reference = dependencies.models.build(
        dependencies.config_type(
            mmdetection_backend.build_detector_config(
                architecture, foreground_classes=num_classes - 1
            )
        )
    )
    assert type(reference).__name__ == detector_type
    checkpoint = _checkpoint(
        architecture=architecture,
        num_classes=num_classes,
        model_state_dict={
            f"detector.{name}": value
            for name, value in reference.state_dict().items()
        },
        model_config=_model_config(input_size=320),
    )

    model = predictor._build_model(checkpoint, source="best.pt")
    with torch.no_grad():
        outputs = model([torch.rand(3, 240, 320)])

    output = outputs[0]
    assert output["boxes"].shape[0] == output["labels"].shape[0]
    if output["labels"].numel():
        # MMDetection이 0부터 세는 label을 저장소의 1..N으로 되돌려야 합니다.
        assert int(output["labels"].min()) >= 1
        assert int(output["labels"].max()) <= num_classes - 1
        # box는 padding한 크기가 아니라 원본 이미지 좌표여야 합니다.
        assert float(output["boxes"][:, 2].max()) <= 320 * 1.05
        assert float(output["boxes"][:, 3].max()) <= 240 * 1.05
    predictions = predictor._outputs_to_predictions(
        output,
        record={"image_id": "img-1", "image_key": "img-1"},
        category_ids=checkpoint["category_ids"],
    )
    # category_ids[0]은 background 자리라 예측에 나오면 안 됩니다.
    assert {entry["category_id"] for entry in predictions} <= {10, 20, 30}


@pytest.mark.parametrize(
    "state_dict",
    [
        pytest.param({"backbone.conv1.weight": torch.zeros(2)}, id="unprefixed"),
        pytest.param(
            {
                "detector.backbone.conv1.weight": torch.zeros(2),
                "backbone.conv2.weight": torch.zeros(2),
            },
            id="mixed",
        ),
        pytest.param({}, id="empty"),
    ],
)
def test_state_dict_must_be_entirely_detector_prefixed(monkeypatch, state_dict):
    """train은 detector를 감싼 adapter의 state를 저장합니다.

    접두사가 없는 key를 조용히 버리거나 그대로 쓰면, 일부 가중치만 실린 model로
    점수를 내고도 아무도 눈치채지 못합니다.
    """

    _install(monkeypatch, _fake_dependencies([]))

    with pytest.raises(PredictionError, match="detector."):
        predictor._build_model(
            _checkpoint(model_state_dict=state_dict), source="best.pt"
        )


def test_mmdetection_checkpoint_needs_a_background_class(monkeypatch):
    """num_classes는 background를 포함하므로 1이면 foreground가 0개가 됩니다."""

    _install(monkeypatch, _fake_dependencies([]))

    with pytest.raises(PredictionError, match="num_classes"):
        predictor._build_model(_checkpoint(num_classes=1), source="best.pt")


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param({"register_error": KeyError("scope")}, id="register"),
        pytest.param({"build_error": ValueError("bad config")}, id="build"),
    ],
)
def test_registry_failures_become_prediction_errors(monkeypatch, failure):
    """run()은 EvaluateError만 잡습니다. 서드파티 오류가 새어 나가면 실행이 죽습니다."""

    _install(monkeypatch, _fake_dependencies([], **failure))

    with pytest.raises(PredictionError, match="best.pt"):
        predictor._build_model(
            _checkpoint(model_state_dict=_state()),
            source="best.pt",
        )


def test_state_that_does_not_fit_the_model_is_reported(monkeypatch):
    """모양이 맞지 않는 state는 torch가 냅니다. 그대로 두면 run() 밖으로 새어 나갑니다."""

    _install(monkeypatch, _fake_dependencies([]))

    with pytest.raises(PredictionError, match="best.pt"):
        predictor._build_model(
            _checkpoint(
                model_state_dict={"detector.backbone.conv1.weight": torch.zeros(7)}
            ),
            source="best.pt",
        )


def test_inference_failure_becomes_a_prediction_error(monkeypatch):
    built: list[_StubDetector] = []
    _install(monkeypatch, _fake_dependencies(built))
    model = predictor._build_model(
        _checkpoint(model_state_dict=_state()),
        source="best.pt",
    )

    def broken(batch, samples, rescale=False):
        raise RuntimeError("CUDA out of memory")

    built[0].predict = broken

    with pytest.raises(PredictionError, match="추론"):
        model([torch.ones((3, 32, 64))])


@pytest.mark.parametrize("stage", ["to_device", "data_sample"])
def test_device_move_and_sample_construction_failures_are_converted(monkeypatch, stage):
    """추론 직전 준비 단계도 MMDetection의 몫이라 계약 오류로 바꿔야 합니다.

    바깥 predictor는 device 오류를 RuntimeError·ValueError·AssertionError만 잡습니다.
    그 밖의 예외가 여기서 새어 나가면 run()이 잡지 못해 실행 자체가 죽습니다.
    """

    built: list[_StubDetector] = []
    dependencies = _fake_dependencies(built)
    if stage == "data_sample":

        class _BrokenDataSample(_FakeDataSample):
            def set_metainfo(self, metainfo):
                raise KeyError("batch_input_shape")

        dependencies.data_sample_type = _BrokenDataSample
    _install(monkeypatch, dependencies)
    model = predictor._build_model(
        _checkpoint(model_state_dict=_state()),
        source="best.pt",
    )

    if stage == "to_device":
        def broken_to(device):
            raise OSError("driver is missing")

        built[0].to = broken_to
        with pytest.raises(PredictionError, match="device"):
            model.to("cuda")
        return

    with pytest.raises(PredictionError, match="추론"):
        model([torch.ones((3, 32, 64))])


def test_malformed_detector_output_becomes_a_prediction_error(monkeypatch):
    built: list[_StubDetector] = []
    _install(monkeypatch, _fake_dependencies(built))
    model = predictor._build_model(
        _checkpoint(model_state_dict=_state()),
        source="best.pt",
    )
    built[0].predict = lambda batch, samples, rescale=False: []

    with pytest.raises(PredictionError, match="추론"):
        model([torch.ones((3, 32, 64))])
