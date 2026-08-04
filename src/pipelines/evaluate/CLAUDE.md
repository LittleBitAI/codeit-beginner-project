# Evaluate Pipeline

이 directory는 evaluate pipeline 담당자가 소유합니다. 다른 pipeline의 내부 모듈을 import하지 않고, 공용 코드는 `src/common/`의 storage interface만 사용합니다.

## 책임

Validation manifest와 예측을 받아 COCO 스타일 detection metric과 예측 JSON을 만듭니다. 학습은 하지 않으며, 다른 component가 만든 artifact를 수정하거나 덮어쓰지 않습니다.

## 공개 interface

```python
from src.pipelines.evaluate import run

result = run(config)  # {"status", "artifacts", "summary", "message"}
```

성공하면 `artifacts`는 계약대로 `run_id`, `metrics_uri`, `predictions_uri`만 담습니다. 실패하면 `status="error"`, `artifacts={}`이며 부분 성공을 `ok`로 보고하지 않습니다.

## 설정 (`config["evaluate"]`)

| key | 기본값 | 설명 |
| --- | --- | --- |
| `run_id` | `inputs.train.run_id` → `evaluate-<UTC timestamp>` | 산출물 directory 이름 |
| `validation_manifest_uri` | `inputs.data.validation_manifest_uri` | 필수. JSONL manifest |
| `class_map_uri` | `inputs.data.class_map_uri` | 선택. 없으면 class 이름에 category_id 사용 |
| `checkpoint_uri` | `inputs.train.best_checkpoint_uri` | checkpoint 추론에 사용 |
| `predictions_input_uri` | 없음 | 지정하면 추론을 건너뛰고 이 예측으로 metric만 계산 |
| `output_dir` | `artifacts/evaluate/<run_id>` | 산출물 위치. `s3://`도 가능 |
| `metrics_filename` / `predictions_filename` | `metrics.json` / `predictions.json` | 산출물 file 이름 |
| `iou_thresholds` | `[0.50, 0.55, ..., 0.95]` | 0 초과 1 이하 숫자 list |
| `score_threshold` | `0.0` | 이 값 미만 confidence는 평가에서 제외 |
| `max_detections_per_image` | `4` | 이미지당 상위 detection 수. `false`면 제한 없음 |
| `device` | `"cpu"` | torch device 이름 |
| `seed` | `config["seed"]` → `0` | random 동작에 사용 |
| `overwrite` | `false` | 기존 산출물이 있으면 기본적으로 실패 |

`config["inputs"]`는 읽기만 하며 수정하지 않습니다. `config["execution"]["mode"] == "dummy"`이고 `config["evaluate"]`가 없으면 저장소 공통 dummy 실행으로 보고 평가를 건너뜁니다.

## 입력 형식

### validation manifest (JSONL, 한 줄이 이미지 한 장)

```json
{"image_id": "img-1", "image_uri": "datasets/.../img-1.jpg", "width": 640, "height": 480,
 "annotations": [{"category_id": 1, "bbox": [10, 10, 40, 40]}]}
```

`bbox`는 COCO와 같은 pixel 단위 `[x, y, width, height]`이며 이미지 범위를 벗어나면 오류입니다. `image_id`는 manifest 안에서 유일해야 합니다.

### class map (JSON)

`{"classes": [{"id": 1, "name": "..."}]}` 또는 `{"1": "..."}` 형식을 지원합니다.

### 예측 입력 (JSON, 선택)

```json
[{"image_id": "img-1", "category_id": 1, "bbox": [10, 10, 40, 40], "score": 0.93}]
```

`{"predictions": [...]}`로 감싼 형식도 읽습니다. manifest에 없는 `image_id`는 오류입니다.

### checkpoint (train pipeline 계약 요청 사항)

`predictions_input_uri` 없이 실행하면 checkpoint에서 다음 key를 읽습니다. 최상위 `model` object 안에 두어도 됩니다.

| key | 필수 | 설명 |
| --- | --- | --- |
| `architecture` | 필수 | `torchvision.models.detection`의 builder 이름 |
| `num_classes` | 필수 | background를 포함한 class 수 |
| `state_dict` (또는 `model_state_dict`) | 필수 | 해당 model의 state dict |
| `category_ids` | 선택 | model label index → dataset category_id 매핑 |

`category_ids`가 없으면 model label을 그대로 category_id로 사용합니다. 이 형식은 train 담당자와 확정해야 하는 계약이며, 형식이 다르면 추론을 시도하지 않고 명확한 오류로 중단합니다.

## 산출물

- `metrics.json`: 실행 metadata, `iou_thresholds`, `metrics`(`mAP`, `mAP50`, `mAP75`, `precision50`, `recall50`), `per_class` 목록
- `predictions.json`: 실행 metadata와 평가에 실제로 사용한 detection 목록(`bbox_format`은 `xywh`)

`iou_thresholds`에 0.5 또는 0.75가 없으면 계산하지 않은 `mAP50`, `mAP75`, `precision50`, `recall50`은 `0.0`이 아니라 `null`로 남습니다. `predictions.json`의 `bbox`와 `score`는 평가에 사용한 값 그대로 저장합니다. 반올림하면 저장된 predictions로 다시 평가할 때 metric이 달라질 수 있기 때문입니다.

Local 산출물 URI는 저장소 기준 상대 경로, S3 산출물은 `s3://bucket/key`로 반환합니다. 두 file 모두 UTF-8 without BOM, LF로 저장합니다. `metrics_filename`과 `predictions_filename`이 같아 한 산출물이 다른 산출물을 덮어쓰는 설정은 실행 전에 거부합니다.

Local 경로는 읽기와 쓰기 모두 저장소 root 안으로 제한합니다. `..`나 저장소 밖 절대 경로는 오류입니다.

## Metric 정의

pycocotools를 추가하지 않고 COCO 정의를 numpy로 구현합니다. 101-point interpolated AP, score 내림차순 greedy 매칭, ground truth가 없는 class는 mAP 평균에서 제외합니다(`per_class`에는 `ap: null`로 남습니다).

## 실패 처리

모든 실패는 `EvaluateError` 계열(`ConfigurationError`, `InputArtifactError`, `PredictionError`, `ArtifactWriteError`)로 모아 `status="error"`와 설명 message로 반환합니다. 산출물은 metric 계산이 모두 끝난 뒤에 씁니다.

두 번째 file 저장이 실패하면 **이번 실행에서 새로 만든** local file만 지웁니다. `overwrite=true`로 기존 file 위에 덮어쓴 경우 그 file은 실행 전부터 있던 산출물이므로 지우지 않습니다. S3 object는 어떤 경우에도 자동으로 지우지 않고 message로 알립니다.

## Test

```
python -m pytest src/pipelines/evaluate/tests -q
```

이전 pipeline 없이 계약 형식 fixture만으로 실행되며, checkpoint 추론 test도 작은 이미지 한 장을 CPU에서 처리합니다.
