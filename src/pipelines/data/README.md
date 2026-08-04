# Data pipeline

`run(config) -> dict` 하나만 공개합니다. 반환값은 공통 계약대로 `status`,
`artifacts`, `summary`, `message` 네 key입니다.

## 실행 경로 세 가지

| 조건 | 하는 일 |
| --- | --- |
| `execution.mode == "dummy"` | 기존 dummy 결과를 그대로 반환합니다. |
| `data.prepare == true` | 원본에서 학습용 artifact 4개를 만들어 저장하고 위치를 공개합니다. |
| 그 외 | `inputs.data`의 URI 4개를 검증해 그대로 공개합니다. |

`data.prepare`를 켜지 않으면 준비 경로는 동작하지 않습니다. 값은 `true` 또는
`false`만 허용하며, 다른 값은 오류로 알려 줍니다.

## 준비 경로 설정

```json
{
  "storage": {
    "backend": "s3",
    "s3": { "bucket": "<bucket 이름>" }
  },
  "data": {
    "prepare": true,
    "split_ratio": "8:2",
    "seed": 42,
    "overwrite": false
  }
}
```

| key | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `prepare` | 예 | `false` | `true`일 때만 준비 경로를 실행합니다. |
| `split_ratio` | 예 | 없음 | `"8:2"`(validation 0.2) 또는 `"9:1"`(validation 0.1). **다른 값은 모두 거부**합니다. |
| `seed` | 아니요 | `42` | 분할에 쓰는 seed. 0 이상 2**32 미만의 정수. |
| `overwrite` | 아니요 | `false` | `false`면 이미 있는 산출물을 덮어쓰지 않고 오류를 냅니다. |
| `raw_prefix` | 아니요 | `datasets/pill_detection/raw/v1/` | 원본 prefix. `datasets/`로 시작해야 합니다. |
| `processed_root` | 아니요 | `datasets/pill_detection/processed/` | 산출물 상위 prefix. `datasets/`로 시작해야 합니다. |

Storage backend, bucket, credential은 `src/common/storage.py`의
`create_storage(config)`와 환경 변수(`PILL_STORAGE_*`, `AWS_*`)가 담당합니다.
이 pipeline은 `boto3`를 직접 쓰지 않습니다.

## 저장 위치

산출물 directory 이름에 seed와 비율이 들어가므로 **8:2와 9:1의 산출물은 서로
덮어쓰지 않습니다.**

```
{processed_root}v1-seed{seed}-{8020|9010}/
├── train_manifest.json
├── validation_manifest.json
├── class_map.json
└── dataset_summary.json
```

기본 설정에서는 다음과 같습니다.

- 8:2 → `datasets/pill_detection/processed/v1-seed42-8020/`
- 9:1 → `datasets/pill_detection/processed/v1-seed42-9010/`

## 산출물 형식

### `train_manifest.json`, `validation_manifest.json`

COCO 형식입니다. `file_name`에는 원본 이미지의 **절대 위치**(S3 backend면
`s3://` URI)를 넣습니다. manifest는 `processed/`, 이미지는 `raw/` 아래에 있어서
상대경로로 두면 소비자가 엉뚱한 위치를 찾기 때문입니다.

```json
{
  "info": {
    "description": "Pill detection processed COCO manifest",
    "split": "train",
    "seed": 42,
    "split_ratio": "8:2",
    "validation_ratio": 0.2
  },
  "images": [
    {
      "id": 1,
      "file_name": "s3://<bucket>/datasets/pill_detection/raw/v1/train_images/img_001.jpg",
      "width": 976,
      "height": 1280
    }
  ],
  "annotations": [
    { "id": 1, "image_id": 1, "category_id": 7, "bbox": [10.0, 12.0, 90.0, 80.0], "iscrowd": 0 }
  ],
  "categories": [{ "id": 7, "name": "pill_a" }]
}
```

### `class_map.json`

원본 COCO category id를 그대로 남기는 `{"<category id>": "<name>"}` 형태입니다.
소비자는 category id 오름차순으로 1부터 이어지는 model label을 붙입니다(0은
background).

```json
{ "7": "pill_a", "12": "pill_b" }
```

### `dataset_summary.json`

어떤 원본을 어떤 비율과 seed로 나눴는지, category 분포와 제외 이미지가 무엇인지
남기는 JSON object입니다. `generated_at`만 실행 시각이고 나머지 값은 같은 입력에
대해 항상 같습니다.

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-08-05T00:00:00Z",
  "source_prefix": "datasets/pill_detection/raw/v1/",
  "processed_prefix": "datasets/pill_detection/processed/v1-seed42-8020/",
  "split": {
    "method": "deterministic_multilabel_distribution_preserving",
    "split_ratio": "8:2",
    "validation_ratio": 0.2,
    "seed": 42
  },
  "raw": { "listed_train_images": 40, "annotation_documents": 40, "unreferenced_train_images": 0, "test_images_used": 0 },
  "excluded_images": [],
  "train_images": 32,
  "validation_images": 8,
  "category_count": 3,
  "categories": [{ "id": 7, "name": "pill_a", "model_label": 1, "train_image_count": 10, "validation_image_count": 3 }],
  "class_distribution": { "train": { "pill_a": 10 }, "validation": { "pill_a": 3 } },
  "artifacts": { "train_manifest_uri": "s3://...", "validation_manifest_uri": "s3://...", "class_map_uri": "s3://..." }
}
```

## 보장하는 것

- **유출 방지**: `train_images/`와 `train_annotations/`만 읽습니다. competition
  평가용 `test_images/`는 목록 단계에서 제외되어 어떤 split에도 들어가지 않고,
  prefix 설정으로도 지정할 수 없습니다.
- **category 보장**: 희귀 category부터 validation에 배치해 모든 category가 train과
  validation 양쪽에 나타납니다. 이미지 1장에만 있는 category가 있으면 오류입니다.
- **재현성**: 같은 원본, 같은 seed, 같은 비율이면 항상 같은 분할이 나옵니다.
- **덮어쓰기 방지**: 기본값은 덮어쓰지 않기이며, 산출물이 이미 있으면 아무것도
  쓰지 않고 오류로 알려 줍니다.
- **오류 처리**: 실패해도 예외를 밖으로 던지지 않고 `status: "error"`와 사람이
  읽을 message를 반환합니다. message에는 credential이나 개인 절대경로를 넣지
  않습니다.

## Module 구성

| file | 역할 |
| --- | --- |
| `__init__.py` | 공개 `run(config)`와 실행 경로 선택 |
| `preparation.py` | config 검증, 원본 목록, artifact 생성과 저장 |
| `coco.py` | 이미지별 COCO 문서를 하나의 dataset으로 합치기 |
| `split.py` | 결정적 train/validation 분할 |
| `errors.py` | 준비 경로 공개 예외 |

## 테스트

```
python -m pytest src/pipelines/data -q
```

테스트는 실제 S3에 접속하지 않고 `unittest.mock`으로 만든 in-memory storage
대역을 사용합니다.
