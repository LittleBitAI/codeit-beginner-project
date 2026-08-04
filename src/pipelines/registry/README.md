# Registry pipeline

data, train, evaluate pipeline이 만든 artifact를 모아 재현 가능한 experiment
record 하나를 생성합니다. 외부에 공개하는 interface는 `run(config) -> dict`
하나입니다.

## registry가 요구하는 입력 artifact

registry는 `config["inputs"]`에서 이전 pipeline 결과를 읽고 **수정하지 않습니다**.
아래 key는 모두 **필수**이며 값의 타입은 모두 **비어 있지 않은 `str`** 입니다.
`bool`, `int`, `None`, `list`, 중첩 `dict`는 모두 오류입니다.

### `config["inputs"]["data"]`

| key | 타입 | 설명 |
| --- | --- | --- |
| `train_manifest_uri` | `str` | 학습 manifest artifact 위치 |
| `validation_manifest_uri` | `str` | 검증 manifest artifact 위치 |
| `class_map_uri` | `str` | class 이름과 id 대응표 위치 |
| `dataset_summary_uri` | `str` | dataset 요약 위치 |

### `config["inputs"]["train"]`

| key | 타입 | 설명 |
| --- | --- | --- |
| `run_id` | `str` | 학습 실행 식별자. URI가 아니며 해시 대상이 아닙니다. |
| `best_checkpoint_uri` | `str` | 최고 성능 checkpoint 위치 |
| `last_checkpoint_uri` | `str` | 마지막 checkpoint 위치 |
| `training_history_uri` | `str` | 학습 이력 위치 |

### `config["inputs"]["evaluate"]`

| key | 타입 | 설명 |
| --- | --- | --- |
| `run_id` | `str` | 평가 실행 식별자. `train.run_id`와 같아야 합니다. |
| `metrics_uri` | `str` | 평가 지표 위치 |
| `predictions_uri` | `str` | 예측 결과 위치 |

### URI 규칙

- 이름이 `_uri`로 끝나는 key만 실제 artifact 파일을 가리킵니다.
- local artifact는 **저장소 root 기준 상대 경로**를 씁니다. 절대 경로와 저장소
  밖을 가리키는 경로(`../`)는 오류입니다.
- 원격 artifact는 `s3://bucket/key` 형식을 씁니다.
- 위 표에 없는 key가 더 들어와도 오류는 아니며, record에 기록되지 않고 무시됩니다.

## key 누락과 타입 오류 시 동작

검증은 `record.validate_inputs()`가 담당하며, 파일을 쓰기 전에 모두 끝납니다.
따라서 검증에 실패하면 experiment record는 **아예 생성되지 않습니다**(부분 성공
없음). 내부에서는 아래 예외가 발생하고, `run()`은 이를 잡아 `status: "error"`와
한국어 `message`로 바꿔 반환합니다.

| 상황 | 예외 | message 예시 |
| --- | --- | --- |
| pipeline 자체가 없음 | `MissingInputError` | `evaluate pipeline의 artifact가 없습니다. ...` |
| 필수 key 누락 | `MissingInputError` | `train pipeline artifact에 필요한 key가 없습니다: training_history_uri` |
| 값 타입이 `str`이 아님 | `InvalidSchemaError` | `config['inputs']['data']['class_map_uri']는 str이어야 하는데 int을(를) 받았습니다.` |
| 값이 빈 문자열 | `InvalidSchemaError` | `... 비어 있지 않은 문자열이어야 합니다.` |
| pipeline 값이 object가 아님 | `InvalidSchemaError` | `config['inputs']['train']는 object여야 합니다.` |
| `train.run_id != evaluate.run_id` | `InvalidSchemaError` | `train과 evaluate의 run_id가 서로 다릅니다: ...` |
| local artifact 경로가 절대/저장소 밖 | `InvalidSchemaError` | `local artifact는 저장소 기준 상대 경로여야 합니다: ...` |
| local artifact file 없음/읽기 실패 | `CorruptedArtifactError` | `local artifact file을 찾을 수 없습니다: ...` |

세 예외 모두 `RegistryError`를 상속하므로, registry 내부에서 한 번에 잡을 수
있습니다.

## registry 자기 설정 (`config["registry"]`, 전부 선택)

| key | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `run_id` | `str` | train/evaluate의 `run_id` 승계 | record의 실행 식별자 |
| `record_uri` | `str` | `registry/<run_id>/experiment_record.json` | record 저장 위치 (storage root 기준) |
| `overwrite` | `bool` | `false` | 이미 있는 record를 덮어쓸지 여부 |
| `verify_artifacts` | `bool` | `true` | local artifact의 sha256 계산 여부 |
| `repo_root` | `str` | module 위치에서 계산한 저장소 root | local artifact를 찾을 기준 경로 |
| `seed` | `int` | `config["seed"]` 또는 `0` | record에 남길 seed |

registry에는 무작위 동작이 없습니다. `run_id`를 새로 만들지 않고 이전 pipeline
값을 그대로 승계하므로, seed는 재현 정보로 기록만 합니다.

## 실행

```bash
# dummy 포함 전체 pipeline
python -m src.main_pipeline

# registry 단독
python -m src.main_pipeline --only registry

# 담당 테스트 (network, AWS, GPU 불필요)
python -m pytest src/pipelines/registry/tests -q

# 실제 S3 왕복 smoke test (AWS credential과 bucket 필요, 비용 발생 가능)
python -m src.pipelines.registry.smoke_s3 --config configs/env.aws.json
```

S3 smoke test는 `registry/smoke-tests/` prefix 아래에 작은 record 하나를
등록하고 다시 조회합니다. `scripts/s3_smoke_test.py`와 같은 정책으로 생성한
object를 **자동 삭제하지 않으며**, 결과의 `experiment_record_uri`로 남은 object를
알려 줍니다.
