# 002. Submission CSV 검사와 experiment index·summary 계약

## 상태와 목적

제안. 세 가지를 정한다. 첫째, Registry가 Evaluate의 제출 CSV를 001 규격에 대해
검사해 규격을 어긴 제출물이 실험으로 등록되지 않게 한다. 둘째, 실행이 끝난 실험을
재시작 뒤에도 찾을 수 있도록 run별 index sidecar를 남긴다. 셋째, Web이 읽을 공통
experiment summary 형식을 확정한다. 기존 실행과 exact-URI 조회는 그대로 동작해야
한다.

## Submission CSV 검사

Registry는 `evaluate.submission_uri`가 있고 local 경로이며 `verify_artifacts=true`일
때 CSV 내용을 001의 규격에 대해 검사한다. 검사 항목은 001에 이미 적힌 것만이다.

- header가 정확히 `annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score`
- 모든 행이 field 8개
- `annotation_id`가 파일에 나온 순서대로 `1..N` 정수
- `image_id`와 `category_id`는 정수, bbox 4개와 `score`는 유한한 실수(NaN·Infinity 거부)
- image 하나당 행 4개 이하
- 정렬은 `image_id` 오름차순, 같은 image 안에서 `score` 내림차순, 이어서
  `category_id`, `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` 오름차순
- header만 있고 행이 없는 CSV는 유효하다
- 파일은 BOM 없는 UTF-8이어야 한다

검사하지 않는 것: `category_id`가 test manifest에 있는지, score threshold가
적용되었는지. 001이 Evaluate의 책임으로 정한 부분이므로 Registry는 다시 판정하지
않는다.

위반은 `RegistryError` 하위의 새 typed error `InvalidSubmissionError`로 처리하고
기존 경계대로 `status="error"`로 반환한다. 검사는 record를 저장하기 **전에**
끝나므로 실패한 실행은 experiment record를 남기지 않는다.

`s3://` submission은 "원격 artifact는 AWS 접근 없이 참조만 기록한다"는 기존 정책을
그대로 따라 건너뛴다. `verify_artifacts=false`는 존재 확인·해시와 함께 내용 검사도
건너뛰지만, **URI shape 검증은 지금처럼 항상 수행한다.** 검사를 끄는 설정이 안전
장치까지 끄지는 않는다.

## Record schema 1.2

`schema_version`을 `"1.2"`로 올리고 record 최상위에 `submission_check` 하나를
추가한다. 기존 필드는 이름·의미·순서를 바꾸지 않으므로 1.1 record를 읽던 소비자는
그대로 동작한다.

```json
"submission_check": {
  "checked": true,
  "row_count": 123,
  "image_count": 40,
  "max_detections_per_image": 4,
  "skipped_reason": null
}
```

submission이 없거나 원격이거나 `verify_artifacts=false`이면 `checked`는 `false`,
숫자 필드는 `null`, `skipped_reason`에 한국어 사유를 적는다.

## Experiment summary

Registry는 record와 별개로 실험 하나를 요약한 문서를 발행한다. Web을 포함한 모든
소비자가 읽는 공통 형식이며 `summary_version`은 `"1"`이다.

```json
{
  "summary_version": "1",
  "run_id": "exp-0001",
  "created_at": "2026-08-06T00:00:00+00:00",
  "seed": 42,
  "schema_version": "1.2",
  "experiment_record_uri": "artifacts/registry/exp-0001/experiment_record.json",
  "metrics": {"mAP": 0.31, "mAP50": 0.55, "mAP75": 0.33,
              "precision50": 0.61, "recall50": 0.48},
  "metrics_source": "metrics_file",
  "artifacts": {"train_manifest_uri": "...", "submission_uri": null, "...": "..."},
  "verification": {"artifacts_checked": 11, "artifacts_hashed": 11,
                   "artifacts_skipped_remote": 0},
  "submission_check": {"...": "..."}
}
```

`metrics`는 Evaluate의 `metrics.json`에 있는 이름을 그대로 쓴다. Registry는 그
파일을 **방어적으로** 읽는다. 파일이 없거나 원격이거나 JSON이 깨졌거나 키가 없으면
예외를 내지 않고 모든 지표를 `null`로 두고 `metrics_source`를 `"unavailable"`로
적는다. 지표를 못 읽었다는 이유로 실행이 실패하지 않는다.

`artifacts`는 선언된 필수·선택 artifact key 전부를 담은 평면 map이며, 없는 선택
artifact는 `null`이다. 소비자가 키 존재 여부로 분기하지 않게 하기 위해서다.

## Index layout

summary는 `registry/index/<run_id>.json`에 저장한다. prefix는
`config["registry"]["index_prefix"]`로 바꿀 수 있고 기본값은 `registry/index`이다.
run마다 파일 하나를 새로 쓰므로 기존 파일을 덮어쓰지 않으며, `overwrite` 정책은
experiment record와 같은 설정을 따른다.

**index는 record에서 다시 만들 수 있는 cache이고 record가 진실이다.** 따라서 index
쓰기가 실패해도 실행은 실패하지 않는다. `status`는 `"ok"`로 두고 `summary`의
`index_status`에 `"failed"`를 적어 알린다. 성공하면 `"written"`이다. Registry는 빠진
index만 다시 만드는 자체 CLI를 제공한다.

Registry의 `artifacts`에 `experiment_summary_uri`를 추가한다. `src/main_pipeline.py`의
필수 artifact 목록은 바뀌지 않는다.

## 목록·검색·비교 interface

`src/common`에 세 함수를 추가한다.

```python
list_experiment_summaries(config, *, limit=None, offset=0) -> list[dict]
search_experiment_summaries(config, *, run_id_contains=None, created_after=None,
                            created_before=None, min_map=None, has_submission=None,
                            sort_by="created_at", descending=True, limit=None) -> list[dict]
compare_experiment_summaries(run_ids, config) -> dict
```

**기존 `read_experiment_record()`는 바꾸지 않는다.** 그 함수는 계속 exact-URI 하나만
읽고 prefix listing이나 최신 record 탐색을 하지 않는다. 새 함수는 index prefix만
읽는 별개 경로이며, 원하는 index가 없다고 해서 record를 추측해 fallback하지 않는다.
실패는 모두 기존 `ExperimentRegistryError`로 보고하고, public message에 입력 URI나
backend 오류 원문을 넣지 않는 기존 규칙을 그대로 따른다.

index 항목 하나를 읽지 못해도 목록 전체가 실패하지 않는다. 해당 항목은 건너뛴다.

## 영향 범위와 구현 소유권

Evaluate와 Train, Web의 public interface는 바꾸지 않는다. Registry 구현은 registry
owner가 자기 branch에서 맡고, `src/common`과 `contracts/README.md` 변경은 저장소
규칙대로 각각 단독 PR로 낸다.

Web의 `/api/train/experiments`를 이 summary로 바꿀지는 Web owner가 정한다. 이
제안은 Web이 읽을 형식만 확정하고 Web 코드를 수정하지 않는다.
