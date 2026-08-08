# 006. 제출 CSV에서 보조 class를 제외한다

## 상태와 목적

제안. `005`의 `competition-plus-other` label 공간을 쓰면 대회에 없는 알약 62종이
`기타 알약`(`category_id` `999999`) 한 class로 묶인다. 이 class는 학습과 validation에서는
정상적인 class지만 **대회 제출 CSV에 실려서는 안 된다.** 지금 evaluate는 걸러내지 않는다.

## 무엇이 문제인가

`src/pipelines/evaluate/pipeline.py`가 test 예측을 그대로 넘긴다.

```python
submission_text = render_submission_csv(
    test_predictions,
    category_ids=test_category_ids,
)
```

`submission.py`에는 class 기준 필터가 없다. 그래서 model이 `기타 알약`을 예측하면
`category_id` `999999`가 제출 CSV의 행으로 나간다. 채점기에 그 값은 맞을 수 없고,
그 행이 차지한 자리만큼 실제로 채점될 예측을 덜 내보내게 된다.

## class map에서 빼는 우회는 안 된다

`class_map.json` → `test_manifest.json`의 `categories` → `test_category_ids`로 이어지므로
class map에서 `999999`를 빼면 `submission.py`가 "test manifest categories에 없는
category_id"라며 `PredictionError`를 올린다. 조용히 무시되는 것이 아니라 **evaluate 실행
전체가 실패한다.**

class map에서 빼면 train도 그 class를 배우지 못한다. 그러면 대회 밖 알약이 다시 라벨
없는 상태가 되어 "알약은 배경"으로 학습된다. 이 label 공간을 만든 이유가 없어진다.

## evaluate에 요청하는 변경

`render_submission_csv`에 넘기기 **전에** 제외 대상 category의 예측을 버려 주기를
요청한다. 설정으로 받고 기본값은 빈 목록이면 좋겠다.

```
evaluate.submission_excluded_category_ids: [999999]   # 기본값 []
```

세 가지를 함께 부탁한다.

- **기본값이 빈 목록**이면 `full`(118 class) 실행과 지금까지의 동작이 그대로 유지된다.
- **상수를 코드에 박지 말아 주기를** 바란다. `999999`는 `scripts/aihub_to_competition.py`의
  값이고 pipeline 사이의 계약이 아니다. evaluate가 `scripts/`를 import하는 것도 피해야 한다.
- **CSV를 만들기 전에** 걸러야 한다. `render_submission_csv`는 정렬한 뒤 `annotation_id`를
  1부터 붙이므로, 만들어진 뒤에 행을 지우면 번호에 구멍이 생긴다. 대회 요구사항은
  로우 개수만큼의 고유한 값이다.

## 이미지당 detection 상한보다 먼저 걸러 주기를 바란다

`filter_predictions`는 score로 한 번 걸러 정렬한 뒤 이미지당 `max_detections_per_image`
개만 남긴다. 기본값은 **4**다. 제외를 이 상한보다 **나중에** 적용하면 `기타 알약` 예측이
4칸 중 일부를 차지한 뒤 버려져, 그 이미지에서 제출되는 실제 예측이 4개보다 줄어든다.
상한 앞에서 걸러야 4칸을 대회 class로 채울 수 있다. 점수에 직접 영향을 주는 순서다.

## validation 경로에는 적용하지 말아 주기를 바란다

`filter_predictions`는 validation(`pipeline.py`의 첫 호출)과 test(두 번째 호출) 두 곳에서
쓰인다. 제외는 **test 호출에만** 넘겨야 한다. validation에도 적용하면 아래 지표 항목이
함께 바뀐다.

`기타 알약`은 학습에 쓰는 실제 class이므로 validation mAP에는 포함되는 것이 맞다. 그
class의 지표는 "대회 밖 알약을 알약으로 잡아냈는가"를 보여 주는 정보다. 빼야 하는 것은
제출 CSV뿐이다.

## 왜 data가 직접 고치지 않는가

`src/pipelines/evaluate/`는 evaluate 담당자 소유이고, 저장소 규칙상 다른 area를 직접
수정하지 않는다. 그래서 제안서로 요청한다.

## 언제 필요한가

`competition-plus-other`로 만든 산출물로 **첫 제출을 만들기 전**이다. 그때까지는 영향이
없고, `full` 산출물만 쓴다면 이 변경은 필요하지 않다.

## 호환성

- 기본값이 빈 목록이면 `metrics.json`과 `predictions` 형식, `artifacts` key 구성은
  바뀌지 않는다.
- 제외 대상이 있을 때만 `submission.csv`의 행 수가 줄고 `summary`의 제출 행 수도 그만큼
  줄어든다. 줄어든 수를 어디에 남길지는 evaluate 판단에 맡긴다.
- data와 train에는 영향이 없다. 두 pipeline은 `기타 알약`을 다른 class와 똑같이 다룬다.
