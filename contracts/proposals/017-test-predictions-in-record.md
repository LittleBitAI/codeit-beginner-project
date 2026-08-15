# 017 · experiment record가 test 예측 파일을 가리키게 한다

- 제안자: registry 작업자
- 대상: `src/pipelines/registry/` (record schema)
- 상태: 구현 완료 (이 제안과 같은 PR)

## 왜 제안서인가

registry 지침서는 이렇게 못 박고 있습니다.

> The record schema is what every consumer reads. Adding, renaming, or removing a
> field is a `contracts/proposals/` proposal, not an edit.

record에 field를 더하므로 제안서를 함께 남깁니다.

## 배경

로컬 검증은 mAP 0.935인데 Kaggle은 0.611입니다. 격차 0.324를 두고 색·흐림·해상도·
잡음 네 축을 재봤지만 로컬에서 만들 수 있는 최대 손상이 0.038로, 격차의 12%밖에
설명하지 못했습니다. 남은 유력한 설명은 **모델이 실물 개체를 외웠다**는 것이고,
그렇다면 서로 다르게 외운 모델을 합치는 앙상블이 가장 값싼 대응입니다.

이미 Kaggle 점수가 달린 체크포인트가 7개 있습니다. 재학습 없이 합칠 수 있습니다.

## 무엇이 막고 있었나

evaluate는 test 예측을 만들어 제출 CSV로 쓰고 **버렸습니다**. 합치려면 각 실행의
예측을 다시 읽어야 하는데, CSV는 사람이 올리는 형식이라 상자를 문자열에서 되돌려
읽어야 합니다.

evaluate가 같은 줄을 `test_predictions.json`으로도 남기도록 바뀌었고(`artifacts`에
`test_predictions_uri` 추가), registry가 그것을 기록하지 않으면 **어느 실행의 예측을
합쳤는지 재현할 수 없습니다.**

## 제안 내용

`OPTIONAL_ARTIFACT_KEYS["evaluate"]`에 `test_predictions_uri`를 더합니다.

```python
OPTIONAL_ARTIFACT_KEYS = {
    "data": ("test_manifest_uri",),
    "evaluate": ("submission_uri", "test_predictions_uri"),
}
```

선택 artifact이므로 기존 실행은 그대로 통과합니다. 있으면 다른 artifact와 **똑같은**
URI 안전성·검증·출처·해시 규칙을 따릅니다.

## 버전

| | 이전 | 이후 |
| --- | --- | --- |
| `SCHEMA_VERSION` | `1.2` | **`1.3`** |
| `SUMMARY_VERSION` | `"3"` | **`"4"`** |

둘 다 **key를 더하기만 합니다.** 기존 key는 이름과 뜻을 그대로 유지합니다. 옛 record와
summary는 다시 등록되기 전까지 옛 모양을 유지하므로, 읽는 쪽은 `test_predictions_uri`가
`null`인 것뿐 아니라 **아예 없는 경우도** 견뎌야 합니다.

## 소비자에게 미치는 영향

- **web**: `submission_uri`처럼 `.get()`으로 읽으므로 지금 그대로 동작합니다. 화면에
  새로 보일 것은 없습니다. 나중에 "이 실행들로 앙상블"을 붙일 때 이 값을 씁니다.
- **`src/common/experiment_registry.py`**: key 이름으로 읽으므로 영향 없습니다.
- **옛 record**: 이 key가 없습니다. 정상이며, 그 실행은 앙상블 재료가 없다는 뜻입니다.

## 검증

- registry test **136 passed**.
- 선택 artifact를 넣은 실행에서 검사·해시 개수가 11 → 12로 늘고, record의 evaluate
  key 순서가 `run_id, metrics_uri, predictions_uri, submission_uri,
  test_predictions_uri`인 것을 확인했습니다.
- 선택 artifact가 **없는** 옛 실행은 그대로 통과합니다(`artifacts_hashed == 9`).

## 함께 가는 변경

- evaluate: `test_predictions_uri` artifact 추가 (`pipeline/evaluate/test-predictions-artifact`)
- web: **변경 없음.** `output_dir`을 이미 지정하고 있어 파일이
  `experiments/completed/{run_id}/evaluate/` 아래 `metrics.json` 옆에 그대로 떨어집니다.
