# 002. Evaluate metric 화면 label 명확화

## 상태와 목적

제안. Evaluate가 내는 `precision50`과 `recall50`의 **정의가 바뀌었지만 값의 key
이름은 그대로**이므로, 화면 label을 그대로 두면 보는 사람이 예전 의미로 읽는다.
Web 화면의 label 문자열 두 개를 바꿔 이 오해를 막는 것이 목적이다.

계산 로직이나 API 응답 형식은 바뀌지 않는다. 순수하게 표시 문자열 변경이다.

## 무엇이 바뀌었나

Evaluate의 metric 계산을 pycocotools `COCOeval` 기반으로 옮기면서 두 값의 정의가
아래와 같이 달라졌다.

| key | 이전 정의 | 새 정의 |
| --- | --- | --- |
| `precision50` | IoU 0.5에서 PR 곡선의 마지막 점 | IoU 0.5, **confidence 0.5 이상** 예측만으로 집계한 `TP / (TP + FP)` |
| `recall50` | IoU 0.5에서 PR 곡선의 마지막 점 | IoU 0.5, **confidence 0.5 이상** 예측만으로 집계한 `TP / (TP + FN)` |

새 정의가 해석하기 쉽다. "confidence 0.5를 넘긴 예측 중 몇 개가 맞았나"로 바로
읽히고, 같은 기준의 `TP`/`FP`/`FN` 원본 수치가 `metrics.json`의
`analysis.by_iou`에 함께 저장되어 교차 확인도 된다.

문제는 **같은 이름에 다른 값**이 들어간다는 점이다. 이전 실행 결과와 나란히
비교하면 값이 달라 보이는데, 화면에는 그 이유가 드러나지 않는다.

`precision75`와 `recall75`도 같은 정의로 새로 추가되었다.

## Web에 요청하는 변경

`src/pipelines/web/frontend/src/components/EvaluatePanel.tsx`의 `METRICS`
배열에서 label 문자열 두 개를 바꿔 주기를 요청한다.

```
"Precision@50"  →  "Precision@IoU0.5 (score≥0.5)"
"Recall@50"     →  "Recall@IoU0.5 (score≥0.5)"
```

`key` 값(`precision50`, `recall50`)은 그대로 두면 되고, 배열에 항목을 더하거나
빼지 않아도 된다.

## 왜 Evaluate가 직접 고치지 않는가

`src/pipelines/web/`는 web 담당자 소유이고, 저장소 규칙상 다른 area를 직접
수정하지 않는다. 그래서 제안서로 요청한다.

## 일정

Evaluate는 **이 제안서의 머지를 기다리지 않고** 값을 채워서 내보낸다. 두 값을
`null`로 비워 두는 방안도 검토했지만, precision과 recall은 "mAP가 왜 높고
낮은지"를 설명하는 가장 기본적인 보조 지표이므로 결과물 최상위가 비어 있으면
"구현하지 않았다"로 읽힌다. `per_class`에는 값이 있는데 최상위만 비는 불일치도
생긴다.

따라서 이 제안서가 머지되기 전까지는 화면에 **옛 label과 새 의미의 값**이 함께
표시된다. 팀 내부에서 감당 가능한 범위로 판단했다. label이 바뀌면 이 시차는
사라진다.

## 호환성

- API 응답 형식과 `metrics.json`의 key 구성은 바뀌지 않는다.
- `DetectionMetrics` type과 `summary` 형식도 그대로다.
- 화면 문자열만 바뀌므로 다른 pipeline에는 영향이 없다.
