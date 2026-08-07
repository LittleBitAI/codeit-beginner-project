# 004. Train 상세 loss와 early stopping 계약

## 상태와 목적

제안. Train은 현재 epoch별 total loss만 저장하고 진행 로그로 보낸다. 모델이 반환한
상세 loss를 이름을 제한하지 않고 확인하고, validation loss가 개선되지 않으면 학습을
일찍 끝낼 수 있도록 Train과 Web의 입력·진행 로그 계약을 확장한다.

기존 `run(config)`의 네 key와 artifact URI key, stdout 출력은 바꾸지 않는다.
Evaluate와 Registry는 각각 기존 best checkpoint와 artifact URI를 계속 사용한다.

## Train config

`config["train"]`에 선택 객체 `early_stopping`을 추가한다.

```json
"early_stopping": {"patience": 5, "min_delta": 0.0}
```

- 객체가 없으면 비활성화해 기존 실행 결과를 보존한다.
- `patience`는 필수이며 bool이 아닌 1 이상의 정수다.
- `min_delta`는 선택이며 기본값은 `0.0`이다. bool이 아닌 0 이상의 유한한 수다.
- 모르는 key, 잘못된 타입과 범위는 artifact를 만들기 전에 거부한다.
- Web은 같은 기본값과 검증 규칙을 복제하고 입력 form에서 객체를 구성한다.

전체 `validation_loss`가 기준 loss보다 `min_delta`를 초과해 낮아질 때 patience를
초기화한다. 연속 미개선 횟수가 `patience`에 도달하면 그 epoch까지 기록하고 끝낸다.
Best checkpoint는 min_delta와 별개로 실제 최저 validation loss를 보존하고, last
checkpoint는 실제 마지막 수행 epoch를 보존한다.

## 상세 loss 형식

`training_history.json`의 최상위 배열과 기존 세 필드는 유지하고 두 객체를 추가한다.

```json
{
  "epoch": 1,
  "train_loss": 1.25,
  "validation_loss": 1.38,
  "train_loss_components": {"classification": 0.72, "bbox_regression": 0.53},
  "validation_loss_components": {"classification": 0.79, "bbox_regression": 0.59}
}
```

두 component 객체는 `string -> finite number` mapping이다. Faster R-CNN, RetinaNet 등
모델이 반환한 key를 그대로 보존하며 이름을 열거하거나 공통 이름으로 번역하지 않는다.
값은 기존 total과 같은 batch 평균이고, component 합은 부동소수점 오차 범위에서
total과 같다.

`train.progress/1`의 `epoch_completed`에도 같은 이름의 두 optional 객체를 추가한다.
기존 Web은 모르는 필드를 무시할 수 있으므로 major version은 유지한다.

## 학습 완료 형식

Train은 artifact publish까지 성공한 뒤 다음 event를 stderr에 추가한다.

```json
{
  "schema": "train.progress/1",
  "event": "training_completed",
  "run_id": "exp-0001",
  "planned_epochs": 50,
  "completed_epochs": 12,
  "stopped_early": true,
  "best_epoch": 7,
  "best_validation_loss": 0.41,
  "ts": "2026-08-07T03:35:20.123456Z"
}
```

Train summary는 기존 `epochs`를 계획값으로 유지하고 `planned_epochs`,
`completed_epochs`, `stopped_early`를 추가한다. Checkpoint의 `training_config`에는
정규화한 `early_stopping`을 `null` 또는 객체로 추가하고 기존 schema version은
유지한다.

Web은 `training_completed`를 받으면 진행률을 100%, ETA를 0으로 만들고 실제 완료
epoch와 조기 종료 여부를 표시한다. Event나 새 필드가 없는 옛 실행도 기존처럼 읽는다.

## 구현 순서와 소유권

1. 이 제안을 Train과 Web 담당자가 합의한다.
2. Train 담당자가 합의한 optional config, 동적 loss mapping, 완료 event를 구현한다.
3. Web 담당자가 공개된 schema에 맞춰 config 입력과 진행 event 소비를 구현한다.

Train 담당자는 `src/pipelines/web/`을 수정하지 않는다. Web 담당자는 Train 내부를
import하지 않고 현재처럼 공개 CLI와 progress JSON Lines만 소비한다.
