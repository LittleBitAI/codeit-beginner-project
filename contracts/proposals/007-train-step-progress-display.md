# Train batch 진행 표시 제안

## 요청

Train은 기존 `train.progress/1` JSON Lines에 optional `step_progress` event를
추가한다. Web은 이 event가 있으면 현재 epoch 내부의 train 또는 validation batch
진행률을 표시하고, event가 없는 이전 실행은 기존 epoch 진행률만 표시한다.

```json
{
  "schema": "train.progress/1",
  "event": "step_progress",
  "run_id": "train-20260808T120000000000Z",
  "epoch": 2,
  "epochs": 50,
  "phase": "train",
  "step": 12,
  "total_steps": 100,
  "ts": "2026-08-08T12:00:05.000000Z"
}
```

`phase`는 `train` 또는 `validation`이다. `step`은 해당 phase에서 성공적으로
완료한 batch 수이며 1부터 시작한다. `total_steps`는 해당 DataLoader의 전체 batch
수다. 각 phase의 첫 batch와 마지막 batch는 항상 나오고, 중간 event는 monotonic
clock 기준 직전 출력 후 5초 이상 지났을 때만 나온다. 한 batch뿐인 phase는 한 번만
나온다.

## 호환 조건

- 기존 `run_started`, `epoch_started`, `epoch_completed`, `training_completed`의
  의미와 필드는 바꾸지 않는다.
- event 순서는 `epoch_started`, train step, validation step, `epoch_completed`다.
- `training_history.json`, checkpoint와 `run(config)` 반환값은 바꾸지 않는다.
- Web은 누락되거나 잘못된 `step_progress`를 무시하고 학습 job을 실패시키지 않는다.
- 이번 Train 변경은 Web 파일을 수정하지 않는다. 전용 UI 반영은 Web 소유자의 별도
  branch와 Pull Request에서 수행한다.
