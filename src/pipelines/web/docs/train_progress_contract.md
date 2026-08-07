# train 진행 로그 계약 (`train.progress/1`)

web은 학습을 공개 CLI(`python -m src.main_pipeline --only train`)로만 실행하므로, subprocess 밖에서는 학습 상태를 알 수 없습니다. 이 계약은 그 간극을 메우는 진행 로그의 형식입니다.

**이 계약은 web이 train 담당자에게 제안했고, train 담당자가 합의해 이미 구현했습니다.** train 쪽 구현은 `src/pipelines/train/progress.py`이고, web 쪽 소비자는 `src/pipelines/web/progress.py`입니다. 형식을 바꾸려면 양쪽 담당자가 다시 합의해야 합니다.

## 출력 위치: stdout이 아니라 stderr

`src/main_pipeline.py`는 실행 결과를 `indent=2`로 stdout에 씁니다. 즉 stdout에 담기는 것은 **여러 줄에 걸친 JSON 문서 하나**입니다. 진행 로그를 여기 섞으면 그 문서를 다시 조립할 방법이 없습니다.

그래서 진행 로그는 **stderr로만** 나갑니다. 부수 효과도 유리합니다. pipe로 받을 때 stdout은 블록 버퍼링이라 실시간성이 없지만 `sys.stderr`는 리다이렉트되어도 line-buffered라 줄 단위로 바로 흘러나옵니다. `pip`, `docker build`, `ffmpeg`가 모두 같은 이유로 진행 표시를 stderr에 둡니다.

## 형식: JSON Lines

한 줄에 완결된 JSON object 하나. `ensure_ascii=False`, `indent` 없음, `\n`으로 끝, `flush=True`.

```
{"schema":"train.progress/1","event":"epoch_completed","run_id":"web-20260805T012233123456Z","epoch":1,"epochs":50,"train_loss":0.4312,"validation_loss":0.5109,"best_validation_loss":0.5109,"best_epoch":1,"is_best":true,"epoch_seconds":42.1,"ts":"2026-08-05T01:23:15.300000Z"}
```

모든 event 공통 필드: `schema`(고정 문자열), `event`, `run_id`, `ts`(UTC ISO-8601).

| event | 추가 필드 |
| --- | --- |
| `run_started` | `architecture`, `device`, `epochs`, `train_images`, `validation_images`, `class_count` |
| `epoch_started` | `epoch`, `epochs` |
| `epoch_completed` | `epoch`, `epochs`, `train_loss`, `validation_loss`, `train_loss_components`, `validation_loss_components`, `best_validation_loss`, `best_epoch`, `is_best`, `epoch_seconds` |
| `training_completed` | `planned_epochs`, `completed_epochs`, `stopped_early`, `best_epoch`, `best_validation_loss` |

**새로 만든 어휘가 없습니다.** `epoch`·`train_loss`·`validation_loss`는 `trainer.py`의 `epoch_record` 키 그대로이고, 나머지는 train이 이미 반환하는 `summary`·`artifacts` 키 그대로입니다.

두 `*_components`는 `문자열 -> 유한한 수` mapping입니다. 이름은 모델이 돌려준 것을 그대로 쓰므로(Faster R-CNN과 RetinaNet이 다릅니다) **web은 이름을 열거하지 않고 받은 것을 그립니다.**

`training_completed`는 artifact 저장까지 끝난 뒤 한 번 나옵니다. 조기 종료로 `completed_epochs < planned_epochs`여도 web은 진행률을 100%, 남은 시간을 0으로 만듭니다. 이 event가 없는 실행(취소, 이 계약 이전의 옛 실행)은 예전 그대로 읽힙니다.

`run_started`가 따로 필요한 이유는 `train_images`·`validation_images`·`class_count`가 manifest에서 파생되는 값이라 학습이 끝나기 전에는 web이 알 방법이 없기 때문입니다. `epoch_started`가 없으면 첫 epoch이 20분 걸릴 때 화면이 20분 동안 비어 있습니다.

## 지켜야 할 비파괴 보장 6가지

1. **`run(config)` 반환값이 완전히 동일하다.** 4개 key와 값 모두 그대로입니다.
2. **stdout에는 아무것도 추가하지 않는다.**
3. **config key는 합의한 것만 만든다.** 이 계약을 처음 만들 때는 새 key를 아예 두지 않았습니다. 이후 `contracts/proposals/004-train-observability-and-early-stop.md`로 선택 key `early_stopping` 하나를 합의했고, web은 그 검증 규칙을 `train_config.py`에 복제합니다.
4. **`training_history.json` 형식을 바꾸지 않는다.** evaluate와 registry가 읽는 파일입니다.
5. **emitter는 예외 안전해야 한다.** `try/except Exception: pass`로 감쌉니다. web이 학습을 취소하면 pipe가 닫히고 다음 write가 `BrokenPipeError`(Windows에서는 맨 `OSError`)를 냅니다. 이 예외가 학습을 중단시키거나 exit code를 바꾸면 안 됩니다. **여섯 중 가장 중요한 조항입니다.**
6. **main process만 출력한다.** `num_workers > 0`이면 DataLoader worker가 stderr를 상속하므로 중복 출력이 나오면 안 됩니다.

## web이 이 스트림을 읽는 방식

`src/pipelines/web/progress.py`의 `consume_line(state, line)`은 I/O 없는 순수 함수이고 **어떤 입력에도 예외를 던지지 않습니다.**

| 입력 상황 | 처리 |
| --- | --- |
| 빈 줄 | 건너뜀 |
| `{`로 시작하지 않음 (torch 경고 등) | 원문 로그로 표시 |
| JSON 파싱 실패 | 원문 로그로 표시. 버리지도 않고 예외도 아님 |
| `schema` 없음 / 모르는 major 버전 | 원문 로그로 표시 |
| 모르는 `event` | 원문 로그로 표시 — event를 추가해도 schema 버전을 올릴 필요가 없습니다 |
| 필드 타입 오류 | 그 필드만 버리고 나머지 event는 살림 |
| loss가 `NaN`/`inf` | `None`으로 바꿈. 브라우저 `JSON.parse`가 맨 `NaN`에서 실패합니다 |
| `*_components`가 mapping이 아님 | 그 필드만 `None`. epoch 자체는 살림 |
| component 값 하나가 수가 아니거나 `NaN` | 그 이름만 버리고 나머지 이름은 살림. 남는 게 없으면 `None` |
| `epoch` 중복·역순 | last-write-wins로 담고 정렬해 표시. 단조 증가를 가정하지 않음 |
| 진행 줄이 한 번도 없음 | `available: false` — GUI가 "진행률 정보 없음"을 표시. 가짜 값 없음 |

남은 시간은 `epoch_completed`가 2건 이상 쌓였을 때 실제 `epoch_seconds` 평균으로만 계산합니다. 그 전에는 추정하지 않습니다.

`src/pipelines/web/tests/test_web_progress.py`가 위 표의 모든 열화 동작을 검증합니다.
