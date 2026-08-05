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
| `epoch_completed` | `epoch`, `epochs`, `train_loss`, `validation_loss`, `best_validation_loss`, `best_epoch`, `is_best`, `epoch_seconds` |

**새로 만든 어휘가 없습니다.** `epoch`·`train_loss`·`validation_loss`는 `trainer.py`의 `epoch_record` 키 그대로이고, 나머지는 train이 이미 반환하는 `summary`·`artifacts` 키 그대로입니다.

`run_started`가 따로 필요한 이유는 `train_images`·`validation_images`·`class_count`가 manifest에서 파생되는 값이라 학습이 끝나기 전에는 web이 알 방법이 없기 때문입니다. `epoch_started`가 없으면 첫 epoch이 20분 걸릴 때 화면이 20분 동안 비어 있습니다.

## 지켜야 할 비파괴 보장 6가지

1. **`run(config)` 반환값이 완전히 동일하다.** 4개 key와 값 모두 그대로입니다.
2. **stdout에는 아무것도 추가하지 않는다.**
3. **새 config key를 만들지 않는다.** on/off 플래그를 두면 web이 미러링할 검증 규칙이 하나 늘어 drift 표면만 커집니다.
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
| `epoch` 중복·역순 | last-write-wins로 담고 정렬해 표시. 단조 증가를 가정하지 않음 |
| 진행 줄이 한 번도 없음 | `available: false` — GUI가 "진행률 정보 없음"을 표시. 가짜 값 없음 |

남은 시간은 `epoch_completed`가 2건 이상 쌓였을 때 실제 `epoch_seconds` 평균으로만 계산합니다. 그 전에는 추정하지 않습니다.

`src/pipelines/web/tests/test_web_progress.py`가 위 표의 모든 열화 동작을 검증합니다.
