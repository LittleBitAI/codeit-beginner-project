# evaluate 진행 로그 계약 (`evaluate.progress/1`)

web은 평가를 공개 CLI(`python -m src.main_pipeline --only evaluate`)로만 실행하므로, subprocess 밖에서는 지금 무엇을 하고 있는지 알 수 없습니다. 이 계약은 그 간극을 메우는 진행 로그의 형식입니다. `train.progress/1`, `data.progress/1`과 같은 모양이며 같은 원칙을 따릅니다.

이 저장소의 평가 기록 3건은 **1013초·1271초·1396초(17~23분)** 걸렸고 모두 성공했습니다. 대회 test 이미지가 **842장 / 1.52 GB**라 매번 다시 받아 추론하기 때문입니다. 그동안 화면에는 `평가 및 submission 생성 중…` 한 줄만 떠 있어서 멈춘 줄 알고 취소하는 일이 있었고, 취소하면 1.52 GB를 처음부터 다시 받으므로 더 느려집니다.

**web 쪽 소비자는 `src/pipelines/web/evaluate_progress.py`이고, 내보내는 쪽은 evaluate 담당이 구현합니다.** 형식을 바꾸려면 양쪽 담당자가 다시 합의해야 합니다.

## 출력 위치: stderr 전용

이 pipeline에서는 특히 중요합니다. `COCOeval`이 stdout에 쓰기 때문에 evaluate는 그 호출을 `redirect_stdout`으로 감싸 두었고, **web은 그 subprocess의 stdout을 결과 JSON 문서로 파싱합니다.** 그러므로 진행 로그는 **stderr로만** 나가고 stdout에는 한 글자도 더하지 않습니다. `sys.stderr`는 리다이렉트되어도 line-buffered라 줄 단위로 바로 흘러나옵니다.

## 형식: JSON Lines

한 줄에 완결된 JSON object 하나. `ensure_ascii=False`, `indent` 없음, `\n`으로 끝, `flush=True`.

```
{"schema":"evaluate.progress/1","event":"predict_progress","stage":"test","done":421,"total":842,"ts":"2026-08-07T03:35:20.123456Z"}
```

모든 event 공통 필드: `schema`(고정 문자열 `evaluate.progress/1`), `event`, `ts`(UTC ISO-8601).

| event | 추가 필드 |
| --- | --- |
| `evaluate_started` | `run_id`, `device`, `validation_images`, `test_images`(없으면 `0`) |
| `predict_progress` | `stage`(`"validation"` 또는 `"test"`), `done`, `total` |
| `metrics_computed` | `mAP`, `mAP50`, `mAP75` |
| `submission_written` | `rows` |
| `evaluate_completed` | `validation_images`, `test_images` |

**새로 만든 어휘가 없습니다.** 이름은 evaluate가 이미 쓰는 것 그대로입니다.

`evaluate_started`의 `test_images`가 있어야 화면이 "842장 중 몇 장"을 그릴 수 있습니다. `predict_progress`의 `stage`로 validation 추론과 test 추론을 구분합니다. 가장 오래 걸리는 구간이 `stage="test"`입니다.

**계산되지 않은 지표는 `null`입니다.** 이 pipeline의 기존 규칙(`0.0`으로 만들지 않는다)을 진행 로그에서도 지킵니다.

## 출력량 제한

`predict_progress`는 이미지마다 내보내지 않습니다. 전체의 2% 이상 진행했거나 마지막 event로부터 1초 이상 지났을 때만 내보내고, `done == total`은 반드시 내보냅니다. 시간 기준은 `time.monotonic()`을 씁니다. 842장이라 이 제한이 없으면 로그가 진행 로그로 뒤덮입니다.

## 지켜야 할 비파괴 보장 6가지

1. **`run(config)` 반환값이 완전히 동일하다.** `artifacts`의 key 구성도 그대로입니다.
2. **stdout에는 아무것도 추가하지 않는다.** 위의 `redirect_stdout` 사정 때문에 특히 그렇습니다.
3. **새 config key를 만들지 않는다.** on/off 플래그를 두지 않습니다.
4. **산출물(metrics, predictions, submission.csv) 내용을 바꾸지 않는다.** 박스와 점수는 계속 반올림하지 않고, 측정되지 않은 지표는 계속 `null`입니다.
5. **emitter는 예외 안전해야 한다.** `try/except Exception: pass`로 감쌉니다. web이 평가를 중단하면 pipe가 닫히고 다음 write가 `BrokenPipeError`(Windows에서는 맨 `OSError`)를 냅니다. 이 예외가 평가를 중단시키거나 exit code를 바꾸면 안 됩니다. **여섯 중 가장 중요한 조항입니다.**
6. **main process만 출력한다.** emitter를 만든 process의 pid에서만 씁니다.

## web이 이 스트림을 읽는 방식

`evaluate_progress.py`의 `consume_line(state, line)`은 I/O 없는 순수 함수이고 **어떤 입력에도 예외를 던지지 않습니다.**

| 입력 상황 | 처리 |
| --- | --- |
| 빈 줄 | 건너뜀 |
| `{`로 시작하지 않음 (COCOeval 로그 등) | 원문 로그로 표시 |
| JSON 파싱 실패 | 원문 로그로 표시. 버리지도 않고 예외도 아님 |
| `schema` 없음 / 모르는 major 버전 | 원문 로그로 표시 |
| 모르는 `event` | 원문 로그로 표시 — event를 추가해도 schema 버전을 올릴 필요가 없습니다 |
| 필드 타입 오류 | 그 필드만 버리고 나머지 event는 살림 |
| 지표가 `NaN`/`inf` | `None`으로 바꿈. 브라우저 `JSON.parse`가 맨 `NaN`에서 실패합니다 |
| `done`이 `total`보다 큼 / 역순 | last-write-wins. 단조 증가를 가정하지 않음. 막대만 100%에서 멈춤 |
| 진행 줄이 한 번도 없음 | `available: false` — 화면이 고정 안내 문구를 표시. 가짜 값 없음 |

남은 시간은 같은 추론 단계에서 두 번 이상 관측했고 장 수와 시간이 모두 늘었을 때, 그 실제 속도로만 계산합니다. 추론 단계가 바뀌면 기준점을 버립니다. validation의 속도로 test의 남은 시간을 재면 거짓말이 되기 때문입니다.

`src/pipelines/web/tests/test_web_evaluate_progress.py`가 위 표의 모든 열화 동작을 검증합니다.

## web이 subprocess를 띄우는 방식

`evaluation.run_evaluation()`은 `runner.spawn`으로 띄우고 pipe마다 thread 하나로 읽습니다. `subprocess.run(capture_output=True)`를 쓰면 자식 출력이 끝날 때까지 pipe에 갇혀 20분 내내 아무것도 볼 수 없고, 한쪽 pipe만 읽으면 반대쪽 OS 버퍼가 차는 순간 교착합니다. stdout은 예전 그대로 모아서 결과 JSON 문서를 파싱하고, stderr는 줄 단위로 `consume_line`에 넣습니다. 진행 로그 처리가 실패해도 평가는 계속됩니다. `EVALUATE_TIMEOUT_SECONDS`(1시간)와 실패·타임아웃 시 반환 형태는 그대로입니다.
