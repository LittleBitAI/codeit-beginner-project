# data 진행 로그 계약 (`data.progress/1`)

web은 데이터 준비를 공개 CLI(`python -m src.main_pipeline --only data`)로만 실행하므로, subprocess 밖에서는 준비 상태를 알 수 없습니다. 이 계약은 그 간극을 메우는 진행 로그의 형식입니다. `train.progress/1`(`train_progress_contract.md`)과 같은 모양이며 같은 원칙을 따릅니다.

이 저장소의 원본은 S3에 1,842개 객체 / 1.93 GB이고 준비에 474초(7분 54초)가 걸립니다. 그동안 화면에 고정된 문장 한 줄만 떠서 멈춘 것처럼 보였습니다.

**web 쪽 소비자는 `src/pipelines/web/data_progress.py`이고, 내보내는 쪽은 data 담당이 구현합니다.** 형식을 바꾸려면 양쪽 담당자가 다시 합의해야 합니다.

## 출력 위치: stdout이 아니라 stderr

`src/main_pipeline.py`는 실행 결과를 `indent=2`로 stdout에 씁니다. 즉 stdout에 담기는 것은 **여러 줄에 걸친 JSON 문서 하나**입니다. 진행 로그를 여기 섞으면 그 문서를 다시 조립할 방법이 없습니다.

그래서 진행 로그는 **stderr로만** 나갑니다. pipe로 받을 때 stdout은 블록 버퍼링이라 실시간성이 없지만 `sys.stderr`는 리다이렉트되어도 line-buffered라 줄 단위로 바로 흘러나옵니다. `pip`, `docker build`, `ffmpeg`가 모두 같은 이유로 진행 표시를 stderr에 둡니다.

## 형식: JSON Lines

한 줄에 완결된 JSON object 하나. `ensure_ascii=False`, `indent` 없음, `\n`으로 끝, `flush=True`.

```
{"schema":"data.progress/1","event":"read_progress","stage":"annotations","done":400,"total":1842,"ts":"2026-08-07T03:35:20.123456Z"}
```

모든 event 공통 필드: `schema`(고정 문자열 `data.progress/1`), `event`, `ts`(UTC ISO-8601).

| event | 추가 필드 |
| --- | --- |
| `prepare_started` | `raw_prefix`, `split_ratio`, `seed`, `split_method` |
| `sources_listed` | `train_images`, `annotations`, `test_images` (원본에서 찾은 개수) |
| `read_progress` | `stage`(`"annotations"` 또는 `"test_images"`), `done`, `total` |
| `step_started` | `step`(`"split"`, `"manifests"`, `"publish"` 중 하나) |
| `prepare_completed` | `train_images`, `validation_images`, `category_count` |

**새로 만든 어휘가 없습니다.** 이름은 이미 `dataset_summary.json`과 `run()` 요약이 쓰는 키 그대로입니다.

`sources_listed`가 따로 필요한 이유는, 전체 개수를 알기 전에는 화면이 진행률을 그릴 수 없기 때문입니다. `read_progress`가 없으면 가장 오래 걸리는 구간이 통째로 깜깜해집니다.

## 출력량 제한

`read_progress`는 **읽을 때마다 내보내지 않습니다.** 전체의 2% 이상 진행했거나 마지막 event로부터 1초 이상 지났을 때만 내보냅니다. 마지막 항목(`done == total`)은 반드시 내보냅니다. 이 제한이 없으면 1,842줄이 pipe로 쏟아져 로그가 진행 로그로 뒤덮입니다.

## 지켜야 할 비파괴 보장 6가지

1. **`run(config)` 반환값이 완전히 동일하다.** key와 값 모두 그대로입니다.
2. **stdout에는 아무것도 추가하지 않는다.**
3. **새 config key를 만들지 않는다.** on/off 플래그를 두지 않습니다.
4. **산출물(`train_manifest.json`, `dataset_summary.json` 등) 내용을 바꾸지 않는다.** `schema_version`도 올리지 않습니다. 이 변경은 관찰용 출력만 더합니다.
5. **emitter는 예외 안전해야 한다.** `try/except Exception: pass`로 감쌉니다. web이 준비를 중단하면 pipe가 닫히고 다음 write가 `BrokenPipeError`(Windows에서는 맨 `OSError`)를 냅니다. 이 예외가 준비를 중단시키거나 exit code를 바꾸면 안 됩니다. **여섯 중 가장 중요한 조항입니다.**
6. **main process만 출력한다.** 읽기는 `ThreadPoolExecutor`로 도는데 thread는 같은 process이므로 문제없지만, 앞으로 process가 늘어도 안전하도록 emitter를 만든 process의 pid에서만 씁니다.

## web이 이 스트림을 읽는 방식

`data_progress.py`의 `consume_line(state, line)`은 I/O 없는 순수 함수이고 **어떤 입력에도 예외를 던지지 않습니다.**

| 입력 상황 | 처리 |
| --- | --- |
| 빈 줄 | 건너뜀 |
| `{`로 시작하지 않음 (boto3 경고 등) | 원문 로그로 표시 |
| JSON 파싱 실패 | 원문 로그로 표시. 버리지도 않고 예외도 아님 |
| `schema` 없음 / 모르는 major 버전 | 원문 로그로 표시 |
| 모르는 `event` | 원문 로그로 표시 — event를 추가해도 schema 버전을 올릴 필요가 없습니다 |
| 필드 타입 오류 | 그 필드만 버리고 나머지 event는 살림 |
| `done`이 `total`보다 큼 / 역순 | last-write-wins. 단조 증가를 가정하지 않음. 막대만 100%에서 멈춤 |
| 진행 줄이 한 번도 없음 | `available: false` — 화면이 고정 안내 문구를 표시. 가짜 값 없음 |

남은 시간은 같은 읽기 단계에서 두 번 이상 관측했고 개수와 시간이 모두 늘었을 때, 그 실제 속도로만 계산합니다. 읽기 단계가 바뀌면 기준점을 버립니다. 관측이 부족하면 추정하지 않습니다.

`src/pipelines/web/tests/test_web_data_progress.py`가 위 표의 모든 열화 동작을 검증합니다.

## web이 subprocess를 띄우는 방식

`datasets.prepare_dataset()`은 `runner.spawn`으로 띄우고 pipe마다 thread 하나로 읽습니다. `subprocess.run(capture_output=True)`를 쓰면 자식 출력이 끝날 때까지 pipe에 갇혀 8분 내내 아무것도 볼 수 없고, 한쪽 pipe만 읽으면 반대쪽 OS 버퍼가 차는 순간 교착합니다. stdout은 모아서 마지막 JSON 문서를 파싱하고, stderr는 줄 단위로 `consume_line`에 넣습니다. 진행 로그 처리가 실패해도 준비는 계속됩니다.
