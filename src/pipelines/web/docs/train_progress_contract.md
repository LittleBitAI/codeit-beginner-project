# train 진행 로그 계약 제안 (`train.progress/1`)

이 문서는 **web pipeline이 train pipeline 담당자에게 요청하는 제안**입니다. web은 `src/pipelines/train/`을 수정하지 않으며, 이 계약은 train 담당자가 검토하고 합의한 뒤에만 반영됩니다.

## 왜 필요한가

Training GUI(`src/pipelines/web/`)는 학습을 오직 아래 공개 CLI로만 실행합니다.

```
python -m src.main_pipeline --config <config> --only train
```

현재 train pipeline은 학습 중 아무 출력도 내지 않습니다. 따라서 subprocess를 띄운 web은 학습이 완전히 끝날 때까지 **프로세스가 살아 있다는 사실 외에는 아무것도 알 수 없습니다**. epoch이 몇 번째인지, loss가 줄고 있는지, 몇 분이 남았는지 표시할 방법이 없습니다.

web은 없는 정보를 지어내지 않습니다. 진행 로그가 없는 동안 GUI는 "진행률 정보 없음"을 그대로 표시하며, 가짜 퍼센트나 시간 기반 외삽을 만들지 않습니다. 이 계약이 반영되면 진행 바·남은 시간·loss 곡선이 **web 코드 변경 없이** 켜집니다.

## 어디로 출력하는가: stdout이 아니라 **stderr**

`src/main_pipeline.py:229`는 실행 결과를 stdout에 씁니다.

```python
print(json.dumps(result, ensure_ascii=False, indent=2))
```

`indent=2`라서 stdout에 담기는 것은 **여러 줄에 걸친 JSON 문서 하나**입니다. 진행 로그를 stdout에 섞으면 이 문서를 다시 조립할 방법이 없습니다("마지막 한 줄만 파싱"이 통하지 않습니다). 그래서 진행 로그는 **stderr로만** 보냅니다.

부수 효과 두 가지가 모두 유리합니다.

- `tests/test_main_pipeline.py:304`는 `json.loads(capsys.readouterr().out)`으로 stdout **전체**를 파싱합니다. stderr로 보내면 이 테스트가 손대지 않아도 그대로 통과합니다.
- pipe로 받을 때 stdout은 블록 버퍼링(약 8KB)이라 실시간성이 없지만, CPython의 `sys.stderr`는 리다이렉트되어도 line-buffered라 줄 단위로 바로 흘러나옵니다.

`pip`, `docker build`, `ffmpeg`, `curl`이 모두 같은 이유로 진행 표시를 stderr에 둡니다.

## 형식: JSON Lines

한 줄에 완결된 JSON object 하나. `ensure_ascii=False`, `indent` 없음, `\n`으로 끝, `flush=True`.

```
{"schema":"train.progress/1","event":"run_started","run_id":"web-20260805T012233123456Z","architecture":"fasterrcnn_resnet50_fpn","device":"cuda","epochs":50,"train_images":3200,"validation_images":800,"class_count":12,"ts":"2026-08-05T01:22:33.123456Z"}
{"schema":"train.progress/1","event":"epoch_started","run_id":"web-20260805T012233123456Z","epoch":1,"epochs":50,"ts":"2026-08-05T01:22:33.200000Z"}
{"schema":"train.progress/1","event":"epoch_completed","run_id":"web-20260805T012233123456Z","epoch":1,"epochs":50,"train_loss":0.4312,"validation_loss":0.5109,"best_validation_loss":0.5109,"best_epoch":1,"is_best":true,"epoch_seconds":42.1,"ts":"2026-08-05T01:23:15.300000Z"}
```

### 필드

모든 event에 공통: `schema`(고정 문자열 `"train.progress/1"`), `event`, `run_id`, `ts`(UTC ISO-8601).

| event | 추가 필드 |
| --- | --- |
| `run_started` | `architecture`, `device`, `epochs`, `train_images`, `validation_images`, `class_count` |
| `epoch_started` | `epoch`, `epochs` |
| `epoch_completed` | `epoch`, `epochs`, `train_loss`, `validation_loss`, `best_validation_loss`, `best_epoch`, `is_best`, `epoch_seconds` |

**새로 만든 어휘가 하나도 없습니다.** `epoch` / `train_loss` / `validation_loss`는 `trainer.py:125-129`의 `epoch_record` 키 그대로이고, 나머지는 train이 이미 반환하는 `summary` / `artifacts` 키 그대로입니다. 그래서 이 변경은 값을 새로 계산하는 일이 아니라 **이미 손에 있는 값을 한 줄 찍는 일**입니다.

`run_started`가 필요한 이유: `train_images` / `validation_images` / `class_count`는 manifest에서 파생되는 값이라 **학습이 끝나기 전에는 web이 알 방법이 전혀 없습니다**. 그런데 GUI는 학습이 도는 **동안** 이 숫자를 보여줘야 합니다.

`epoch_started`가 필요한 이유: 첫 epoch이 20분 걸리면, 이게 없을 때 화면은 20분 동안 완전히 비어 있습니다.

## 지켜야 할 비파괴 보장 6가지

1. **`run(config)` 반환값이 완전히 동일하다.** 4개 key와 값 모두 그대로이고, 동작 변화가 없습니다.
2. **stdout에는 아무것도 추가하지 않는다.** 위에서 설명한 이유입니다.
3. **새 config key를 만들지 않는다.** 조건부 on/off 플래그를 두면 web이 미러링해야 할 검증 규칙이 하나 늘어 drift 표면만 커집니다. 무조건 출력이 더 단순하고 안전합니다.
4. **`training_history.json` 형식을 바꾸지 않는다.** downstream(evaluate·registry)이 읽는 파일입니다.
5. **emitter는 예외 안전해야 한다.** 반드시 `try/except Exception: pass`로 감쌉니다. web이 학습을 취소하면 pipe가 닫히고, 그 다음 write는 `BrokenPipeError`(Windows에서는 맨 `OSError`)를 냅니다. 이 예외가 학습을 중단시키거나 exit code를 바꾸면 안 됩니다. **여섯 가지 중 가장 중요한 조항입니다.**
6. **main process만 출력한다.** `num_workers > 0`이면 DataLoader worker가 stderr를 상속하므로, worker에서 중복 출력이 나오지 않아야 합니다.

## 제안하는 call site 3곳

train 담당자가 판단할 부분이지만, 참고용으로 값이 이미 scope에 있는 지점을 적어 둡니다.

| event | 위치 | 근거 |
| --- | --- | --- |
| `run_started` | `src/pipelines/train/pipeline.py::_execute`, 246줄 `set_seed(...)` 부근 | `train_dataset`, `validation_dataset`, `class_map`, `settings`가 모두 살아 있습니다 |
| `epoch_started` | `src/pipelines/train/trainer.py::_train_model`, 106줄 `for epoch in range(...)` 진입 직후 | |
| `epoch_completed` | 같은 파일 130줄 `history.append(epoch_record)` 직후 | `epoch_record`에 이미 `epoch`·`train_loss`·`validation_loss`가 들어 있습니다 |

`epoch_seconds`는 epoch 시작 시각을 지역 변수로 잡아 두면 계산됩니다. `best_validation_loss` / `best_epoch` / `is_best`는 131줄의 기존 `if validation_loss < best_loss:` 분기에서 이미 판정하고 있는 값입니다.

## web이 이 스트림을 어떻게 읽는가

`src/pipelines/web/progress.py`의 `consume_line(state, line)` — I/O 없는 순수 함수이고, **어떤 입력에도 예외를 던지지 않습니다.**

| 입력 상황 | 처리 |
| --- | --- |
| 빈 줄 | 건너뜀 |
| `{`로 시작하지 않음 (torch 경고 등) | 원문 로그로 표시, 상태 변화 없음 |
| JSON 파싱 실패 | 원문 로그로 표시. 버리지도 않고 예외도 아님 |
| `schema` 없음 / 모르는 major 버전 | 원문 로그로 표시 |
| 모르는 `event` | 원문 로그로 표시 — 나중에 `batch_progress` 같은 event를 추가해도 schema 버전을 올릴 필요가 없습니다 |
| 필드 타입 오류 | 그 필드만 버리고 나머지 event는 살림 |
| loss가 `NaN`/`inf` | `None`으로 바꿈. `json.dumps`가 맨 `NaN`을 뱉으면 브라우저 `JSON.parse`가 실패합니다 |
| `epoch` 중복·역순 | `dict[int, …]`에 last-write-wins로 담고 정렬해 표시. 단조 증가를 가정하지 않음 |
| 진행 줄이 한 번도 없음 | `{"available": false}` — GUI가 "진행률 정보 없음"을 표시. 가짜 값 없음 |

남은 시간은 `epoch_completed`가 2건 이상 쌓였을 때 실제 `epoch_seconds` 평균으로만 계산합니다. 그 전에는 "남은 시간을 추정할 수 없습니다"로 표시합니다.

## 합의 후 검증

train 변경이 반영되면 web 쪽에서 end-to-end 테스트를 하나 추가합니다. 이미지 2장짜리 CPU fixture로 1 epoch을 `--only train`으로 돌려서

1. stderr에 `epoch_completed` 줄이 1건 이상 나오고,
2. **stdout이 여전히 단일 JSON 문서로 파싱되는지**

를 확인합니다. train의 내부 모듈을 import하지 않고 공개 CLI로만 접근하므로 경계를 넘지 않습니다.
