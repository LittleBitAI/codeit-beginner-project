# 공통 계약

이 문서는 모든 pipeline이 공유하는 최소 interface와 경계만 정의합니다. pipeline별 입력, 처리 방식, 산출물 schema는 각 담당 directory에서 별도로 정합니다.

## 실행 interface

각 pipeline은 외부에 `run(config: dict) -> dict` 하나만 공개합니다. pipeline끼리는 서로의 내부 모듈을 import하거나 직접 호출하지 않으며, 실행 순서는 `src/main_pipeline.py`만 관리합니다.

## 공통 반환값

모든 `run(config)` 결과는 아래 네 key를 **정확히** 포함합니다. 계약에 없는 key를 추가하면 실행이 중단되므로, 새 key가 필요하면 먼저 이 문서를 합의해 갱신합니다.

| key | 타입 | 설명 |
| --- | --- | --- |
| `status` | `str` | 실행 상태. `main_pipeline`은 `"ok"`가 아니면 이후 pipeline을 실행하지 않고 중단합니다. |
| `artifacts` | **JSON 직렬화 가능한 `dict`** | pipeline이 생성한 artifact 정보. 다음 pipeline에는 `config["inputs"]`로 전달됩니다. |
| `summary` | **JSON 직렬화 가능한 `dict`** | 실행 결과 요약. |
| `message` | `str` | 사용자에게 전달할 설명 또는 오류 내용. |

`bool`은 `str`이나 `dict` 자리에 올 수 없습니다.

### `artifacts`와 `summary`의 추가 제약

두 값은 그대로 JSON으로 기록되므로 **일반 `Mapping`이 아니라 `dict`여야 하고, `json.dumps()`로 직렬화할 수 있어야 합니다.**

- `MappingProxyType`처럼 `Mapping`이지만 `json.dumps()`에서 실패하는 값은 거부합니다. 검증을 통과시키면 나중에 저장 시점에 깨집니다.
- `set`, `datetime` 등 JSON으로 못 바꾸는 값이 안에 들어 있으면 거부합니다.
- `NaN`과 `Infinity`도 거부합니다. 표준 JSON이 아니라 다른 도구가 읽을 때 깨집니다.
- `OrderedDict`처럼 `dict`를 상속하고 직렬화 가능한 값은 허용합니다.

값을 `dict`로 만들려면 `dict(...)`로 복사해서 반환하세요.

### 검증 방식

이 계약은 문서로만 두지 않고 `src/common/contract.py`의 `validate_pipeline_result()`가 실행 시점에 확인합니다. `src/main_pipeline.py`는 각 pipeline의 `run()` 결과를 받은 직후 이 함수를 호출합니다.

위반이 있으면 `PipelineContractError`(`ValueError`의 하위 클래스)가 발생하며, 메시지에 pipeline 이름과 함께 다음을 모두 모아 알려줍니다.

- 누락된 필수 key 이름
- 계약에 없는 key 이름
- 타입이 다른 key의 기대 타입과 실제 타입

한 pipeline은 다른 component가 소유한 artifact를 수정하지 않습니다. 구체적인 경로, file name, schema가 확정되기 전에는 이 문서에 임의로 추가하지 않습니다.

## Experiment registry exact-URI 조회 interface

Experiment record 소비자는 registry pipeline의 내부 module을 import하거나
registry pipeline을 다시 실행하지 않습니다. 아래 `src.common` facade에 registry가
반환한 **정확한** `experiment_record_uri`와 같은 storage config를 전달합니다.

```python
from src.common import ExperimentRegistryError, read_experiment_record

record = read_experiment_record(
    experiment_record_uri,
    config,
    expected_run_id="exp-0001",  # 선택
)
```

이 interface는 `create_storage(config).read_json(...)`으로 지정된 record 하나만
읽습니다. Prefix listing, 최신 record 검색, 다른 실험으로의 fallback은 하지
않습니다. Local URI는 `config["registry"]["repo_root"]`(없으면 common module
위치에서 계산한 repository root) 기준 절대 후보로 해석합니다. 이 후보가 실제
`LocalStorage.root` 안에 있을 때만 storage 기준 상대 경로로 바꾸므로, 이름이
같다는 이유만으로 URI segment를 제거하지 않습니다. `s3://` URI는 변경하지 않고
그대로 storage에 전달합니다.

조회 결과는 object(`dict`)여야 하고 `run_id`는 비어 있지 않은 문자열이어야
합니다. `expected_run_id`를 주면 record의 `run_id`와 정확히 같아야 합니다.
Storage, schema, run ID 검증 실패는 모두 public `ExperimentRegistryError`로
보고됩니다. Storage 실패의 public message에는 입력 URI, query, backend 오류
원문을 넣지 않고 안전한 예외 type만 표시하며, 원래 예외는 `__cause__`로
보존합니다.

## Experiment 목록·검색·비교 interface

위 exact-URI 조회와 **별개 경로**입니다. Registry는 실행마다 experiment record와
함께 run별 summary sidecar를 `registry/index/<run_id>.json`에 남기고
(`config["registry"]["index_prefix"]`로 바꿀 수 있습니다), 소비자는 아래 함수로 그
index만 읽습니다.

```python
from src.common import (
    compare_experiment_summaries,
    list_experiment_summaries,
    search_experiment_summaries,
)

summaries = list_experiment_summaries(config)
찾은_것 = search_experiment_summaries(config, run_id_contains="exp-", min_map=0.3)
비교 = compare_experiment_summaries(["exp-0001", "exp-0002"], config)
```

세 함수는 index prefix 아래만 읽습니다. 원하는 index가 없다고 해서 record를 추측해
fallback하지 않으며, `read_experiment_record()`의 동작은 그대로입니다. index 항목
하나를 읽지 못해도 목록 전체를 실패시키지 않고 그 항목만 건너뜁니다. 실패는 모두
`ExperimentRegistryError`로 보고하며 public message에 입력 경로나 backend 오류
원문을 넣지 않습니다.

summary 문서의 형식(`summary_version`, `run_id`, `created_at`, `seed`,
`schema_version`, `experiment_record_uri`, `metrics`, `metrics_source`,
`artifacts`, `verification`, `submission_check`)은
`contracts/proposals/002-experiment-index-and-summary.md`에 있습니다. 값은 없을 수
있으며 그때는 `null`입니다.

**index는 record에서 다시 만들 수 있는 cache이고 record가 진실입니다.** 그래서
registry는 index 저장이 실패해도 실행을 실패시키지 않고 `summary["index_status"]`로
알립니다. 빠진 index는 registry가 제공하는 명령으로 다시 만듭니다.

### Web 경계

Web pipeline이 외부에 공개하는 interface는 계속 `run(config) -> dict` 하나뿐입니다.
`config["web"]["experiment_record_uri"]`가 있으면 위 common facade만 호출하며,
`Path`, `open`, `read_text`, storage 직접 생성, registry pipeline import/호출로
artifact를 읽지 않습니다. 선택 설정 `expected_run_id`로 조회한 실험을 고정할 수
있습니다. URI 설정이 없으면 기존 dummy 결과를 그대로 반환합니다.

실험 목록·검색·비교 화면이 필요하면 위 세 함수를 씁니다. 이때도 `Path`, `open`,
storage 직접 생성, registry pipeline import는 하지 않습니다.
