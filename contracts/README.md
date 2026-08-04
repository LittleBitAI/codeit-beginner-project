# 공통 계약

이 문서는 모든 pipeline이 공유하는 최소 interface와 경계만 정의합니다. pipeline별 입력, 처리 방식, 산출물 schema는 각 담당 directory에서 별도로 정합니다.

## 실행 interface

각 pipeline은 외부에 `run(config: dict) -> dict` 하나만 공개합니다. pipeline끼리는 서로의 내부 모듈을 import하거나 직접 호출하지 않으며, 실행 순서는 `src/main_pipeline.py`만 관리합니다.

## 공통 반환값

모든 `run(config)` 결과는 아래 네 key를 **정확히** 포함합니다. 계약에 없는 key를 추가하면 실행이 중단되므로, 새 key가 필요하면 먼저 이 문서를 합의해 갱신합니다.

| key | 타입 | 설명 |
| --- | --- | --- |
| `status` | `str` | 실행 상태. `main_pipeline`은 `"ok"`가 아니면 이후 pipeline을 실행하지 않고 중단합니다. |
| `artifacts` | `dict` | pipeline이 생성한 artifact 정보. 다음 pipeline에는 `config["inputs"]`로 전달됩니다. |
| `summary` | `dict` | 실행 결과 요약. |
| `message` | `str` | 사용자에게 전달할 설명 또는 오류 내용. |

`bool`은 `str`이나 `dict` 자리에 올 수 없습니다.

### 검증 방식

이 계약은 문서로만 두지 않고 `src/common/contract.py`의 `validate_pipeline_result()`가 실행 시점에 확인합니다. `src/main_pipeline.py`는 각 pipeline의 `run()` 결과를 받은 직후 이 함수를 호출합니다.

위반이 있으면 `PipelineContractError`(`ValueError`의 하위 클래스)가 발생하며, 메시지에 pipeline 이름과 함께 다음을 모두 모아 알려줍니다.

- 누락된 필수 key 이름
- 계약에 없는 key 이름
- 타입이 다른 key의 기대 타입과 실제 타입

한 pipeline은 다른 component가 소유한 artifact를 수정하지 않습니다. 구체적인 경로, file name, schema가 확정되기 전에는 이 문서에 임의로 추가하지 않습니다.
