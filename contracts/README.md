# 공통 계약

이 문서는 모든 pipeline이 공유하는 최소 interface와 경계만 정의합니다. pipeline별 입력, 처리 방식, 산출물 schema는 각 담당 directory에서 별도로 정합니다.

## 실행 interface

각 pipeline은 외부에 `run(config: dict) -> dict` 하나만 공개합니다. pipeline끼리는 서로의 내부 모듈을 import하거나 직접 호출하지 않으며, 실행 순서는 `src/main_pipeline.py`만 관리합니다.

## 공통 반환값

모든 `run(config)` 결과는 다음 key를 포함합니다.

- `status`: 실행 상태
- `artifacts`: pipeline이 생성한 artifact 정보
- `summary`: 실행 결과 요약
- `message`: 사용자에게 전달할 설명 또는 오류 내용

한 pipeline은 다른 component가 소유한 artifact를 수정하지 않습니다. 구체적인 경로, file name, schema가 확정되기 전에는 이 문서에 임의로 추가하지 않습니다.
