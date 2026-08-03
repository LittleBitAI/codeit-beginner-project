# 저장소 공통 AI 작업 규칙

## 작업 시작 전

- 저장소 구조, `git status`, 관련 파일을 먼저 확인합니다.
- repository root 지침과 작업 대상에 가장 가까운 `CLAUDE.md` 또는 `AGENTS.md`를 읽고 함께 따릅니다.
- 사용자가 지정한 범위만 수정합니다. 범위 밖의 정리, 기능 추가, unrelated refactoring은 하지 않습니다.

## 소유권과 경계

- directory 소유권 경계를 넘지 않습니다. 다른 component의 변경이 필요하면 직접 수정하지 말고 사용자에게 보고합니다.
- 다른 pipeline의 내부 모듈을 직접 import하거나 호출하지 않습니다.
- pipeline의 연결과 실행 순서 조정은 지정된 integration entry point에서만 수행합니다.
- 다른 component가 생성하거나 소유한 artifact를 수정, 이동, 덮어쓰기, 삭제하지 않습니다.
- shared contract, config, dependency, common code, integration file은 명시적으로 배정받지 않았다면 변경하지 않습니다.
- pipeline별 책임과 구현 규칙은 해당 directory의 가장 가까운 지침을 따르며 repository root 지침에 추가하지 않습니다.

## 보안과 저장소 위생

- absolute path를 hardcode하지 않습니다.
- credential, token, secret, `.env` 내용이나 민감한 값을 code, 문서, log, 예시에 노출하지 않습니다.
- raw/processed dataset, checkpoint·weight, TensorBoard event, training log·run, cache, local environment, 대량 generated file을 commit 대상으로 추가하지 않습니다.
- 사용자가 명시적으로 요청하지 않으면 commit을 만들지 않습니다. 모든 변경은 Pull Request를 통해 반영하며 `main`에 직접 commit하지 않습니다.
- Pull Request용 작업 branch는 `pipeline/<area>/<task-summary>` 형식을 사용하며 merge 후 삭제하는 임시 branch입니다. 한 branch와 Pull Request에는 한 가지 focused change만 담습니다.
- `<area>`는 `data`, `train`, `evaluate`, `registry`, `web` 중 배정받은 pipeline 이름을 사용합니다. repository-wide 변경처럼 해당 값이 정해지지 않은 작업은 임의로 만들지 말고 사용자에게 확인합니다.
- `main` 또는 GitHub remote가 없어 Pull Request를 만들 수 없다면 commit이나 local merge로 대신하지 말고 사용자에게 확인합니다.
- commit을 요청받은 경우 message는 한국어로 작성하며 필요한 표준 기술 용어만 English로 유지합니다.
- commit 후 push와 Pull Request 생성을 명시적으로 요청받으면 관련 check와 clean working tree를 확인한 뒤 `git pr`을 사용합니다. 요청 없이 자동으로 push하거나 Pull Request를 만들지 않습니다.
- `git pr`은 규칙에 맞는 작업 branch에서만 실행합니다. 먼저 `git pr --dry-run`으로 대상 branch와 계획을 확인하고, 생성된 draft Pull Request의 template과 검증 내용을 점검합니다.
- text file은 UTF-8 without BOM과 LF line ending을 사용합니다.

## 검증과 완료 보고

- 변경 범위에 맞는 관련 check와 test를 실행하고 결과를 확인합니다.
- 완료 시 변경 파일, 실행한 test·check와 결과, unresolved issue·TODO, `git status`를 보고합니다.
- 실행하지 못한 test가 있으면 생략 이유와 남은 risk를 명시합니다.

## 중단하고 확인할 상황

다음 상황에서는 추측하거나 범위를 넓히지 말고 작업을 중단한 뒤 사용자에게 확인합니다.

- shared contract 또는 공개 interface 변경
- directory 소유권을 넘는 수정이나 integration 경계 변경
- 삭제, overwrite, history 변경 등 destructive action
- credential 또는 secret이 필요하거나 노출될 가능성이 있는 작업
- 유료 infrastructure 생성·변경 등 비용이 발생할 수 있는 작업
- train, validation, test 또는 competition data leakage 위험
- competition 규칙이 불명확해 구현이나 검증 결과가 달라지는 작업
