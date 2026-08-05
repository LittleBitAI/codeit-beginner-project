# 공용 파일

누구 한 사람의 것이 아니라서, 여러 명이 동시에 작업하면 부딪히는 파일 목록입니다.

## 규칙

1. 아래 파일을 바꾸는 변경은 **단독 Pull Request**로 만듭니다. 파이프라인 작업과 섞지 않습니다. 섞으면 review가 어려워지고, 되돌릴 때 멀쩡한 작업까지 같이 되돌아갑니다.
2. 내가 소유하지 않은 파일이 바뀌어야 하면 직접 고치지 말고 `contracts/proposals/NNN-<주제>.md`에 제안을 씁니다. 소유자가 읽고 직접 반영합니다.
3. 바꾸기 전에 `git branch -r`로 같은 파일을 건드리는 열린 branch가 있는지 먼저 봅니다.

## 목록

| 파일 | 왜 부딪히는가 | 바꿀 때 |
| --- | --- | --- |
| `src/main_pipeline.py` | `_STAGES`와 `_REQUIRED_ARTIFACTS`가 **모든 파이프라인의 artifact key를 복사**해 두고 있습니다. 산출물을 하나 추가하면 자기 소유가 아닌 이 파일을 고쳐야 합니다. | 단독 PR. 해당 파이프라인 담당자와 key 이름을 먼저 맞춥니다. |
| `src/common/__init__.py` | `__all__`이 이름 20개를 정렬해 둔 목록 하나입니다. 두 사람이 각자 export를 추가하면 같은 자리에서 충돌합니다. | 단독 PR. 이름만 추가하고 다른 줄은 건드리지 않습니다. |
| `src/common/*.py` | storage, contract, config를 5개 파이프라인이 모두 씁니다. 동작이 바뀌면 전부 영향받습니다. | 단독 PR + 전체 test. |
| `contracts/README.md` | 파이프라인 사이의 합의 문서입니다. 합의 없이 바꾸면 실행이 깨집니다. | 제안 → 합의 → 반영. |
| `requirements.txt` | CUDA torch, FastAPI, boto3가 한 파일에 있습니다. 버전이 어긋나면 팀 전체가 실행하지 못합니다. | 단독 PR. `onboarding/docs/onboarding.md`의 고정 버전과 같이 확인합니다. |
| `configs/` | 공용 실행 설정입니다. | 단독 PR. |
| `pytest.ini` | marker 등록. 모든 test 실행에 영향을 줍니다. | 단독 PR. |
| `.github/` | CODEOWNERS, Pull Request template, workflow. | 단독 PR. |
| `tools/`, `scripts/` | 팀 전원이 쓰는 git alias와 검사 도구입니다. | 단독 PR + 해당 test. |

## 부딪히지 않는 곳

`src/pipelines/<area>/` 안은 그 담당자만 바꿉니다. 이 안에서는 자유롭게 작업해도 다른 사람과 겹치지 않습니다. 각 디렉터리의 지침서가 그 범위를 설명합니다.

## Git이 아니라 실행 중에 부딪히는 것

- `artifacts/web/jobs/`, `artifacts/web/configs/` — web server가 공유하는 상태입니다. 같은 clone에서 두 사람이 `python -m src.pipelines.web.server`를 띄우면 job 기록이 섞이고 port 8000도 겹칩니다. 한 번에 한 명만 띄웁니다.
- pytest 임시 디렉터리 — `--basetemp=artifacts/pytest-tmp`처럼 각자 다른 경로를 씁니다.

## 진행 상황 파일을 만들지 않는 이유

"지금 누가 무엇을 하는지" 적는 공유 markdown을 두면, 충돌을 막으려고 만든 그 파일이 **가장 자주 충돌하는 파일**이 됩니다. 모두가 매번 같은 줄을 고치기 때문입니다.

지금 무엇이 진행 중인지는 `git branch -r`과 GitHub Pull Request 목록이 이미 보여줍니다. branch 이름이 `pipeline/<area>/<task-summary>` 형식이라 누가 어느 영역을 잡고 있는지 그대로 드러납니다. 따로 파일을 만들지 않습니다.
