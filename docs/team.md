# 팀 작업 규칙

저장소를 **쓰는** 방법은 `README.md`에 있습니다. 이 문서는 저장소에 **기여하는** 사람을 위한 것입니다 — 구조, 소유권, 저장 위치, Git 규칙.

## 저장소 구조

```
.
├── configs/                 # 공용 실행 설정
├── contracts/               # pipeline 간 공용 계약
├── docs/                    # 저장소 전체에 걸친 문서
├── onboarding/              # 신규 팀원 환경 확인
├── scripts/                 # 일회성·보조 스크립트
├── src/
│   ├── main_pipeline.py     # 지정된 통합 진입점
│   ├── common/              # 승인된 공용 코드 (storage, contract, config)
│   └── pipelines/           # data / train / evaluate / registry / web
└── data/, artifacts/        # Git에 저장하지 않는 데이터와 산출물
```

`src/pipelines/<area>/`는 담당자가 소유하는 독립 작업 구역입니다. 다른 담당자는 그 경계를 넘어 직접 수정하지 않고 Pull Request로 변경을 요청합니다. 요청은 `contracts/proposals/NNN-<주제>.md`에 적고, 소유자가 구현합니다.

각 pipeline은 다른 pipeline의 내부 모듈을 알거나 직접 호출하지 않습니다. 외부에는 `run(config) -> dict` interface만 제공하고, 실행 순서와 연결은 `src/main_pipeline.py`만 담당합니다. `web`은 그 순서에 들어가지 않는 별도 사용자 interface입니다.

## AI 도구 지침서

| 문서 | 누가 읽나 | 무엇이 있나 |
| --- | --- | --- |
| `CLAUDE.md` / `AGENTS.md` | AI 도구 | 코드 작성법, TDD 절차, PR 규칙 |
| `src/pipelines/<area>/CLAUDE.md` / `AGENTS.md` | AI 도구 | 그 디렉터리에서 무엇을 해도 되는가 |

두 파일은 **내용이 같습니다**. Claude Code는 앞쪽만, Codex는 뒤쪽만 읽습니다. 항상 같이 고칩니다. 모든 문서는 8,000자 이내이고, `python scripts/check_docs.py`가 이 규칙과 두 파일의 동일성을 검사합니다. pipeline 디렉터리에는 `README.md`를 두지 않습니다 — 낡기 때문입니다.

## 저장·실행 역할

- **GitHub** — code, 가벼운 config·contract·metadata, 작은 sample, 문서. Pull Request review의 기준점입니다.
- **외부 artifact 저장소** — 공통 storage interface를 통해 local filesystem 또는 Amazon S3에 dataset, checkpoint, weight, training log, 대량 prediction을 보관합니다. 개별 pipeline은 `boto3`를 직접 쓰지 않습니다.
- **로컬 또는 Colab** — 같은 Git revision과 config로 pipeline을 실행합니다.
- **Kaggle** — competition 규칙에 따른 최종 검증과 제출.

dataset, checkpoint, weight, event, log, cache, 환경, 큰 생성 파일은 commit하지 않습니다.

## Local 및 Amazon S3 storage

`src/common/storage.py`의 `create_storage(config)`가 local과 S3 backend를 같은 interface로 제공합니다. 기존 대상은 `overwrite=True`를 명시하지 않으면 덮어쓰지 않습니다.

Backend는 config 또는 환경 변수로 고르며 환경 변수가 우선합니다.

| 환경 변수 | 용도 |
| --- | --- |
| `PILL_STORAGE_BACKEND` | `local` 또는 `s3` |
| `PILL_STORAGE_LOCAL_ROOT` | local artifact root |
| `PILL_STORAGE_S3_BUCKET` | S3 bucket 이름 |
| `PILL_STORAGE_S3_PREFIX` | 선택적 공통 S3 key prefix |
| `PILL_WEB_STATE_WORKSPACE` | GUI 상태를 S3의 자기 자리에 함께 두는 이름 (Colab용) |
| `AWS_PROFILE`, `AWS_REGION` | AWS profile과 region |

AWS account, bucket, 권한은 저장소 밖에서 준비합니다. `aws sso login --profile <profile-name>`으로 임시 credential을 받은 뒤 **실제 값은 shell 환경 변수로 넣습니다.** `.env.example`은 어떤 이름이 필요한지 보여 주는 목록일 뿐이고, `.env` 파일을 자동으로 읽는 코드는 없습니다.

S3 object는 `datasets/`, `experiments/{uploading,completed,failed,web-state}/`, `registry/`, `submissions/`, `final-models/` prefix를 씁니다. `experiments/web-state/<이름>/`은 런타임이 사라져도 GUI가 그 학습을 이어서 할 수 있도록 job 기록과 설정을 두는 자리이고, `datasets/pill_detection/image-cache/`에는 실행 사이에 재사용하는 이미지 묶음이 있습니다.

별도 승인을 받은 경우에만 연결을 확인하는 smoke test를 실행합니다. 작은 임시 object 하나를 올리고 내려받아 확인한 뒤 지웁니다.

```powershell
python scripts/s3_smoke_test.py --config configs/env.aws.json
```

## Git 협업

모든 변경은 Pull Request로 반영하며 `main`에 직접 commit하지 않습니다. 작업 branch는 `pipeline/<area>/<task-summary>` 형식이고, `<area>`는 `data`, `train`, `evaluate`, `registry`, `web`, 저장소 전체 문서 작업은 `docs`입니다. 한 branch에는 한 가지 변경만 담고, merge 후 branch를 지웁니다. commit message는 한국어로 씁니다.

clone 후 한 번만 alias를 설치합니다. `git pr`은 GitHub CLI 설치와 `gh auth login`이 먼저 필요합니다.

```powershell
python tools/git_update_main.py --install   # git update-main
python tools/git_pr.py --install            # git pr
```

`git update-main`은 clean working tree를 확인하고 `main`에서 `git pull --ff-only`를 실행합니다. `git pr`은 branch 이름과 clean working tree를 확인한 뒤 push하고 `main` 대상 draft Pull Request를 만듭니다. `git pr --dry-run`으로 먼저 계획만 확인할 수 있습니다. 요약과 이유는 한국어로 전달해야 하고, 둘이 같으면 거부합니다.

## Test

저장소 root에서 실행합니다. `src`는 설치된 package가 아니라서 다른 위치에서는 import가 깨집니다.

```powershell
python -m pytest src tests onboarding/tests -q
cd src/pipelines/web/frontend; npm run typecheck; npm test
```

frontend의 `typecheck`는 별도 CI 단계이고 test 파일까지 읽습니다. 남길 test와 지울 test의 기준은 `docs/testing.md`에 있습니다.

## 남은 일

- `.github/CODEOWNERS`의 담당자를 실제 GitHub 계정으로 교체하고 approval 보호 규칙을 설정합니다.
- pipeline별 S3 artifact lifecycle 정책을 담당자 협의 후 확정합니다.
- `design_handoff_pill_detect_platform/`의 mock data와 가정값을 실제 계약과 구분합니다.
