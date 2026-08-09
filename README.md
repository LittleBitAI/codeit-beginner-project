# 알약 객체 탐지 프로젝트

이미지에서 알약을 찾아 class, bounding box, confidence score를 예측하는 object detection 모델과, 재현 가능한 실험 관리 기반을 만드는 팀 프로젝트입니다. 과제는 이미지 한 장당 최대 4개의 알약을 예측하는 것입니다. 결과는 모델 실험과 competition 검증을 위한 것이며, 의약품 식별의 정확성이나 복약·의료 안전을 보장하지 않습니다.

## 문서 지도

문서마다 역할이 다릅니다. 찾는 내용에 맞는 문서 하나만 보면 됩니다.

| 문서 | 누가 읽나 | 무엇이 있나 |
| --- | --- | --- |
| `README.md` (이 문서) | 팀원·팀장 | 프로젝트가 무엇이고 어떻게 실행하는지 |
| `CLAUDE.md` / `AGENTS.md` | AI 도구 | **어떻게 일하는가** — 코드 작성법, TDD 절차, PR 규칙 |
| `src/pipelines/<area>/CLAUDE.md` / `AGENTS.md` | AI 도구 | **그 디렉터리에서 무엇을 해도 되는가** |
| `contracts/README.md` | 전원 | pipeline 사이의 공용 계약 |
| `docs/shared-files.md` | 전원 | 여러 명이 동시에 건드리면 부딪히는 파일 |
| `docs/testing.md` | 전원 | 어떤 test를 남기고 어떤 test를 지우는가 |
| `docs/colab.md` | GPU가 없는 팀원 | Colab에서 팀 S3 데이터로 학습 돌리기 |
| `onboarding/docs/onboarding.md` | 신규 팀원 | 환경 설치와 dependency 확인 |

`CLAUDE.md`와 `AGENTS.md`는 **내용이 같은 파일**입니다. Claude Code는 `CLAUDE.md`만, Codex는 `AGENTS.md`만 읽습니다. 두 파일은 항상 같이 고칩니다. 모든 문서는 5,000자 이내로 유지합니다.

## 저장소 구조

```
.
├── configs/                 # 공용 실행 설정
├── contracts/               # pipeline 간 공용 계약
├── docs/                    # 저장소 전체에 걸친 문서
├── src/
│   ├── main_pipeline.py     # 지정된 통합 진입점
│   ├── common/              # 승인된 공용 코드 (storage, contract, config)
│   └── pipelines/           # data / train / evaluate / registry / web
└── data/, artifacts/        # Git에 저장하지 않는 데이터와 산출물
```

`src/pipelines/<area>/`는 담당자가 소유하는 독립 작업 구역입니다. 다른 담당자는 그 경계를 넘어 직접 수정하지 않고 Pull Request로 변경을 요청합니다.

각 pipeline은 다른 pipeline의 내부 모듈을 알거나 직접 호출하지 않습니다. 외부에는 `run(config)` interface만 제공하고, 실행 순서와 연결은 `src/main_pipeline.py`만 담당합니다.

## 실행

저장소 root에서 실행해야 합니다. `src`는 설치된 package가 아니라서 다른 위치에서 실행하면 import가 깨집니다.

```powershell
python -m src.main_pipeline --config configs/base.json   # 전체 연결 확인
python -m src.main_pipeline --only train                 # 한 pipeline만
python -m pytest -q                                      # 공통 test
```

## 저장·실행 역할

- **GitHub** — code, 가벼운 config·contract·metadata, 작은 sample, 문서. Pull Request review의 기준점입니다.
- **외부 artifact 저장소** — 공통 storage interface를 통해 local filesystem 또는 Amazon S3에 dataset, checkpoint, weight, training log, 대량 prediction을 보관합니다. 개별 pipeline은 `boto3`를 직접 쓰지 않습니다.
- **로컬 또는 Colab** — 같은 Git revision과 config로 pipeline을 실행합니다.
- **Kaggle** — competition 규칙에 따른 최종 검증과 제출. 제출 형식과 평가 규칙은 competition 명세 확인 전까지 확정하지 않습니다.

## Local 및 Amazon S3 storage

`src/common/storage.py`의 `create_storage(config)`가 local과 S3 backend를 같은 interface로 제공합니다. 기존 대상은 `overwrite=True`를 명시하지 않으면 덮어쓰지 않습니다.

Backend는 config 또는 환경 변수로 고르며 환경 변수가 우선합니다.

| 환경 변수 | 용도 |
| --- | --- |
| `PILL_STORAGE_BACKEND` | `local` 또는 `s3` |
| `PILL_STORAGE_LOCAL_ROOT` | local artifact root |
| `PILL_STORAGE_S3_BUCKET` | S3 bucket 이름 |
| `PILL_STORAGE_S3_PREFIX` | 선택적 공통 S3 key prefix |
| `AWS_PROFILE`, `AWS_REGION` | AWS profile과 region |

AWS account, bucket, 권한은 저장소 밖에서 준비합니다. `aws sso login --profile <profile-name>`으로 임시 credential을 받은 뒤 실제 값을 shell 환경이나 commit하지 않는 `.env`에 넣습니다. S3 object는 `datasets/`, `experiments/{uploading,completed,failed,web-state}/`, `registry/`, `submissions/`, `final-models/` prefix를 씁니다. `experiments/web-state/<이름>/`은 런타임이 사라져도 GUI가 그 학습을 이어서 할 수 있도록 job 기록과 설정을 두는 자리이고, `datasets/pill_detection/image-cache/`에는 실행 사이에 재사용하는 이미지 묶음이 있습니다.

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

`git update-main`은 clean working tree를 확인하고 `main`에서 `git pull --ff-only`를 실행합니다. `git pr`은 branch 이름과 clean working tree를 확인한 뒤 push하고 `main` 대상 draft Pull Request를 만듭니다. `git pr --dry-run`으로 먼저 계획만 확인할 수 있습니다.

자세한 작업 규칙은 `CLAUDE.md`(또는 `AGENTS.md`)에 있습니다.

## 알려진 제한 및 TODO

- 실제 pipeline 구현에 필요한 dependency와 환경별 설치 방법을 구현 시점에 확정합니다.
- `.github/CODEOWNERS`의 담당자를 실제 GitHub 계정으로 교체하고 approval 보호 규칙을 설정합니다.
- pipeline별 S3 artifact schema와 lifecycle 정책을 담당자 협의 후 확정합니다.
- 실제 dataset·class 정의, Kaggle 제출 형식·제한·평가 규칙을 competition 원문으로 확인합니다.
- `design_handoff_pill_detect_platform/`의 mock data와 가정값을 실제 계약과 구분합니다.
