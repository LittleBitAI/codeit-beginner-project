# 알약 객체 탐지 프로젝트

## 프로젝트 목적

이 프로젝트는 이미지에서 알약 객체를 찾고, 각 객체의 class, bounding box, confidence score를 함께 예측하는 object detection 모델 및 재현 가능한 실험 관리 기반을 만드는 팀 프로젝트입니다. 현재 프로젝트 자료에서 정의한 과제는 이미지 한 장당 최대 4개의 알약을 예측하는 것입니다. 결과는 모델 실험과 competition 검증을 위한 것이며, 의약품 식별의 정확성이나 복약·의료 안전을 보장하지 않습니다.

## 저장소 아키텍처

팀 pipeline guide가 정의한 구조에 따라 최소 dummy pipeline, 공용 config·contract, test skeleton이 준비되어 있습니다. 실제 data processing, training, evaluation, API, AWS, frontend logic은 아직 구현하지 않았습니다.

```
.
├── README.md
├── CLAUDE.md
├── AGENTS.md
├── .github/                 # GitHub 협업 규칙과 소유권
├── configs/                 # 공용 실행 설정
├── contracts/               # pipeline 간 공용 계약
├── src/
│   ├── main_pipeline.py     # 지정된 통합 진입점
│   ├── common/              # 승인된 공용 코드
│   └── pipelines/
│       ├── data/
│       ├── train/
│       ├── evaluate/
│       ├── registry/
│       └── web/
└── data/, artifacts/        # Git에 저장하지 않는 데이터와 대용량 산출물
```

`src/pipelines/<area>/`는 담당자가 소유하는 독립 작업 구역입니다. 다른 담당자는 그 경계를 넘어서 직접 수정하지 않고 이슈와 review로 변경을 요청합니다. 공용 영역과 통합 파일은 명시적으로 배정된 담당자만 변경합니다.

각 pipeline은 다른 pipeline의 내부 모듈을 알거나 직접 호출하지 않습니다. 외부에는 공통 `run(config)` interface만 제공하고, pipeline 실행 순서와 연결은 지정된 통합 진입점만 담당합니다. 구체적인 입력·출력 schema와 artifact 계약은 향후 `contracts/`에서 확정합니다.

## 저장·실행 역할

- GitHub: code, 가벼운 config, contract, metadata, 작은 sample, 문서를 관리하고 branch review와 Pull Request merge의 기준점으로 사용합니다.
- 외부 artifact 저장소: dataset, checkpoint, weight, training log, 대량 prediction 등 Git에 넣지 않는 파일을 보관합니다. 저장소 종류와 경로는 TODO입니다.
- 로컬 또는 Colab: 같은 Git revision과 config를 기준으로 pipeline을 실행하는 환경입니다. 현재는 외부 runtime dependency가 없는 dummy pipeline만 제공합니다.
- Kaggle: competition 규칙에 따른 최종 검증과 제출에 사용합니다. 제출 형식, 제한, 평가 규칙은 실제 competition 명세 확인 전까지 확정하지 않습니다.

## 기본 dummy 실행

전체 pipeline 연결을 확인합니다.

```
python -m src.main_pipeline --config configs/base.json
```

하나의 pipeline만 확인할 때는 `--only`를 사용합니다.

```
python -m src.main_pipeline --only train
```

공통 test는 다음 명령으로 실행합니다.

```
python -m pytest -q
```

## Git 협업 규칙

모든 변경은 Pull Request로 반영하며 `main`에 직접 commit하지 않습니다. Pull Request에는 변경을 담을 임시 작업 branch가 필요하지만, 장기간 유지하는 개인 branch를 추가로 만들 필요는 없습니다. merge가 끝난 작업 branch는 삭제합니다.

1. `main`에서 `pipeline/<area>/<task-summary>` 형식의 임시 작업 branch를 만듭니다.
2. 한 branch에는 한 가지에 집중한 변경만 commit합니다.
3. `git pr`로 branch를 push하고 `main` 대상 draft Pull Request를 만듭니다.
4. Pull Request template과 검증 결과를 작성하고 담당자 review를 받습니다.
5. 승인을 받은 Pull Request만 merge하고 작업 branch를 삭제합니다.

`<area>`는 담당 pipeline 이름인 `data`, `train`, `evaluate`, `registry`, `web` 중 하나를 사용합니다. repository-wide 문서나 공용 파일처럼 어느 area에도 속하지 않는 변경은 임의의 값을 만들지 말고 팀장에게 branch 이름을 확인합니다.

commit은 요청된 변경만 담고, message는 한국어로 작성합니다. 표준 기술 용어는 English로 남길 수 있습니다.

예: `프로젝트 공통 문서와 AI 작업 규칙 추가`, `README에 Git 협업 규칙 정리`

### `git pr`로 Pull Request 만들기

저장소를 clone한 뒤 한 번만 다음 명령으로 local Git alias를 설치합니다. GitHub CLI 설치와 `gh auth login` 인증이 먼저 필요합니다.

```
python tools/git_pr.py --install
```

작업 branch에서 변경을 commit한 뒤 `git pr`을 실행합니다.

```
git pr
```

이 명령은 branch 이름과 clean working tree를 확인하고, 현재 branch를 `origin`에 push한 다음 `main` 대상 draft Pull Request를 만듭니다. 같은 branch에 열린 Pull Request가 있으면 새로 만들지 않고 push된 commit으로 기존 Pull Request를 갱신합니다. `main`이나 규칙에 맞지 않는 branch에서는 실행되지 않습니다.

외부 변경 없이 동작 조건과 실행 계획만 확인하려면 `git pr --dry-run`을 사용합니다. draft 생성 후 Pull Request template을 작성하고 검증 결과를 확인한 다음 review를 요청합니다.

## 저장소 공통 정책

- 모든 text file은 UTF-8 without BOM과 LF line ending을 사용합니다.
- 일반 Git history에는 raw/processed dataset, model checkpoint·weight, TensorBoard event, training run, 대량 prediction, local environment·cache, `.env`를 포함하지 않습니다.
- Git에는 code, 가벼운 config·contract·metadata, 작은 sample, 문서만 저장합니다.
- Git LFS 또는 GitHub Release 사용은 팀이 별도로 확정하기 전까지 선택 사항입니다.
- AWS, Kaggle, API credential과 기타 secret은 commit하거나 문서·log에 노출하지 않습니다. 환경 변수나 승인된 secret store를 사용하며 실제 값은 저장소 밖에서 관리합니다.

## 알려진 제한 및 TODO

- TODO: 실제 pipeline 구현에 필요한 dependency와 환경별 설치 방법을 구현 시점에 확정합니다.
- TODO: GitHub approval, CODEOWNERS 보호 규칙을 설정하고 실제 담당자 계정을 등록합니다.
- TODO: 외부 artifact 저장소, 접근 권한, 경로 정책을 확정합니다.
- TODO: 실제 dataset·class 정의, Kaggle 제출 형식·제한·평가 규칙을 competition 원문으로 확인합니다.
- TODO: 기존 design handoff의 mock data와 가정값을 실제 계약과 명확히 구분합니다.
