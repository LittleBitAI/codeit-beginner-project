# 알약 객체 탐지 — 사용 안내

이미지에서 알약을 찾아 class, bounding box, confidence score를 예측합니다. 이미지 한 장당 최대 4개까지 예측하며, 전처리부터 제출 파일까지 전 과정을 **브라우저 화면 하나**에서 할 수 있습니다.

결과는 모델 실험과 competition 검증을 위한 것이며, 의약품 식별의 정확성이나 복약·의료 안전을 보장하지 않습니다.

이 문서는 저장소를 **쓰는** 방법입니다. 기여하는 방법(구조, 소유권, Git 규칙)은 `docs/team.md`에 있습니다.

## 무엇까지 되나

화면에서 **전처리 → EDA → 학습 → 평가 → 앙상블 → `submission.csv`** 를 순서대로 합니다. Kaggle 최고 점수 **0.63594**를 만든 구성(검출 3개를 WBF로 합치고 crop 임베딩 3개로 점수를 다시 매기는 것)도 같은 화면에서 재현할 수 있습니다.

## 어디서 실행할지 고르기

| 상황 | 어디로 |
| --- | --- |
| NVIDIA GPU가 있는 Windows·Linux | 아래 **로컬에서 시작하기** |
| GPU가 없다 | `docs/pill-detection-colab.ipynb` (Colab, 위에서부터 실행) |
| 팀 S3 자격 증명이 없다 | `docs/reproduce.md` (공개 번들로 재현만) |

**MMDetection detector는 `device="cuda"`가 아니면 시작을 거부합니다.** `mmcv` 설치 파일은 Windows·Linux × Python 3.11·3.12용만 있어 macOS에서는 학습이 되지 않습니다.

## 로컬에서 시작하기

### 1. 설치

```powershell
git clone https://github.com/LittleBitAI/codeit-beginner-project.git
cd codeit-beginner-project
conda create -n pill-detection python=3.11 -y
conda activate pill-detection
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -c "import mmcv._ext; print('mmcv ops OK')"
```

마지막 줄이 통과해야 detector를 만들 수 있습니다. 더 자세한 확인은 `python onboarding/scripts/verify_onboarding.py`가 합니다.

### 2. 팀 S3 자격 증명

데이터와 checkpoint는 모두 팀 S3에 있습니다. 저장소에는 넣지 않으니 shell 환경이나 commit하지 않는 `.env`에 넣으세요.

```powershell
$env:PILL_STORAGE_BACKEND = "s3"
$env:PILL_STORAGE_S3_BUCKET = "<팀 bucket 이름>"
$env:AWS_ACCESS_KEY_ID = "<받은 값>"
$env:AWS_SECRET_ACCESS_KEY = "<받은 값>"
$env:AWS_DEFAULT_REGION = "ap-northeast-2"
```

SSO를 쓴다면 `aws sso login --profile <이름>` 뒤에 `$env:AWS_PROFILE`만 지정하면 됩니다. 연결 확인은 `python scripts/s3_smoke_test.py --config configs/env.aws.json`입니다(작은 임시 파일 하나를 올렸다 지웁니다).

### 3. 화면 띄우기

frontend는 처음 한 번만 빌드하면 됩니다.

```powershell
cd src/pipelines/web/frontend; npm ci; npm run build; cd ../../../..
python -m src.pipelines.web.server
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. 로그인은 없습니다. 한 clone에서 서버는 하나만 띄웁니다 — `artifacts/web/`와 8000번 port를 함께 쓰기 때문입니다.

## 화면에서 하는 일

왼쪽 목록을 위에서부터 따라가면 됩니다.

1. **데이터 준비** — 원본 경로에 `datasets/pill_detection/raw/v5/original/`을 넣고 비율 8:2, Seed 42로 실행합니다. 나누는 방식은 칸이 없고 data pipeline이 `group`으로 나눕니다 — 같은 조합의 이미지가 학습과 검증으로 갈리지 않게 묶는 방식이고, 갈리면 검증 점수가 실제보다 높게 나옵니다. **`참조 crop 은행도 만들기`를 함께 켜 두세요** — 6번 재순위에 필요합니다. 나중에 더할 수도 있지만 `이미 있으면 덮어쓰기`를 켜고 준비를 통째로 다시 돌아야 합니다.
2. **EDA** — 준비한 판을 그대로 읽어 class 분포와 상자 크기를 잽니다.
3. **새 실험** — 학습을 시작합니다. `점수를 받은 설정 채우기 · 최고 점수 detector` 버튼이 있으면 0.62437을 받은 설정이 한 번에 채워집니다. 직접 고른다면 model `dino`, backbone `resnet50`, `input_size` 1280, `precision` `amp`, `device` `cuda`가 출발점입니다. **시연이라면 `epochs`를 1~2로 낮추세요** — 12 epoch은 진짜 판에서 하루 가까이 걸립니다.
4. **평가** — 끝난 실행을 평가합니다. test manifest가 있으면 `submission.csv`가 함께 만들어지고 화면에서 내려받을 수 있습니다.
5. **임베딩 학습** — 1번에서 만든 crop 은행으로 잘라 낸 알약이 어떤 class인지 재는 자를 만듭니다. 학습 대기열을 함께 쓰므로 detector 학습 중에는 시작하지 않습니다.
6. **앙상블** — 끝난 실행 둘 이상을 골라 합치고(**모델**), 임베딩으로 점수를 다시 매깁니다(**임베딩**). 합치기 전에 얼마나 닮았는지 진단해 줍니다. **약한 실행을 넣으면 점수가 내려갑니다** — 7개 0.62087 < 단독 0.62437 < 상위 3개 0.62645.

로컬 검증 점수는 Kaggle 점수를 예측하지 못합니다. 독립 실험 셋이 모두 무관하거나 반대로 움직였으니, 화면의 mAP만 보고 제출을 고르지 마세요.

## 명령줄로 돌리기

화면 없이 같은 일을 할 수 있습니다. **저장소 root에서** 실행해야 합니다 — `src`는 설치된 package가 아니라서 다른 위치에서는 import가 깨집니다.

```powershell
python -m src.main_pipeline --config configs/base.json    # data → train → evaluate → registry
python -m src.main_pipeline --only data --config configs/prepare.v5.aws.json
python -m src.main_pipeline --only evaluate --config configs/reproduce.best.json
```

마지막 줄이 최고 점수 구성을 그대로 재현합니다(RTX 3080에서 5~6분). 앙상블과 재순위는 별도 stage가 아니라 `evaluate`가 config를 보고 함께 합니다.

## 문서 지도

| 문서 | 무엇이 있나 |
| --- | --- |
| `README.md` (이 문서) | 설치하고 화면을 띄워 제출까지 |
| `docs/pill-detection-colab.ipynb` | Colab에서 같은 일을 하기 |
| `docs/reproduce.md` | 자격 증명 없이 최고 점수 재현하기 |
| `docs/colab.md` | Colab에서 명령줄로 학습만 돌리기 |
| `docs/team.md` | 저장소 구조, 소유권, storage, Git 규칙 |
| `docs/testing.md` | 어떤 test를 남기고 어떤 test를 지우는가 |
| `docs/shared-files.md` | 여러 명이 동시에 건드리면 부딪히는 파일 |
| `contracts/README.md` | pipeline 사이의 공용 계약 |
| `onboarding/docs/onboarding.md` | 환경 설치와 dependency 확인 |
