# Dependency 호환성 온보딩

이 절차는 모든 팀원이 같은 dependency version을 설치하고 PyTorch 기반 이미지 처리 library가 함께 작동하는지 확인합니다. 로컬 CPU, 로컬 NVIDIA GPU, Colab NVIDIA GPU를 지원하며 Intel XPU는 사용하지 않습니다. 실제 dataset, checkpoint, AWS, training pipeline은 실행하지 않습니다.

## 1. 저장소 접근과 clone

1. GitHub에서 저장소 invitation을 수락합니다.
2. 저장소를 clone하고 directory로 이동합니다.

```text
git clone https://github.com/LittleBitAI/codeit-beginner-project.git
cd codeit-beginner-project
```

clone이 성공하면 invitation과 repository read access가 확인된 것입니다.

## 2. 로컬 Conda 환경과 dependency 설치

Conda는 Python 3.11 환경 생성과 활성화에만 사용합니다. Project dependency는 모두 pip로 설치합니다. `requirements.txt`는 모든 팀원에게 PyTorch 2.12.1, TorchVision 0.27.1, CUDA 12.6 build를 설치합니다. NVIDIA GPU가 없는 PC도 같은 build를 CPU mode로 사용할 수 있습니다.

```text
conda create -n pill-detection python=3.11 -y
conda activate pill-detection
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

별도의 CUDA Toolkit은 필요하지 않습니다. NVIDIA GPU를 사용하는 팀원은 CUDA 12.x를 지원하는 NVIDIA driver가 필요합니다. Windows에서는 driver 528.33 이상을 사용합니다.

`requirements.txt`는 MMDetection도 함께 설치합니다(mmcv 2.2.0, mmdet 3.3.0, mmengine 0.10.7). mmcv는 torch와 CUDA에 맞춰 컴파일된 확장을 싣는데 공식 색인에 이 torch용 wheel이 없어, wheel 파일을 URL로 직접 가리키고 `#sha256`으로 내용까지 못 박습니다. 색인을 더하지 않으므로 다른 package는 영향을 받지 않고, 파일이 바뀌면 설치가 그 자리에서 멈춥니다. Windows와 Linux wheel만 있고 Python 3.11과 3.12용으로 각각 줄이 나뉘어 있으므로, 그 밖의 OS나 Python version에서는 mmcv가 설치되지 않습니다. 설치가 그 줄에서 멈춰도 임의로 `mmcv-lite`를 대신 깔지 마세요. 이름은 비슷하지만 CUDA 연산자가 없어서 모델을 만들 수 없습니다.

설치가 끝나면 컴파일된 확장이 실제로 실리는지 확인합니다. `mmcv-lite`가 깔렸거나 wheel이 torch와 어긋나면 여기서 드러납니다.

```text
python -c "import mmcv._ext; print('mmcv ops OK')"
```

## 3. 로컬 온보딩 검증

Repository root에서 다음 명령을 실행합니다.

CPU만 사용하거나 NVIDIA GPU가 없는 팀원:

```text
python onboarding/scripts/verify_onboarding.py --profile cpu
```

NVIDIA GPU를 로컬 학습에 사용할 팀원:

```text
python onboarding/scripts/verify_onboarding.py --profile cuda
```

마지막에 `ONBOARDING VERIFICATION PASSED`가 출력되어야 합니다. CPU profile은 Python 3.11, 고정 package version, `pip check`, PyTorch tensor와 autograd, NumPy 변환, Pillow image 변환, TorchVision NMS를 확인합니다. CUDA profile은 여기에 NVIDIA driver, CUDA 사용 가능 여부, GPU tensor와 GPU NMS를 추가로 확인합니다.

검증은 작은 in-memory tensor와 image만 사용합니다. 모델, dataset 또는 checkpoint를 다운로드하거나 파일로 저장하지 않습니다.

## 4. Colab 학습 환경 검증

Intel 내장 그래픽 또는 GPU가 없는 팀원은 Colab에서 NVIDIA GPU runtime을 선택합니다.

1. Colab의 `런타임 > 런타임 유형 변경`에서 NVIDIA GPU를 선택합니다.
2. 저장소를 clone하고 dependency를 설치합니다.
3. Colab profile로 검증합니다.

```text
!git clone https://github.com/LittleBitAI/codeit-beginner-project.git
%cd codeit-beginner-project
!python -m pip install --upgrade pip setuptools wheel
!python -m pip uninstall -y torchaudio
!python -m pip install -r requirements.txt
!python onboarding/scripts/verify_onboarding.py --profile colab
```

이 프로젝트는 audio를 사용하지 않습니다. Colab에 미리 설치된 `torchaudio`는 기존 PyTorch version을 요구할 수 있으므로 설치 전에 제거합니다. Colab profile은 Python 3.11 또는 현재 Colab의 Python 3.12를 허용하고 CUDA를 필수로 검사합니다. 이미 현재 runtime에서 `torch`를 import한 뒤 dependency를 다시 설치했다면 runtime을 재시작하고 검증을 다시 실행합니다.

실패 메시지는 Python version, 누락 또는 불일치 package, `pip` dependency 충돌, CUDA build, NVIDIA driver, GPU 사용 가능 여부, 실제 tensor 연산 중 어느 단계가 실패했는지 표시합니다. 문제를 수정한 뒤 같은 검증 명령을 다시 실행합니다.

## 5. 개인 상태 변경

GitHub username을 사용해 개인 branch를 만듭니다.

```text
git switch -c onboarding/<github-username>
```

`onboarding/docs/onboarding-status.md`에서 자신의 한 줄만 다음과 같이 변경합니다.

```text
- 팀원 N (미완)
- 팀원 N (완료)
```

다른 사람의 상태나 다른 파일은 변경하지 않습니다. 변경 범위는 다음 명령으로 확인합니다.

```text
git diff -- onboarding/docs/onboarding-status.md
git status --short
```

## 6. Commit, push, Pull Request

자신의 상태 파일만 commit합니다.

```text
git add onboarding/docs/onboarding-status.md
git commit -m "온보딩 환경 확인 완료"
```

GitHub CLI 인증 상태를 확인하고 이 repository에서 `git pr` alias를 한 번 설치합니다.

```text
gh auth status
python tools/git_pr.py --install
```

먼저 외부 변경 없이 계획을 확인한 뒤 같은 설명으로 `git pr`을 실행합니다. 요약은 무엇을
바꿨는지, 이유는 기존에 무엇이 불편했는지를 서로 다르게 한국어로 적습니다.

```text
git pr --dry-run --summary "내 온보딩 확인 상태를 완료로 바꿨습니다." --reason "담당자가 환경 준비 여부를 확인할 수 있게 하기 위해서입니다." --check "python onboarding/scripts/verify_onboarding.py --profile cpu → PASSED"
git pr --summary "내 온보딩 확인 상태를 완료로 바꿨습니다." --reason "담당자가 환경 준비 여부를 확인할 수 있게 하기 위해서입니다." --check "python onboarding/scripts/verify_onboarding.py --profile cpu → PASSED"
```

`git pr`은 `onboarding/<github-username>` branch를 `origin`에 push하고 `main` 대상 draft Pull Request를 만듭니다. Pull Request에는 `onboarding/docs/onboarding-status.md`의 자신의 상태 한 줄만 포함되어야 합니다.

Pull Request 설명에 자신이 사용한 profile의 검증 결과를 포함합니다.

```text
python onboarding/scripts/verify_onboarding.py --profile cpu
ONBOARDING VERIFICATION PASSED
```

검증 결과를 확인한 뒤 review를 요청합니다. 승인 전에는 merge하지 않습니다.
