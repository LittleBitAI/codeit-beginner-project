# 저장소 온보딩

이 절차는 역할 배정 전에 저장소 접근, Python 환경, 기본 Git 협업 흐름만 확인합니다. 프로젝트 pipeline, dataset, artifact, AWS, training, evaluation은 사용하지 않습니다.

## 1. 저장소 접근과 clone

1. GitHub에서 저장소 invitation을 수락합니다.
2. 저장소를 clone하고 directory로 이동합니다.

```text
git clone https://github.com/LittleBitAI/codeit-beginner-project.git
cd codeit-beginner-project
```

clone이 성공하면 invitation과 repository read access가 확인된 것입니다.

## 2. Conda 환경과 dependency 설치

Conda는 Python 환경 생성과 활성화에만 사용합니다. Project dependency는 모두 pip로 설치합니다.

```text
conda create -n pill-detection python=3.11 -y
conda activate pill-detection
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 3. 온보딩 검증

Repository root에서 다음 명령을 실행합니다.

```text
python onboarding/scripts/verify_onboarding.py
```

마지막에 `ONBOARDING VERIFICATION PASSED`가 출력되어야 합니다. 이 명령은 Python 3.11, required package import, required file, UTF-8 호환성과 onboarding marker test를 확인하며 project pipeline은 실행하지 않습니다.

## 4. 개인 상태 변경

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

## 5. Commit, push, Pull Request

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

먼저 외부 변경 없이 계획을 확인한 뒤 `git pr`을 실행합니다.

```text
git pr --dry-run
git pr
```

`git pr`은 `onboarding/<github-username>` branch를 `origin`에 push하고 `main` 대상 draft Pull Request를 만듭니다. Pull Request에는 `onboarding/docs/onboarding-status.md`의 자신의 상태 한 줄만 포함되어야 합니다.

Pull Request 설명에 다음 검증 결과를 포함합니다.

```text
python onboarding/scripts/verify_onboarding.py
ONBOARDING VERIFICATION PASSED
```

검증 결과를 확인한 뒤 review를 요청합니다. 승인 전에는 merge하지 않습니다.
