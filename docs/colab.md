# Colab에서 학습 돌리기

GPU가 없거나 로컬이 느릴 때 Colab에서 학습만 돌립니다. 데이터와 checkpoint는 모두 팀 S3에 있으므로 파일을 올리고 내릴 일은 없습니다.

셀을 위에서부터 그대로 실행하세요. config는 손으로 쓰지 말고 스크립트가 만들게 합니다.

## 0. GPU runtime 선택

`런타임 > 런타임 유형 변경`에서 **NVIDIA GPU**를 고릅니다. 이걸 빠뜨리면 아래가 다 되고 학습 시작 직전에 실패합니다.

## 1. 저장소와 dependency

```python
!git clone https://github.com/LittleBitAI/codeit-beginner-project.git
%cd codeit-beginner-project
!python -m pip install --upgrade pip setuptools wheel
!python -m pip uninstall -y torchaudio
!python -m pip install -r requirements.txt
```

**여기서 런타임을 재시작합니다** (`런타임 > 세션 다시 시작`). Colab에 미리 깔린 torch가 메모리에 남아 있어 재시작 없이는 새로 설치한 version이 잡히지 않습니다. 재시작 뒤에는 `%cd codeit-beginner-project`를 다시 실행해야 합니다.

## 2. 환경 검증

```python
%cd /content/codeit-beginner-project
!python onboarding/scripts/verify_onboarding.py --profile colab
```

Python version, package, CUDA, 실제 tensor 연산까지 확인합니다. 실패하면 어느 단계인지 알려 주므로 그걸 먼저 고칩니다.

## 3. AWS 자격 증명과 bucket

```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = ""      # 팀에서 받은 값
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_DEFAULT_REGION"] = "ap-northeast-2"
os.environ["PILL_STORAGE_S3_BUCKET"] = "" # 팀 bucket 이름
```

키를 노트북에 그대로 적으면 저장된 노트북에 남습니다. **단기 키를 쓰고 학습이 끝나면 세션을 삭제**하세요. 저장소에 commit하지 않습니다. Colab의 보안 비밀 기능을 쓰면 더 낫습니다.

## 4. 어떤 dataset이 있는지 확인

```python
!python scripts/make_colab_config.py --list-datasets
```

쓸 수 있는 이름과, 필수 artifact가 없어서 쓸 수 없는 이름을 나눠서 보여 줍니다.

## 5. config 만들기

```python
!python scripts/make_colab_config.py \
    --dataset v1-seed42-8020 \
    --architecture retinanet_resnet50_fpn_v2 \
    --optimizer AdamW \
    --epochs 30 \
    --batch-size 2
```

결과를 바꾸는 값은 기본값을 두지 않고 직접 받습니다. 지원하는 architecture 목록은 `src/pipelines/train/model.py`, optimizer는 AdamW·SGD·Adam입니다. `--learning-rate`를 주지 않으면 train이 optimizer별 기본값을 씁니다.

실행 이름은 `colab-<시각>-<네 자리>`로 자동으로 만들어집니다. 어디서 돌린 실험인지 목록에서 바로 보이고, 팀원끼리 이름이 겹치지 않습니다.

## 6. 학습 실행

```python
!python -m src.main_pipeline --only train --config artifacts/colab/train.json
```

`--only train`을 빼면 data부터 registry까지 전부 돌려고 합니다.

## 결과는 어디로 가나

checkpoint와 학습 기록은 `s3://<bucket>/experiments/completed/<실행 이름>/`에 올라갑니다. 같은 이름의 결과가 이미 있으면 덮어쓰지 않고 실행을 거부합니다.

평가와 registry 등록은 이 문서 범위가 아닙니다. 학습이 끝나면 로컬 GUI에서 그 checkpoint로 이어서 하세요.

## 인자 대신 화면으로 하고 싶다면

5·6번 대신 GUI를 띄웁니다. 화면에서 고르고 시작하고 진행 로그를 봅니다.

```python
!cd src/pipelines/web/frontend && npm ci && npm run build
```

```python
import os, subprocess, time
from google.colab import output
os.environ["PILL_TEAM_SYNC_ENABLED"] = "true"
os.environ["PILL_TEAM_ACTOR"] = "이름 (Colab)"   # 팀 화면에 이렇게 남습니다
os.environ["PILL_WEB_STATE_WORKSPACE"] = "hyunwoo-colab"   # 런타임이 끊겨도 되찾을 이름
# 나머지 PILL_TEAM_* 값은 팀에서 받은 것을 그대로 넣습니다
subprocess.Popen(["python", "-m", "src.pipelines.web.server"])
time.sleep(8)
output.serve_kernel_port_as_iframe(8000, height=900)   # 창 방식은 브라우저 보안 변경으로 막혔습니다
```

`PILL_TEAM_ACTOR`를 넣으면 **로그인 없이도 이 학습이 팀 활동에 올라갑니다.** 다른 팀원이 자기 PC에서 `/team` 화면을 열면 진행 상황과 로그가 실시간으로 보입니다.

이 이름은 Cognito가 확인해 주는 값이 아니라 직접 적는 값이라, 팀 화면에는 `이름 직접 입력`이라고 함께 표시됩니다. 본인 이름을 정확히 적어 주세요.

**`PILL_TEAM_ACTOR`를 반드시 넣으세요.** 넣지 않으면 화면이 로그인부터 요구하는데, Colab 주소는 세션마다 바뀌어 Cognito에 등록할 수 없으므로 그 벽을 넘을 수 없습니다. 팀 공유가 필요 없으면 `PILL_TEAM_SYNC_ENABLED=false`로 두면 로그인 없이 열립니다.

**Colab에서 `/team` 탭은 "이 환경에서는 팀 기록을 볼 수 없습니다"만 보여 줍니다.** 팀 기록을 *읽는* 것은 로그인이 필요하고, 이 주소로는 로그인할 수 없기 때문입니다. 쓰기만 IAM으로 열려 있어서 여기서 시작한 학습은 팀에 잘 올라갑니다. 팀 활동은 로그인이 되는 PC의 화면에서 보세요.

### 런타임이 끊겨도 화면에서 이어서 합니다

`PILL_WEB_STATE_WORKSPACE`를 넣어 두면 job 기록과 설정이 그 이름의 칸으로 S3에 함께
남습니다. 새 런타임에서 위 셀들을 처음부터 다시 실행하면 그 학습이 **중단됨**으로
보이고 `이어서 학습` 버튼이 붙습니다. 누르면 epoch마다 저장한 checkpoint에서
이어집니다.

**매번 같은 이름을 쓰세요.** 이름이 곧 자기 칸이라, 이름이 달라지면 앞의 기록을 찾지
못합니다. 다른 팀원과 같은 이름을 쓰면 기록이 섞입니다.

### 서버 셀을 다시 실행하지 마세요

서버가 하나 더 뜨는데 port 8000이 이미 잡혀 있어 곧 죽습니다. 화면을 다시 열고 싶으면 `output.serve_kernel_port_as_iframe(8000, height=900)` 줄만 따로 실행하세요.

### 서버가 죽어도 학습은 계속 돕니다

학습은 서버와 다른 session에서 돌기 때문에 서버만 죽어도 함께 죽지 않습니다. 화면이 `서버가 다시 시작되어…`라고 해도 먼저 확인하세요.

```python
!ps -ef | grep "[s]rc.main_pipeline"
```

부모 PID가 `1`인 줄이 보이면 서버는 죽었고 학습은 살아 있습니다. **그 process를 죽이지 마세요.** 그대로 두면 결과는 평소대로 S3에 올라갑니다. 죽이면 돌고 있던 epoch을 버리게 되고, 로그도 화면에 다시 이어지지 않으며 팀 활동에는 중단으로 남습니다. 같은 GPU에 새 학습을 얹지도 마세요.

## 자주 막히는 곳

| 증상 | 원인 |
| --- | --- |
| 화면이 `서버가 다시 시작되어…`라고 한다 | 서버 process만 죽었습니다. 위 `ps`로 학습이 살아 있는지 보고, 살아 있으면 그대로 두세요. `런타임이 사라졌습니다`라고 한다면 다른 런타임에서 돌던 학습이라 찾을 process가 없습니다 |
| `PILL_STORAGE_S3_BUCKET 환경 변수가 없습니다` | 3번 셀을 건너뛰었거나 런타임을 재시작한 뒤 다시 실행하지 않았습니다 |
| `dataset '...'의 artifact를 S3에서 찾지 못했습니다` | 이름 오타입니다. 4번 셀로 확인하세요 |
| CUDA를 못 찾는다 | 0번의 GPU runtime 선택 또는 1번 뒤 재시작을 빠뜨렸습니다 |
| 학습 시작이 오래 걸린다 | 이미지를 S3에서 받는 중입니다. 첫 batch 전에 한꺼번에 받아 두므로 여기서 한 번 기다립니다. `image_cache_progress` 줄에 받은 장수가 나옵니다 |
| 세션이 끊겼다 | 화면을 쓴다면 위 "런타임이 끊겨도 화면에서 이어서 합니다"를 보세요. 명령줄로 돌렸다면 `train.resume_from`에 `s3://<bucket>/experiments/completed/<실행 이름>/running/last_checkpoint.pt`를 적고 **새 실행 이름**으로 돌립니다. `epochs`는 남은 수가 아니라 전체 목표를 그대로 적습니다 |

이미지는 학습을 시작하기 전에 여러 장씩 동시에 받아 둡니다. 런타임이 끊겨 다시 시작해도 이미 받아 둔 것은 건너뛰고 남은 것만 받습니다. 다만 런타임이 바뀌면 디스크가 비므로 그때는 처음부터 다시 받습니다.
