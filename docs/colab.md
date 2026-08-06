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

## 자주 막히는 곳

| 증상 | 원인 |
| --- | --- |
| `PILL_STORAGE_S3_BUCKET 환경 변수가 없습니다` | 3번 셀을 건너뛰었거나 런타임을 재시작한 뒤 다시 실행하지 않았습니다 |
| `dataset '...'의 artifact를 S3에서 찾지 못했습니다` | 이름 오타입니다. 4번 셀로 확인하세요 |
| CUDA를 못 찾는다 | 0번의 GPU runtime 선택 또는 1번 뒤 재시작을 빠뜨렸습니다 |
| 학습 시작이 오래 걸린다 | 이미지를 S3에서 받는 중입니다. 약 1.9 GB이고 **실행할 때마다 다시 받습니다**. 캐시가 임시 디렉터리라 실행이 끝나면 지워집니다 |
| 세션이 끊겼다 | 그 실행은 사라집니다. 다시 돌리면 새 실행 이름이 만들어지므로 이전 결과와 섞이지 않습니다 |

짧은 실험을 여러 번 돌릴 계획이면 다운로드가 매번 붙는다는 점을 감안하세요. epoch을 늘려 한 번에 오래 돌리는 편이 유리합니다.
