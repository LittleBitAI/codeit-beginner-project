# 015. Web의 DataLoader worker 기본값

## 요청 이유

Train의 `num_workers` 기본값을 `0` 고정에서 실행 환경에 따라 정하도록 바꿉니다. Web은
Train을 import하지 않고 기본값을 복사하므로(`src/pipelines/web/train_config.py`의
`_INTEGER_FIELDS`), 함께 바뀌지 않으면 GUI로 시작한 학습만 계속 `0`으로 돌아 Colab에서
느립니다. `test_web_train_contract.py`는 이 값을 대조하지 않아 **조용히 어긋납니다.**

## 무엇이 문제였나

`num_workers=0`이면 주 process가 batch마다 직접 이미지를 읽습니다. 이 dataset은
976×1280 PNG 한 장이 1.8 MB이고, 한 장을 푸는 데만 22 ms가 듭니다(로컬 측정, Colab은
CPU가 느려 더 걸립니다). 그동안 GPU는 아무것도 하지 않으므로 10,495장이면 epoch마다
몇 분이 그대로 버려집니다.

## Train의 새 기본값

| 실행 환경 | 기본값 | 이유 |
| --- | ---: | --- |
| `device="cuda"`, Windows가 아님 | `min(4, os.cpu_count())` | 기다리는 GPU가 있고 worker를 쓸 수 있습니다 |
| `device="cpu"` | `0` | 미리 풀어 두어도 기다릴 GPU가 없습니다 |
| Windows | `0` | worker를 spawn으로 만들며 dataset을 pickle하는데, 그 안의 S3 client가 pickle되지 않아 첫 batch에서 죽습니다 |

사용자가 직접 적은 값은 어디서나 그대로 씁니다. 최소값 `0`, bool이 아닌 정수라는
검증 규칙은 그대로입니다.

## Web에 부탁드리는 것

1. `_INTEGER_FIELDS`의 `("num_workers", 0, 0)`에서 기본값을 위 표와 같은 규칙으로
   바꿉니다. 화면이 보여 주는 출발값도 그 값이어야 합니다.
2. `train.num_workers` 설명(`"0이면 주 process가 직접 읽습니다."`)에 기본값이 이제
   환경에 따라 정해진다는 것을 덧붙입니다.
3. Windows 경고(`"Windows에서는 worker를 늘려도 효과가 작고 메모리를 더 씁니다."`)는
   더 강한 문장으로 바꾸는 편이 맞습니다. 효과가 작은 것이 아니라 **실행이 죽습니다.**

## 안 바뀌는 것

`run(config)` 반환값, checkpoint payload, progress event 이름, artifact 경로는 그대로
입니다. Train과 Web 어느 쪽도 새 설정 key를 만들지 않습니다.
