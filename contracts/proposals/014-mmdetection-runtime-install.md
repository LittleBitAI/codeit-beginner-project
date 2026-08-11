# 014. MMDetection 실행 환경 설치 경로

## 요청 이유

`012-mmdetection-checkpoint-inference.md`(evaluate)와
`013-web-mmdetection-model-options.md`(web)가 이미 있습니다. 두 신규 architecture를
실제로 고를 수 있게 하려면 하나가 더 필요합니다. **실행 환경에 mmdet이 없습니다.**

Colab은 `requirements.txt`만 설치합니다. 그 파일에는 `mmdet`, `mmcv`, `mmengine`이
없으므로, 깨끗한 runtime에서는 모델을 만들기도 전에 import에서 멈춥니다. adapter는 그
상황을 다음 오류로 알립니다.

```
TrainError: MMDetection architecture requires mmdet, mmcv, and mmengine
```

`requirements.txt`는 공용 파일이라 단독 PR로만 바꿉니다(`docs/shared-files.md`). 그래서
직접 고치지 않고 요청으로 남깁니다.

## 먼저 확인해 주기를 바라는 것

이 문서는 "이 줄을 넣어 달라"가 아니라 **"버전 조합이 성립하는지 먼저 확인해 달라"**는
요청입니다. `mmcv`가 평범한 순수 python 패키지가 아니기 때문입니다.

`mmcv`는 **torch와 CUDA 버전에 맞춰 컴파일된 wheel**입니다. 지금 고정 버전은
`torch==2.12.1+cu126`입니다. 여기에 맞는 wheel이 openmmlab 색인에 없으면 pip가 소스
빌드로 넘어가면서 Colab에서 20분 넘게 걸리거나 그냥 실패합니다.

확인이 필요한 조합입니다.

```
mmengine>=0.10
mmcv>=2.1,<2.3     # torch 2.12 + cu126 wheel 존재 여부가 관건
mmdet>=3.3
```

`onboarding/docs/onboarding.md`의 고정 버전도 같이 봐야 합니다.

## 이것이 이론이 아니라는 근거

이 저장소의 개발 환경(`pill-detection`)에는 `mmdet`, `mmcv`, `mmengine`이 이미
설치돼 있습니다. 그런데도 두 모델을 **한 번도 만들지 못했습니다.**

```
ModuleNotFoundError: No module named 'mmcv._ext'
```

`import mmdet.registry`만 해도 `mmcv.ops`를 타면서 여기서 멈춥니다. DINO는
MultiScaleDeformableAttention, Cascade R-CNN은 RoIAlign을 쓰는데 둘 다 컴파일된
확장이 필요합니다. 즉 **패키지 이름만 requirements에 넣는 것으로는 되지 않습니다.**
어떤 색인에서 어떤 wheel을 받을지가 실제 문제입니다.

같은 이유로 이번 작업에서는 첫 batch를 실제로 돌려 보는 검증을 하지 못했습니다.
대신 설치된 mmdet source를 읽어 계약을 맞췄습니다.

## 이번 PR이 한 것

`src/pipelines/train/mmdetection_adapter.py`와 그 test뿐입니다. 두 architecture는
`model.py`의 `SUPPORTED_ARCHITECTURES`에 **넣지 않았습니다.** `train.architecture`로
이름을 보내면 거부하므로, 위 세 가지가 준비되기 전에는 아무도 고를 수 없습니다.
다른 pipeline의 파일은 하나도 고치지 않았고 train이 새로 받는 설정도 없습니다.
**지금 학습 동작은 이 PR 전과 한 글자도 다르지 않습니다.**

013이 적어 둔 `input_size`와 `gradient_accumulation_steps`도 같은 이유로 이번 PR에서
뺐습니다. train이 숫자 설정을 먼저 받으면 web의 복제본과 어긋나
`test_numeric_defaults_match_train_source`가 울립니다. 순서는 web이 먼저입니다.

## 순서 제안

1. **013** — web이 architecture 목록과 두 설정의 기본값을 복제합니다.
2. **012** — evaluate가 `backend="mmdetection"` checkpoint를 읽습니다.
3. **014** — `requirements.txt` 단독 PR로 설치 경로를 고정합니다.
4. train이 allowlist를 열고 `input_size`, `gradient_accumulation_steps`, 8GB 실행
   제약(`cuda`/`amp`/`AdamW`/`batch_size=1`)을 함께 내는 후속 PR을 냅니다.

1·2·3의 순서는 서로 바뀌어도 됩니다. 네 번째가 마지막이어야 한다는 것만 지키면 됩니다.
