# 018. epoch마다 평가용 checkpoint를 남겨 주기를 바란다

## 왜 필요한가

지금 best epoch은 **validation loss**가 정합니다. `trainer.py`가 매 epoch loss를
비교해 가장 낮은 epoch을 `best_checkpoint.pt`로 남기고, 평가도 제출도 그 하나로
합니다.

그런데 `011`이 잰 것은 이 저장소의 validation이 train과 거의 같은 그림이라는
사실이었고, 로컬 mAP 0.9518 대 0.9572가 Kaggle에서 0.46354 대 0.44651로 **뒤집혔다**는
것이었습니다. loss는 그 mAP보다 더 앞단의 값입니다. 다중 객체 탐지에서 loss가 낮은
epoch이 정말로 상자를 잘 맞히는 epoch인지는 **아직 아무도 재 보지 않았습니다.**

재려면 그 epoch들이 남아 있어야 합니다. 지금은 loss가 가장 낮았던 하나와 마지막
하나뿐이라, "epoch 22가 더 나았을까"라는 질문에 답할 방법이 처음부터 없습니다.

## 요청 — train에게

**1. optional `train.archive_epochs_from`(정수)을 받는다.** 그 epoch부터 epoch마다
checkpoint를 하나씩 더 남긴다. **key가 없으면 지금과 완전히 같다** — 하나도 남기지
않는다. 켜 본 적 없는 실행이 갑자기 수 GB를 쌓으면 안 된다.

**2. 그 파일에는 optimizer 상태를 담지 않는다.** 이 파일의 쓸모는 "그 epoch이 얼마나
맞히는지" 하나뿐이고, 그러려면 `evaluate`가 읽는 값만 있으면 된다. DINO 기준 epoch당
570MB가 190MB가 되고, 20 epoch이면 11GB와 4GB의 차이다. 이어서 학습은 지금까지처럼
`last_checkpoint.pt`로 한다.

**3. 남긴 자리를 `artifacts`로 알려 준다.** 이름을 규칙으로 두면 읽는 쪽이 그 규칙을
복제하게 되고, S3 게시 경로에는 실행마다 다른 attempt id가 들어 있어 규칙만으로는
만들 수도 없다. 하나도 남기지 않은 실행에는 이 key가 없다.

## 요청 — registry에게

`TRAINING_KEYS`에 `archive_epochs_from`을 더해 주기를 바란다. 계약에 설정이 늘면
`test_every_contract_setting_is_summarized_or_deliberately_left_out`이 그 자리에서
빨개지므로, 이 제안이 반영되는 순간 함께 들어가야 한다. 어느 실행이 epoch을 남겨
두었는지는 나중에 "이 실험은 epoch을 골라 볼 수 있나"를 정하는 값이라 요약에 담을
값이지, 결과를 어디에 두는지가 아니다.

## 시작 epoch을 왜 사람이 정하는가

수렴 전 epoch은 어차피 이기지 못하는데 자리는 똑같이 차지한다. 그런데 어디가 수렴
지점인지는 model마다 다르다. 그래서 값 자체는 사람이 정하고, 계약은 **화면이 미리
채워 줄 값**만 들고 있으면 된다.

```
EPOCH_ARCHIVE_START = {MMDetection 두 모델: 8, 나머지 셋: 15}
```

train이 이 표를 기본값으로 쓰지는 않는다. 기본은 "남기지 않음"이다.

## 호환 조건

- `archive_epochs_from`이 없는 실행은 지금과 **완전히 같다.** `artifacts`의 key 4개도
  그대로다.
- checkpoint payload는 evaluate와의 계약이다. 보관본은 그 payload에서 optimizer 상태를
  **뺀 것**이므로, `evaluate/predictor.py`와 `mmdetection_backend.py`가 읽는 key는 전부
  그대로 있다.
- 남기는 시점은 `checkpoint_every`를 함께 따른다(기본 1이라 매 epoch). 주기를 늘려
  두면 보관도 그 주기로만 된다.
- 이어서 학습한 실행은 자기 run_id 아래에 자기가 돈 epoch만 남긴다. 앞선 실행의
  보관본을 건드리지 않는다.

## 이번 변경에서 하지 않는 것

- **어느 epoch이 제일 좋은지 고르지 않는다.** 파일을 남기는 것까지가 이 제안이다.
  재고 고르는 일은 evaluate와 web이 맡는다(`019`, 그리고 web의 훑기 화면).
- 보관본을 지우지 않는다. 승자가 정해져도 나머지는 남는다 — 남의 산출물을 지우는 것은
  사람의 결정이다.
- 보관본으로 이어서 학습하지 않는다. optimizer 상태가 없어서 이어붙일 수 없다.
