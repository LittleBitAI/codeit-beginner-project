# Train 이어서 학습 제안

## 왜 필요한가

지금은 학습이 **끝나야만** checkpoint가 디스크에 생긴다. `trainer.py`의
`_train_model`이 epoch 루프를 다 돈 뒤에야 `(best, last, history)`를 반환하고,
`pipeline.py`의 `_write_checkpoint`가 그때 처음 파일을 쓴다.

그래서 50 epoch 중 40에서 Colab 세션이 끊기면 **이어서 할 대상 자체가 없다.**
`docs/colab.md`도 "세션이 끊겼다 → 그 실행은 사라집니다"라고 적어 두었고, Web은
중단된 학습에 "지금 죽이면 그때까지 학습한 것이 전부 사라집니다"를 띄우고 있다.

데이터가 232장에서 10,553장으로 늘면서 한 번의 학습이 길어졌다. 길어질수록 통째로
잃을 확률이 커진다.

## 요청

**1. epoch마다 checkpoint를 남긴다.** 이것 하나가 나머지를 가능하게 한다. 매 epoch가
부담이면 optional `train.checkpoint_every`(기본 1)로 간격을 정해도 좋다. 쓰는 중에
죽어도 이전 파일이 남도록, 기존 checkpoint와 같이 임시 파일에 쓰고 옮기는 방식이면
좋겠다.

**2. optional `train.resume_from` 문자열을 받는다.** 그 checkpoint에서 이어서
학습한다. key가 없으면 지금과 완전히 같다.

**3. 이어서 시작할 때 아래를 복원한다.**

| 값 | 복원하지 않으면 |
| --- | --- |
| `model_state_dict`, `optimizer_state_dict` | (이미 저장하고 있다) |
| 시작 `epoch` | (이미 저장하고 있다) 몇 번째부터 돌지 알 수 없다 |
| RNG 상태 | `CLAUDE.md`의 "A seeded run must reproduce"를 지킬 수 없다 |
| 조기 종료 카운터와 기준 loss | patience가 초기화되어 멈춰야 할 학습이 계속 돈다 |
| `training_history` | 손실 곡선이 이어서 시작한 지점부터만 그려진다 |

`epoch`와 두 state는 이미 payload에 있으므로, 새로 담을 것은 RNG 상태·조기 종료
상태·지금까지의 history다.

## 호환 조건

- `resume_from`이 없는 실행은 지금과 **완전히 같다.** 기본값이 곧 기존 동작이다.
- `run(config)`의 반환 key 4개와 `artifacts`의 이름은 바뀌지 않는다.
- **checkpoint payload는 evaluate와의 계약이다.** `architecture`와
  `model_state_dict`를 포함해 `src/pipelines/evaluate/predictor.py`가 읽는 key는
  그대로 두고 새 key만 더한다. 새 key가 없는 옛 checkpoint도 계속 평가할 수 있어야
  한다.
- `training_history.json` 형식을 바꾸지 않는다. 이어서 학습한 실행의 history는 앞선
  epoch까지 포함해 한 줄로 이어지는 것이 좋다.
- 이어서 학습한 실행이 어떤 checkpoint에서 왔는지 `summary`나
  `training_config`에 남으면, 나중에 결과를 볼 때 그 실행이 이어붙인 것임을 알 수
  있다.

## Web이 맡는 부분

- 중단된 학습에서 "이어서 학습"을 시작하는 화면. 중단된 job은 이미
  `interrupted`로 구분해 두고 있어 그 자리에 붙일 수 있다.
- `resume_from`의 검증 규칙을 `train_config.py`에 복제하고, 기존 방식대로 train
  source를 `ast`로 읽는 계약 test로 drift를 감시한다.
- 이 제안이 반영되기 전에는 Web에서 할 수 있는 일이 없다. 이어서 할 파일이 없기
  때문이다.

## 이번 변경에서 하지 않는 것

- 중단된 학습을 **자동으로** 이어서 시작하지 않는다. 사람이 화면에서 고른다.
- 학습 중간 결과를 S3에 올리는 주기는 이 제안이 정하지 않는다.
