# 017. Web의 `pill_geometric` 증강 preset 노출

## 요청

Train이 증강 preset `pill_geometric`을 추가했다. Web은
`src/pipelines/web/train_capabilities.py`의 목록에 이름 하나를 넣는다.

```python
SUPPORTED_AUGMENTATIONS = ("none", "pill_basic", "pill_geometric")
```

기본값 `DEFAULT_AUGMENTATION = "none"`은 그대로다.

## 지금 무엇이 울리고 있나

`test_web_train_contract.py::test_augmentation_choices_match_train_source`가
실패한다. 이 test는 train의 `AUGMENTATION_PRESETS`를 `ast`로 읽어 web의 목록과
대조하므로, **감시가 제대로 작동한 것**이다. 다른 test 50개는 그대로 통과한다.

```
assert ('none', 'pill_basic') == ('none', 'pill_basic', 'pill_geometric')
Right contains one more item: 'pill_geometric'
```

## 왜 필요한가

`v6-seed42-8020-group`의 EDA 리포트가 잰 값 세 가지가 근거다.

| 잰 것 | 값 | 뜻 |
| --- | --- | --- |
| `combinations.capture_conditions` | 상위 3종이 9,984장, 나머지 11장 | 촬영 조건이 사실상 각도(70·75·90)만 다르다 |
| `appearance.foreground_color_distance` | 23.17 | 그런데도 대회 test와 알약 색이 벌어져 있다 |
| `object_size.test_over_train.length_ratio` | 0.9925 | 알약 크기는 test와 학습이 같다 |

조명 변화가 데이터에 거의 없는데 test와는 색이 벌어져 있으므로, 그 간극을 메울
수단이 증강밖에 없다. 반대로 크기는 이미 맞으므로 확대는 필요하지 않다.
`pill_basic`의 밝기·대비 ±10%로는 이 색 차이를 덮지 못한다.

`pill_geometric`이 `pill_basic`과 다른 점은 네 가지다.

1. 90°의 배수 회전을 더한다. 좌우 뒤집기와 합쳐 정사각 대칭 8가지가 모두 25%씩
   나온다(상하 뒤집기는 그 안에 이미 들어 있어 0으로 둔다).
2. 색을 더 세게 흔든다(확률 0.3 → 0.8, 밝기·대비 0.1 → 0.3). 색조만 0.02 → 0.03
   으로 좁게 두는데, 알약 색 자체가 class 단서이기 때문이다.
3. 자르기(0.85~1.0배)를 더한다. 잘리는 알약은 box째로 버린다.
4. 약한 잡음(σ=0.008)을 더한다.

## Web에 부탁드리는 것

1. `train_capabilities.py`의 `SUPPORTED_AUGMENTATIONS`에 `"pill_geometric"`을
   더한다. **train과 같은 순서**여야 한다. 계약 test가 tuple을 그대로 비교한다.
2. `_FIELD_LABELS["augmentation"]`의 설명 문구가 `pill_basic`만 가리키고 있다.
   필요하면 함께 손봐 달라. 화면 문구라 web의 판단이다.
3. 기본값은 `none` 그대로 둔다. 이 제안은 **고를 수 있게** 하자는 것이지 기본을
   바꾸자는 것이 아니다.

## 안 바뀌는 것

- `run(config)` 반환값, checkpoint payload, artifact 경로, progress schema.
- `train.augmentation`이 받는 모양(`{"preset": "<이름>"}`)과 `preset` 외의 key를
  거부하는 규칙. 확률과 세기는 preset이 정하고 config로 못 바꾼다.
- `pill_basic`과 `none`의 값. **무작위 수를 뽑는 순서까지 그대로다.** 확률이 0인
  단계는 아예 뽑지 않게 만들었으므로, 같은 seed로 다시 돌린 옛 실험이 이 변경
  전과 똑같이 재현된다. 이미 남긴 checkpoint의 `training_config.augmentation`도
  그대로 읽힌다.
- `_short_augmentation`은 `pill_geometric`을 `geometric`으로 줄인다.
  `pill_basic`의 `basic`과 겹치지 않아 run_id 표기는 손댈 것이 없다.
