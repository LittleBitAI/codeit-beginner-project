# 003. Experiment summary에 학습 설정 넣기

## 상태와 목적

제안. 실험 목록 화면이 **모델과 하이퍼파라미터를 채우지 못한다.** 목록은 계약상
index만 읽는데 index summary에 그 값이 없기 때문이다. Registry가 summary에
`training` 블록을 추가해 주기를 요청한다.

지표 계산이나 record schema는 바뀌지 않는다. index summary 문서에 key를 더하는
변경이다.

## 지금 무엇이 비어 있나

`GET /api/train/experiments`는 `list_experiment_summaries()`가 준 index만 읽어
화면 형식으로 옮긴다. summary에 없는 값은 `src/pipelines/web/experiments.py`의
`_summary_base()`가 전부 `None`으로 둔다.

```python
"model": {"architecture": None, "pretrained": None, "source": "record"},
"optimizer": {"name": None, "learning_rate": None, ...},
"training": {"device": None, "epochs": None, "batch_size": None, ...},
```

그래서 목록에는 `mAP`와 `seed`만 값이 있고 모델·optimizer·하이퍼파라미터 칸은
전부 `-`다. 실제 값은 사용자가 실험을 골라 비교를 요청해 record를 읽어야 나온다.
팀원이 목록만 훑어서는 "누가 어떤 모델로 얼마를 냈는지" 비교할 수 없다.

## Registry에 요청하는 변경

Index summary에 `training` 블록과 `training_source`를 추가해 주기를 요청한다.
`metrics` / `metrics_source`와 같은 짝 구조다.

```json
{
  "summary_version": "1",
  "run_id": "exp-0001",
  "metrics": {"mAP": 0.31, "...": "..."},
  "metrics_source": "metrics_file",
  "training": {
    "architecture": "retinanet_resnet50_fpn_v2",
    "pretrained": true,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "momentum": null,
    "weight_decay": 0.01,
    "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
    "device": "cuda",
    "epochs": 50,
    "batch_size": 4,
    "num_workers": 0
  },
  "training_source": "config_snapshot"
}
```

- 출처는 record에 이미 있는 `config_snapshot.train`이다. Registry가 새로 계산할
  값은 없고, 이미 redact를 거친 값이다.
- `metrics`와 같은 규칙을 따른다. key 이름은 train이 쓰는 이름을 그대로 쓰고 한 번
  더 번역하지 않는다. 값이 없으면 예외 없이 `null`로 두고 `training_source`를
  `"unavailable"`로 적는다.
- `config_snapshot.train` 자체가 없는 옛 record도 실패시키지 않는다. 그때도
  `training_source`는 `"unavailable"`이다.
- `seed`는 summary 최상위에 이미 있으므로 중복해 넣지 않는다.
- 값이 없을 때 기본값을 채워 주지 않기를 요청한다. 기록에 없는 것과 기본값을 쓴
  것은 다르고, 그 구분은 화면이 `(호환 기본값)` 표시로 한다.

## best_epoch과 best_validation_loss는 이 제안에 넣지 않는다

두 값은 **record에도 없다.** train은 그 값을 `run()`이 돌려주는 summary에만 담고,
registry record의 `pipelines.train`에는 artifact URI만 남는다. 즉 registry가 지금
줄 수 있는 값이 아니다.

train → registry 경로를 먼저 정해야 하므로 별도 제안으로 다룬다. 그때까지 화면은
두 값을 계속 `-`로 둔다.

## 왜 Web이 직접 고치지 않는가

`contracts/README.md`가 목록·검색·비교는 index만 읽고 record로 fallback하지 않는다고
정해 두었다. Web이 목록을 채우려고 record를 읽으면 그 규칙을 깨뜨린다. index 문서의
형식은 registry 소유이므로 제안서로 요청한다.

## Web이 하는 일

이 제안이 머지되고 registry가 구현하면, Web은 `_summary_base()`의 하드코딩 `None`을
summary의 `training` 읽기로 바꾼다. `/api/train/experiments` 응답 형식과
`ExperimentSummary` type은 바뀌지 않으므로 화면 코드는 그대로다.

비교 경로(`POST /api/train/experiments/compare`)는 지금처럼 record를 읽어 채운다.
record가 진실이므로 두 경로가 다른 값을 보이면 record가 이긴다.

## 호환성

- record schema(`1.2`)와 `metrics.json`은 바뀌지 않는다.
- `summary_version`은 `"1"` 그대로 둘 수 있다. key 추가만 있고 기존 key의 이름과
  의미는 그대로이며, 옛 index를 읽는 쪽은 `training`이 없으면 지금과 같이 동작한다.
- `read_experiment_record()`와 목록·검색·비교 세 함수의 signature는 바뀌지 않는다.
- 옛 index는 `rebuild_index`로 다시 만들면 `training`이 채워진다.
