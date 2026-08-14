# 016. Web의 image cache 준비 진행 표시

## 요청

Train은 기존 `train.progress/1` JSON Lines에 optional `image_cache_progress` event를
추가한다. Web은 `src/pipelines/web/docs/train_progress_contract.md`의 event 표에 이
event를 넣고, 원하면 화면에 준비 단계를 표시한다.

```json
{
  "schema": "train.progress/1",
  "event": "image_cache_progress",
  "run_id": "colab-20260814T093000000000Z",
  "ready": 4000,
  "total": 10495,
  "ts": "2026-08-14T09:35:12.300000Z"
}
```

| event | 추가 필드 |
| --- | --- |
| `image_cache_progress` | `ready`, `total` |

`ready`는 지금까지 **받아 둔** 이미지 수, `total`은 이 dataset의 train과 validation
이미지를 합친 수다. 실패한 이미지는 `ready`에 넣지 않으므로 마지막 줄이 늘 100%가
되지 않는다. 200장마다, 그리고 마지막에 한 번 나온다.

## 왜 필요한가

Train이 첫 batch 전에 이미지를 전부 미리 받도록 바뀌었다(PR #165). 10,495장이면
이 구간만 몇 분이 걸리고, 그동안 `run_started` 뒤로 아무 줄도 나오지 않는다. Colab
사용자는 화면이 멈춘 것으로 보고 실행을 죽인다. `docs/colab.md`의 "학습 시작이 오래
걸린다" 항목이 이 event를 보라고 안내한다.

## 지금도 깨지지 않는다

계약 문서는 모르는 `event`를 **원문 로그로 표시**하고 "event를 추가해도 schema
버전을 올릴 필요가 없다"고 적어 두었다. 그래서 이 event는 `train.progress/1` 그대로
이고, web을 고치지 않아도 로그 줄로 그냥 보인다. 이 제안은 **계약 문서의 표를
사실과 맞추는 것**이 목적이다.

## Web에 부탁드리는 것

1. `train_progress_contract.md`의 event 표에 위 한 줄을 추가한다.
2. `step_progress`처럼 **선택 event**로 적는다. 이 event가 없는 실행(취소된 옛 실행,
   이미지가 모두 로컬인 실행)도 지금 그대로 읽혀야 한다.
3. 화면 표시는 web의 판단이다. 필요 없다고 보면 표만 갱신해도 된다. 표시한다면
   전체 진행률(끝난 epoch 수)에는 넣지 말아 달라 — 학습 진행이 아니라 준비 단계다.

## 안 바뀌는 것

`schema` 값, 기존 event 5종의 이름과 필드, `run(config)` 반환값, checkpoint payload,
artifact 경로는 그대로다. Train은 stdout에 아무것도 쓰지 않는다.
