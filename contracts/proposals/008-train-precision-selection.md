# Train 정밀도 선택 제안

## 요청

Train은 optional `train.precision` 문자열로 `fp32`와 `amp`를 받는다. key가 없으면
기존과 같은 `fp32`다. `amp`는 CUDA 전용이며 native bf16을 지원하면 bf16, 아니면
fp16을 선택한다. fp16은 GradScaler를 사용한다.

Web은 새 실험 화면에 `fp32`와 `amp` 선택지를 추가하고, 이전 실행과 선택하지 않은
요청에는 `fp32`를 사용한다. Web은 train 내부를 import하지 않고 기존 source 기반
계약 test로 선택지와 기본값 drift를 감시한다.

## 기록과 호환성

- checkpoint `training_config.precision`은 `mode`, 실제 `dtype`, `grad_scaler` 여부를
  기록한다.
- Train summary의 `precision`은 실제 `fp32`, `bf16`, `fp16` 중 하나다.
- `training_history.json`, checkpoint epoch와 model/optimizer state key는 바뀌지
  않는다.
- 기존 checkpoint에는 precision metadata가 없을 수 있으며 evaluate는 이를 요구하지
  않는다.
- 이번 Train 변경은 `src/pipelines/web/`을 수정하지 않는다.
