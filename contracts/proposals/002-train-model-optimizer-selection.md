# Train 모델·optimizer 선택 설정 제안

## 요청 배경

Train은 기존 `run(config)`와 artifact key를 유지하면서 다음 선택을 지원한다.

- `train.architecture`: `fasterrcnn_mobilenet_v3_large_320_fpn`, `fasterrcnn_resnet50_fpn_v2`, `retinanet_resnet50_fpn_v2`
- `train.optimizer`: `AdamW`, `SGD`, `Adam`

Train을 import하지 않는 Web 경계는 유지한다. Web 담당자가 지원 목록과 기본값을 복제하고 기존 AST contract test가 drift를 감시하도록 요청한다.

## Web 설정 요청

- 새 실험 form에 모델과 optimizer enum을 추가한다.
- 새 Web config는 모델 `fasterrcnn_mobilenet_v3_large_320_fpn`, optimizer `AdamW`를 명시적으로 저장한다.
- optimizer별 빈 수치 field의 기본값은 다음과 같다.
  - AdamW: learning rate `0.0001`, weight decay `0.01`, betas `0.9/0.999`, epsilon `1e-8`
  - Adam: learning rate `0.0001`, weight decay `0`, betas `0.9/0.999`, epsilon `1e-8`
  - SGD: learning rate `0.005`, momentum `0.9`, weight decay `0.0005`
- AdamW/Adam을 선택할 때 `momentum`을 보내거나 SGD를 선택할 때 `beta1`, `beta2`, `epsilon`을 보내면 Train이 학습 전에 거부한다. Web도 해당 field를 숨기고 runtime config에 보내지 않는다.
- 저장된 기존 config에 `optimizer`가 없으면 Train의 호환 규칙과 같이 SGD로 표시한다.
- 모델·optimizer 선택과 정규화된 수치가 `/api/train/validate`, 저장 config, review 화면, 실험 비교 기록에서 일치해야 한다.

## 검증 요청

- capability mirror의 기본값과 선택 목록을 Train source에서 AST로 읽어 비교한다.
- 모델이나 optimizer를 바꾸면 runtime config에 선택값이 보존되는지 검사한다.
- optimizer를 바꿀 때 해당 profile의 기본 수치가 적용되고 사용자가 입력한 수치는 유지되는지 검사한다.
- 기존 optimizer 없는 config가 SGD로 해석되는 회귀 test를 유지한다.

이 제안은 Web 파일을 직접 변경하지 않으며, Web 담당자가 별도 단일 목적 PR로 구현한다.
