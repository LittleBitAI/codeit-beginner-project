# 001. 대회용 test manifest와 submission artifact 계약

## 상태와 목적

합의됨. 대회 test image가 학습·검증에 섞이지 않게 별도 manifest로 전달하고,
Evaluate가 정답 label 없이 제출 CSV를 만들 수 있도록 선택 artifact 계약을 추가한다.
기존 학습·검증 실행은 그대로 동작해야 한다.

## Data 계약

`data.prepare=true`인 준비 경로는 test 입력이 있을 때 기존 네 artifact
`train_manifest_uri`, `validation_manifest_uri`, `class_map_uri`,
`dataset_summary_uri`에 선택 key `test_manifest_uri`를 추가한다. 기존 URI를 검증해
그대로 돌려주는 legacy pass-through 경로는 계속 기존 네 key만 반환해도 된다.

`test_manifest_uri`는 COCO JSON 문서를 가리키며 다음을 지킨다.

- `images[].id`는 image filename의 확장자를 뺀 stem을 정수로 변환한
  `image_id`이다. 각 image에는 실제 파일에서 읽은 `width`, `height`와
  downstream이 읽을 수 있는 S3 URI 또는 상대 경로를 `file_name`에 기록한다.
- `categories`는 원본 category ID를 보존하고 ID 오름차순으로 정렬한다. 각
  category의 `supercategory`는 `"pill"`이다.
- 정답이 없는 test set이므로 최상위 `annotations`는 빈 배열이다.
- image와 category 순서를 포함한 문서 출력은 같은 입력에 대해 항상 같다.
- test image가 0개이거나 stem이 숫자가 아니거나 `image_id`가 중복되면 실패한다.
  읽을 수 없는 image와 중복 filename도 실패한다.
- test image는 어떤 경우에도 train/validation split에 들어가지 않는다.

기존 artifact를 포함한 모든 출력은 `overwrite=false`가 기본이며, 이때 이미 있는
파일을 덮어쓰지 않는다.

## Evaluate 계약

Evaluate는 `test_manifest_uri`를 선택 입력으로 받을 수 있다. 입력이 있으면 test
image를 추론해 선택 artifact `submission_uri`를 반환한다. 기존 validation 평가
계약은 유지하며, test manifest에는 정답이 없으므로 test mAP을 계산하거나
보고하지 않는다.

대회 validation mAP의 IoU threshold는 정확히
`[0.75, 0.80, 0.85, 0.90, 0.95]`를 사용한다. 이 목록을 임의로 늘리거나 줄이지
않으며 test 결과에는 적용하지 않는다.

`submission_uri`의 CSV 계약은 다음과 같다.

- header는 정확히
  `annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score`이다.
- 검출 object 하나가 한 행이며 bbox 좌표 형식은 `xywh`이다.
- `image_id`는 manifest의 정수 ID, 즉 숫자 filename stem에서 온 값을 쓴다.
- score threshold 기본값은 `0`이고 image마다 score가 높은 검출을 최대 4개까지
  남긴다.
- 행은 `image_id` 오름차순, 같은 image 안에서는 `score` 내림차순이다. score가
  같으면 `category_id` 오름차순, 이어서 `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`
  오름차순을 안정적인 tie-breaker로 사용한다. 이 최종 순서에 따라
  `annotation_id`를 `1..N`으로 부여하므로 같은 입력의 CSV는 항상 같다.
- bbox 좌표와 score 숫자는 반올림하지 않고 원래 정밀도로 기록한다.
- threshold와 image별 상한을 적용한 뒤 검출이 0개여도 정확한 header만 있는 CSV는
  유효한 submission artifact이다.

## Registry 계약

Registry는 Data의 `test_manifest_uri`와 Evaluate의 `submission_uri`를 선택
artifact로 기록한다. 둘이 없어도 기존 실행은 유효하며, 새 record의
`schema_version`은 `"1.1"`이다. 선택 URI에도 기존 URI 안전성, 존재 여부,
checksum 검증 정책을 동일하게 적용한다.

## 영향 범위와 구현 소유권

Train과 Web의 public interface는 바꾸지 않는다. 이 proposal은 계약만 기록하며
pipeline 구현이나 shared document를 수정하지 않는다. Data, Evaluate, Registry
구현은 각 pipeline owner가 자기 branch와 단일 목적 PR에서 독립적으로 맡는다.
문서와 각 구현은 서로 독립적으로 merge할 수 있지만, 구현은 이 계약을 따라야
하며 다른 pipeline의 내부 코드를 수정하지 않는다.

검증 실패는 기존 `run(config)` 경계대로 `status="error"`로 반환한다. 각 owner는
실패한 실행이 새로 만든 부분 artifact만 자기 pipeline의 기존 cleanup 정책에 따라
정리하고, 실행 전에 존재한 artifact는 수정하거나 삭제하지 않는다.
