# 005. AI Hub 데이터로 학습 데이터 확장과 class map 확장

## 상태와 목적

제안. 대회가 준 train 데이터는 **232장뿐**이다. AI Hub "약품식별 인공지능 개발을
위한 경구약제 이미지 데이터"의 `1.Training`을 같은 형식으로 바꿔 **10,553장**(대회
232장 + AI Hub 10,321장)으로 늘리자는 제안이다. 그러려면 class map이 **56개에서
118개로** 늘어나므로 train과 evaluate의 합의가 필요하다.

## 두 데이터는 같은 출처다

대회 데이터는 AI Hub 데이터에 **식별자만 다시 매긴 것**이다. 실측으로 확인했다.

| | 대회 `raw/v1/original/` | AI Hub `1.Training` |
| --- | --- | --- |
| 디렉터리 | `train_annotations/<조합>_json/<알약코드>/<이미지>.json` | 같음 |
| JSON 최상위 | `images`/`type`/`annotations`/`categories` | 같음 |
| 해상도 | 976×1280 | 976×1280 (40,413개 전부) |
| 인코딩 | UTF-8, BOM 없음 | 같음 |
| 이미지 | train 232장, test 842장 | 10,509장 |
| `image_id`/`category_id`/`annotation_id` | 다시 매겨진 값 | **전부 `1`** |
| `categories` | `{id: <알약코드>, name: <제품명>}` | `{id: 1, name: "Drug"}` |

## 요청하는 변경: class map 확장

`class_map.json`은 data pipeline이 데이터에서 만들어 내므로 **코드에 56이 박혀 있는
곳은 없다**. train은 `dataset.py`에서 category id를 정렬해 `1..N` label로 바꾸고,
model head 크기도 그 개수에서 나온다. 그래서 확장 자체는 코드 수정 없이 된다.

영향은 두 가지다.

1. **기존 checkpoint를 쓸 수 없다.** head가 57개(56+배경)에서 119개로 커지고,
   category id 정렬 순서가 바뀌어 기존 label 번호와 어긋난다. 새로 학습해야 한다.
2. **학습 시간이 늘어난다.** 이미지가 45배다.

evaluate는 manifest와 class map의 category id를 그대로 쓰므로 형식 변경이 없다.

## 두 가지 label 공간

AI Hub는 대회 56개 class 중 **54개를 덮고**, 새 class **62개를 더한다. 없는 것은
`K-003351`(일양하이트린정 2mg), `K-003483`(기넥신에프정) 둘뿐이다.

변환기의 `--label-space`가 둘 중 하나를 고른다. 어느 쪽이든 **AI Hub 이미지 10,264장을
전부 쓰고 annotation 39,488개가 남는다**. 차이는 class 수뿐이다.

| | `full` | `competition-plus-other` |
| --- | --- | --- |
| AI Hub만 변환했을 때 class | 116 | 55 (대회 54 + `기타 알약`) |
| **대회 232장과 합친 prefix의 class** | **118** | **57** |
| 대회 class annotation | 17,847 | 17,847 |
| 나머지 annotation | 62개 class로 21,863 | `기타 알약` 하나로 21,863 |
| 못 본 test class 대응 | 가능 | 불가 |

`full`이 116이 아니라 118이 되는 이유는, 대회에만 있는 `K-003351`·`K-003483` 두
class가 대회 232장에서 더해지기 때문이다(56 + 116 − 공유 54 = 118).

`competition-plus-other`는 대회 밖 알약을 `category_id` `999999`(`기타 알약`) 하나로
합친다. **bbox를 지우는 선택지는 없다.** 한 사진에 알약이 3~4개라서, 대회 밖 알약의
bbox만 빼면 그 알약이 라벨 없이 남아 "알약은 배경"으로 학습된다. 모든 알약이 대회
class인 이미지는 214장뿐이라 그 길로는 아무것도 얻지 못한다.

## 어느 쪽이든 데이터 확장은 성립한다

대회 안내는 test에 train에 없는 class가 있다고 밝히고 해법으로 "데이터 보완"을 권한다.
제출 명세의 `category_id`에도 허용 목록이 없어, 채점기가 대회 56개 밖을 거부할 근거는
보이지 않는다. 그래도 거부하는 경우를 대비해 두 갈래를 준비했다.

| 채점기가 새 class를 받는가 | 선택 |
| --- | --- |
| 받는다 | `full`. 못 본 test class까지 노린다 |
| 받지 않는다 | `full`로 학습하고 제출 때 대회 밖 행을 버리거나, `competition-plus-other`로 학습한다 |

두 경우 모두 **대회 54개 class의 annotation이 763개에서 17,847개로 약 23배** 늘어나는
이득은 그대로다. 그 이득에는 label 공간 변경이 필요 없다.

`src/pipelines/evaluate/submission.py`는 test manifest의 `categories`에 없는
`category_id`를 거부한다. 그 목록은 `class_map.json`에서 나오므로, class map을 넓히면
제출도 함께 넓어진다. 파이프라인 안쪽에는 막는 곳이 없다.

덧붙여 대회 안내는 "Test Data에는 Train Data엔 없는 클래스가 존재한다"고 밝혔다.
새 62개 class는 그 빈칸을 메울 수 있다.

## 유출 검사

`CLAUDE.md`가 되돌릴 수 없는 실수로 꼽는 항목이라 먼저 확인했다.

- test 이미지 842장 중 AI Hub 이미지와 크기가 같은 28장을 내려받아 **객체 본문 MD5**를
  대조 → 일치 **0건**. 이 bucket은 SSE-KMS라 **ETag가 MD5가 아니므로** 본문을 해시해야
  한다. 바이트가 같은 사본만 잡으므로 재인코딩본은 걸리지 않는다.
- 대회 train 조합 114개와 AI Hub 조합 3,487개의 교집합 **0**.

**경고**: 대회 데이터는 AI Hub의 Validation 쪽에서 온 것으로 보인다(조합이 전혀 겹치지
않는다). **AI Hub Validation 세트를 추가로 받으면 test 유출 위험이 실재한다.** 받지
않기를 제안한다.

## 변환 규칙

`scripts/aihub_to_competition.py`가 세 식별자를 다시 매긴다.

- `image_id` — `file_name` 순서로 고유 번호. 같은 이미지의 알약 문서는 같은 값을
  쓴다(`coco.py`가 이를 강제한다). 기본 offset 100,000으로 대회 값(14~1499)과
  겹치지 않는다.
- `category_id` — **`dl_idx`가 아니라 `dl_mapping_code`의 숫자부**. AI Hub의
  `dl_idx`는 그 값보다 1 작고(`K-001900` → `1899`), `K-053384`는 `130376`으로 아예
  다르다. 그대로 쓰면 116개 class 전부가 대회 label 공간과 어긋난다. `categories`의
  `name`도 `"Drug"`이므로 `dl_name`으로 바꾼다.
- `annotation_id` — bbox가 있고 길이가 4인 annotation에만 순번. 기본 offset
  1,000,000.

## 원본 오류와 제외 기준

대회 안내대로 원본에 오류가 있다. 알약 하나가 라벨을 잃은 이미지는 그 알약을 배경으로
가르치므로 이미지를 통째로 제외한다.

| 제외 이유 | 이미지 |
| --- | --- |
| bbox 없음 또는 형식 오류 | 130 |
| 조합 알약 수와 문서 수 불일치 | 27 |
| 같은 알약 문서 중복 | 18 |
| 라벨 json 없음 | 12 |
| `iscrowd`가 0/1이 아님 | 1 |

건너뜬 문서: macOS `._` junk 1개, 파싱 실패 1개, 없는 이미지를 가리키는 문서 24개.

## 검증 결과

두 label 공간의 변환 결과를 각각 프로젝트의 `consolidate()`와 `split_images()`에
그대로 넣었다. 둘 다 통과했고 이미지·annotation 수가 같다.

- `consolidate()` — AI Hub 이미지 10,264장, annotation 39,488개. class는 `full`
  116개, `competition-plus-other` 55개(대회 232장을 합치면 118·57). 57장은 bbox가
  이미지 밖이라 pipeline이 제외했다.
- `split_images()` 8:2, seed 42, group — train 8,211 / validation 2,053 (비율
  0.200), 그룹 3,487개, **그룹 누수 0**, **train에만 남은 category 0**. 두 공간 모두
  모든 class가 양쪽 split에 들어간다.
- 표본 40장의 실제 PNG 크기가 JSON과 일치(전부 976×1280).

## 결정이 필요한 것

1. `full`(118 class)로 갈 것인가 `competition-plus-other`(57 class)로 갈 것인가.
   train·evaluate의 합의가 필요하다. 어느 쪽이든 기존 checkpoint는 못 쓴다.
2. 새 prefix `datasets/pill_detection/raw/v2/original/`에 AI Hub 이미지 17.39 GiB를
   올린다. 대회 232장·763 annotation·test 842장은 `raw/v1/`에서 **S3 내부 복사**로
   가져와 한 prefix에 모은다. `raw/v1/`은 건드리지 않는다. S3 비용이 발생한다.
   `_index.png` 3,503개(6.1 GiB)는 실사진이 아니고 참조하는 라벨도 없어 제외한다.
3. 채점기가 대회 밖 `category_id`를 받는지 확인. 결과는 1번의 선택만 바꾸고, 데이터
   확장 여부는 바꾸지 않는다.
