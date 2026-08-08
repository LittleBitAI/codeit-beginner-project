# Web 이어서 학습 반영 제안

## 왜 필요한가

`009-train-resume.md`가 요청한 이어서 학습을 train이 구현했습니다. 그러면서 설정
두 개가 늘었고, 그중 `train.checkpoint_every`는 `_integer` 기본값이라
`src/pipelines/web/tests/test_web_train_contract.py`의
`test_numeric_defaults_match_train_source`가 **지금 실패합니다.**

이건 고장이 아니라 경보가 울린 것입니다. train은 web 파일을 고치지 않는 것이
규칙이라(`src/pipelines/train/CLAUDE.md`) 여기에 적어 요청합니다. 복제본을 맞추기
전까지 CI 전체가 빨간 상태로 남습니다.

## 요청

**1. `train_config.py`에 `checkpoint_every`를 더한다.** 기본값 `1`, 정수, 최소 1.
`_INTEGER_FIELDS`에 한 줄 넣으면 됩니다. 이것 하나로 실패하는 test가 초록으로
돌아옵니다.

**2. `resume_from` 검증 규칙을 복제한다.** optional 문자열입니다. 비어 있으면 거부
(`train.resume_from must be a non-empty checkpoint path`). train은 repository 안의
경로만 받고 `s3://`도 받습니다.

**3. 이어서 학습 화면.** 중단된 job은 이미 `interrupted`로 구분해 두었으니
(`jobs/manager.py`) 그 자리에 붙일 수 있습니다. 화면이 채워야 할 값은 셋입니다.

| 값 | 무엇 |
| --- | --- |
| `run_id` | **새 이름.** 이어서 하는 실행은 별도 실행입니다 |
| `resume_from` | 중단된 실행의 `last_checkpoint.pt` 경로 |
| `epochs` | 남은 수가 아니라 **전체 목표**. 40에서 끊긴 50 계획이면 그대로 50 |

## 이어서 할 파일을 어디서 찾나

train은 학습 중 checkpoint를 정해진 이름의 작업 폴더에 남깁니다. web은 이미
`output_dir`과 `run_id`를 가지고 있으므로 새 progress event 없이 경로를 만들 수
있습니다.

```
<output_dir>/.<run_id>.partial/last_checkpoint.pt     # 로컬
<output_prefix>/<run_id>/running/last_checkpoint.pt   # S3 backend일 때 함께 올라감
```

옆의 `best_checkpoint.pt`도 함께 있어야 합니다. train이 그 파일에서 이어붙이기
이전의 best 가중치를 읽습니다.

## train이 거부하는 것

화면에서 미리 걸러 주면 사람이 subprocess까지 가지 않고 압니다. 전부 학습 시작
**전에** 거부됩니다.

- `epochs`가 이어붙일 epoch보다 크지 않다
- checkpoint에 `resume_state`가 없다 (이 기능 이전에 만들어진 파일)
- architecture나 class map이 지금 설정과 다르다
- 옆에 `best_checkpoint.pt`가 없다
- 조기 종료 patience를 이미 다 쓴 checkpoint다 (한 epoch만 돌고 다시 멈춥니다)

## 하나 더: run_id 충돌 검사

`check_run_id_collision`은 지금 `<output_dir>/<run_id>`만 봅니다. train은
`<output_dir>/.<run_id>.partial`이 비어 있지 않아도 거부하므로, 같은 자리에서 함께
확인해 주면 화면이 train과 같은 답을 줍니다.

## 이번 제안이 정하지 않는 것

- 고아가 된 `.<run_id>.partial` 폴더를 **누가 지우는가.** train은 지우지 않습니다.
  중단된 실행의 유일한 사본이기 때문입니다. 화면에 "버리기"를 두든 기간이 지난
  것을 정리하든, 정하는 것은 web과 팀입니다. 정하지 않으면 실행마다 checkpoint 두
  개가 계속 쌓입니다.
- 중단된 학습을 **자동으로** 이어서 시작하는 것. 사람이 화면에서 고릅니다.
