# 020. embedding 학습도 되짚을 수 있게 기록해 주기를 바란다

## 왜 필요한가

`#193`에서 train에 `task="embedding"` 갈래가 생겼습니다. detector와 같은 대기열을
지나고 같은 artifact 네 개를 내지만, **무엇으로 학습했는지가 registry 요약에서
사라집니다.**

- `summary.py`의 `TRAINING_KEYS`에는 `task`도 `backbone`도 없습니다. 그래서 요약만
  보면 detector 실행과 embedding 실행이 구별되지 않습니다. `architecture`는 비어
  있고 `backbone`은 담기지 않으니, 어떤 model이었는지 말할 수 있는 값이 하나도
  없습니다.
- embedding의 입력은 manifest 넷이 아니라 **crop 은행 하나**(`crop_bank_uri`)입니다.
  `record.py`가 확인하는 data artifact 목록에 그 key가 없고, `config_snapshot`은
  `inputs`를 담지 않으므로, **어느 은행으로 학습했는지가 기록 어디에도 남지
  않습니다.**

이것이 왜 문제인가 하면, 이 model의 쓸모가 전부 "어느 참조 crop과 짝인가"에 걸려
있기 때문입니다. 재순위(`#194`)는 checkpoint와 은행을 함께 받아 거리를 재는데, 짝이
어긋나면 오류가 나지 않고 **점수만 조용히 나빠집니다.** 기록이 그 짝을 말해 주지
않으면, 나중에 "그때 그 embedding은 어느 은행 것이었나"를 되짚을 방법이 없습니다.

지금은 web의 job 기록이 그 짝을 들고 있습니다(`#195`). 그러나 그것은 **그 서버가 돈
학습만** 알고, 팀 기록으로 남는 것은 registry입니다.

## 요청 — registry에게

**1. `TRAINING_KEYS`에 `task`와 `backbone`을 더한다.** 둘 다 `text`입니다. 없던
실행에는 `None`이 들어가므로 기존 기록의 모양은 바뀌지 않습니다. 이 둘이 있어야
목록에서 detector 실행과 embedding 실행을 가릅니다.

**2. `crop_bank_uri`를 data artifact로 받아들인다.** 지금 확인하는 네 개는 그대로
두고, **있을 때만** 함께 검증하고 기록에 남기는 선택 key입니다. 없으면 지금과
완전히 같습니다. detector 실행에는 이 key가 오지 않습니다.

**3. 위 둘이 어렵다면 최소한 `config_snapshot`에 `inputs.data`를 담아 달라.** 그러면
요약에는 안 보여도 record를 열어 되짚을 수는 있습니다. 지금은 그 길조차 없습니다.

## 이 제안이 없으면 무엇이 남는가

`#193`은 이 제안 없이도 동작합니다. 잃는 것은 **되짚기**뿐입니다. 그래서 이 제안을
`#193`의 머지 조건으로 걸지 않았고, 대신 그 PR의 리뷰에서 나온 지적을 여기에
옮겨 적습니다.

당장은 web 화면이 학습을 걸 때 은행 자리를 자기 job 기록에 함께 적어 두는 것으로
버팁니다(`#195`의 `list_runs`). 서로 다른 은행으로 학습한 embedding을 함께 고르면
그 화면이 거절합니다. 그러나 그것은 **한 서버 안에서만** 참입니다.

## 누가 정하나

registry 소유자입니다. `src/pipelines/registry/`는 train이 건드리지 않습니다.
