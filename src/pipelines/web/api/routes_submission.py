"""만들어 둔 제출 CSV를 화면에서 그대로 내려받는 route.

지금까지 화면은 만들어진 파일의 **경로만** 알려 줬습니다. Kaggle에 올리려면 그 경로를
읽고 탐색기에서 찾아 들어가야 했는데, 저장소를 clone만 해 본 사람에게는 그 자리가
어디인지가 이미 한 단계 문제입니다.

**저장소 안에 만들어진 파일만** 보냅니다. S3에 올라간 결과는 여기서 다루지 않습니다 —
그쪽은 bucket을 볼 수 있는 사람이 이미 다른 길로 받습니다. 경로 검증은
``paths.resolve_within_repo``가 하므로 traversal, 절대 경로, drive 문자는 여기 오기
전에 막힙니다.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ..errors import JobNotFoundError
from ..paths import resolve_within_repo


router = APIRouter(prefix="/api", tags=["submission"])

#: evaluate가 쓰는 제출 file 이름입니다. 이 이름이 아니면 보내지 않습니다 — 이 route로
#: 저장소 안 아무 파일이나 읽어 가지 못하게 하는 자리입니다.
SUBMISSION_FILE_NAME = "submission.csv"


@router.get("/submission")
def download(uri: str = Query(..., min_length=1, max_length=512)) -> FileResponse:
    """제출 CSV 하나를 내려받습니다. 실행 이름을 파일 이름에 붙여 줍니다."""

    path = resolve_within_repo(uri, label="제출 파일")
    if path.name != SUBMISSION_FILE_NAME or not path.is_file():
        raise JobNotFoundError("그 자리에 제출 CSV가 없습니다.")
    return FileResponse(
        path,
        media_type="text/csv",
        # 여러 실행의 제출을 받으면 전부 `submission.csv`가 되어 어느 것인지 모릅니다.
        filename=f"{path.parent.name}-{SUBMISSION_FILE_NAME}",
    )
