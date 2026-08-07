"""Training GUI backend 실행 진입점.

    python -m src.pipelines.web.server

이 서버는 이 컴퓨터에서 학습 process를 실행합니다. 그래서 기본 bind 주소는 항상
``127.0.0.1``이며, 외부에 열려면 명시적으로 지정해야 합니다.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .api.app import create_app


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pill detection Training GUI backend")
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind 주소 (기본 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port (기본 8000)")
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="빌드된 frontend를 제공하지 않고 API만 띄웁니다",
    )
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(serve_frontend=not args.no_frontend)
    # 여기서 남은 기록을 미리 읽지 않습니다. 이미 다른 서버가 port를 잡고 있으면 이
    # process는 곧 죽는데, 그 전에 남의 학습 기록을 interrupted로 덮고 팀에도 그렇게
    # 알려 버립니다. 기록 정리는 첫 요청이 들어올 때 JobManager가 알아서 합니다.
    print(f"Training GUI backend: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
