"""Training GUI backend 실행 진입점.

    python -m src.pipelines.web.server

이 서버는 이 컴퓨터에서 학습 process를 실행합니다. 그래서 기본 bind 주소는 항상
``127.0.0.1``이며, 외부에 열려면 명시적으로 지정해야 합니다.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .api.app import create_app
from .jobs import get_manager


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
    get_manager().load()
    print(f"Training GUI backend: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
