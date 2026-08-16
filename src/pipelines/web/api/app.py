"""FastAPI application factory."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..errors import (
    JobConflictError,
    JobNotFoundError,
    TeamSyncAuthError,
    TeamSyncError,
    WebError,
    WebPathError,
    WebValidationError,
    error_payload,
)
from . import (
    routes_data,
    routes_ensemble,
    routes_gpu,
    routes_meta,
    routes_settings,
    routes_team,
    routes_train,
)


__all__ = ["ALLOWED_ORIGINS", "create_app"]


# 이 서버는 host에서 학습을 실행하므로 wildcard(``*``)를 쓰지 않습니다. 대신 이 컴퓨터
# 안의 origin만 허용합니다.
#
# port를 고정하지 않는 이유: Vite가 빌드한 index.html의 module script에는 ``crossorigin``
# 속성이 붙어 있어서, 같은 origin에서 받아 갈 때도 브라우저가 CORS 모드로 요청합니다.
# 서버 자신의 origin이 허용 목록에 없으면 응답에 Access-Control-Allow-Origin이 빠지고,
# script가 실행되지 않아 화면이 빈 채로 뜹니다. 서버 port는 --port로 바뀔 수 있으므로
# localhost의 모든 port를 regex로 받습니다. 원격 page의 origin은 절대 localhost가 될 수
# 없으므로 이렇게 해도 외부에 열리지 않습니다.
ALLOWED_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

# Vite dev server의 기본 주소. 문서와 test에서 대표 값으로 씁니다.
ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

# Windows는 file 확장자별 MIME type을 registry에서 읽습니다. 그 값이 잘못 설정된
# 컴퓨터에서는 .js가 text/plain으로 나가고, 브라우저는 그런 MIME type의 ES module을
# 실행하지 않아 화면이 빈 채로 뜹니다. 환경에 기대지 않도록 여기서 못 박습니다.
_STATIC_MIME_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}


def register_static_mime_types() -> None:
    for suffix, mime_type in _STATIC_MIME_TYPES.items():
        mimetypes.add_type(mime_type, suffix)


class SpaStaticFiles(StaticFiles):
    """빌드된 frontend를 제공하되, 없는 경로는 index.html로 돌려줍니다.

    화면 전환은 브라우저 주소를 ``/monitor/<job_id>`` 처럼 바꿉니다. 그 주소에 해당하는
    파일은 없으므로, fallback이 없으면 새로고침이나 링크 공유가 404가 됩니다. 라이브
    모니터는 새로고침해도 살아남아야 하는 화면이라 이 fallback이 필요합니다.
    """

    async def get_response(self, path: str, scope):  # type: ignore[no-untyped-def]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as error:
            # API 경로는 fallback하지 않습니다. 없는 API는 404여야 합니다.
            if error.status_code == 404 and not path.startswith("api"):
                return await super().get_response("index.html", scope)
            raise


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(WebValidationError)
    def _validation(_: Request, error: WebValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content=error_payload(error))

    @app.exception_handler(WebPathError)
    def _path(_: Request, error: WebPathError) -> JSONResponse:
        return JSONResponse(status_code=400, content=error_payload(error))

    @app.exception_handler(JobNotFoundError)
    def _not_found(_: Request, error: JobNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content=error_payload(error))

    @app.exception_handler(JobConflictError)
    def _conflict(_: Request, error: JobConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content=error_payload(error))

    @app.exception_handler(TeamSyncAuthError)
    def _team_auth(_: Request, error: TeamSyncAuthError) -> JSONResponse:
        return JSONResponse(status_code=401, content=error_payload(error))

    @app.exception_handler(TeamSyncError)
    def _team_sync(_: Request, error: TeamSyncError) -> JSONResponse:
        return JSONResponse(status_code=503, content=error_payload(error))

    @app.exception_handler(WebError)
    def _generic(_: Request, error: WebError) -> JSONResponse:
        return JSONResponse(status_code=500, content=error_payload(error))


def create_app(*, serve_frontend: bool = True) -> FastAPI:
    """Training GUI backend를 만듭니다. import만으로는 학습이 시작되지 않습니다."""

    app = FastAPI(
        title="Pill Detection Training GUI",
        version=routes_meta.API_VERSION,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=ALLOWED_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    _register_error_handlers(app)
    app.include_router(routes_meta.router)
    app.include_router(routes_train.router)
    app.include_router(routes_data.router)
    app.include_router(routes_gpu.router)
    app.include_router(routes_settings.router)
    app.include_router(routes_team.router)
    app.include_router(routes_ensemble.router)

    if serve_frontend and _FRONTEND_DIST.is_dir():
        # 빌드된 frontend가 있으면 같은 origin에서 함께 제공합니다.
        register_static_mime_types()
        app.mount("/", SpaStaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
    return app
