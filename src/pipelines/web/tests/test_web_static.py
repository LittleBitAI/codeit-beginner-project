"""빌드된 frontend를 제공하는 방식.

여기 있는 두 가지는 실제로 화면이 하얗게 뜨거나 새로고침이 404가 나서 발견한 문제라,
회귀를 막기 위해 test로 고정합니다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.pipelines.web.api import app as app_module
from src.pipelines.web.api.app import create_app


@pytest.fixture
def built_frontend(tmp_path, monkeypatch):
    """빌드 산출물이 있는 것처럼 흉내 냅니다. 실제 npm build에 의존하지 않습니다."""

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><html><body><div id="root"></div>'
        '<script type="module" crossorigin src="/assets/app.js"></script></body></html>',
        encoding="utf-8",
        newline="\n",
    )
    (dist / "assets" / "app.js").write_text("export const ok = 1;\n", encoding="utf-8", newline="\n")
    (dist / "assets" / "app.css").write_text(":root{}\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(app_module, "_FRONTEND_DIST", dist)
    return dist


@pytest.fixture
def static_client(built_frontend, manager):
    with TestClient(create_app(serve_frontend=True)) as client:
        yield client


def test_index_is_served_at_root(static_client):
    response = static_client.get("/")

    assert response.status_code == 200
    assert "<div id=\"root\">" in response.text


def test_javascript_is_served_with_a_javascript_mime_type(static_client):
    """Windows는 확장자별 MIME type을 registry에서 읽습니다.

    그 값이 잘못된 컴퓨터에서는 .js가 text/plain으로 나가고, 브라우저는 그런 MIME
    type의 ES module을 실행하지 않아 화면이 빈 채로 뜹니다.
    """

    response = static_client.get("/assets/app.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")


def test_css_is_served_with_a_css_mime_type(static_client):
    response = static_client.get("/assets/app.css")

    assert response.headers["content-type"].startswith("text/css")


@pytest.mark.parametrize(
    "path",
    (
        "/records",
        "/board",
        "/canvas",
        "/monitor",
        "/monitor/" + "a" * 32,
        "/monitor/anything",
    ),
)
def test_deep_links_fall_back_to_index(static_client, path):
    """화면 주소를 직접 열거나 새로고침해도 404가 나면 안 됩니다.

    라이브 모니터는 주소가 /monitor/<job_id>라서 새로고침에도 살아남아야 합니다.
    """

    response = static_client.get(path)

    assert response.status_code == 200
    assert "<div id=\"root\">" in response.text


@pytest.mark.parametrize("path", ("/api/nope", "/api/train/nope", "/api/train/jobs/nope/nope"))
def test_unknown_api_paths_still_return_404(static_client, path):
    """없는 API가 index.html을 돌려주면 client가 오류를 알아채지 못합니다."""

    response = static_client.get(path)

    assert response.status_code == 404
    assert "<div id=\"root\">" not in response.text


def test_api_still_works_when_frontend_is_mounted(static_client):
    assert static_client.get("/api/health").json()["status"] == "ok"


def test_missing_asset_under_assets_falls_back_but_stays_html(static_client):
    # 없는 asset은 index.html로 떨어집니다. Vite가 만든 파일 이름은 hash가 붙어 있어
    # 실제로는 발생하지 않지만, 여기서 500이 나지 않는다는 것만 확인합니다.
    response = static_client.get("/assets/missing.js")

    assert response.status_code == 200


def test_app_without_frontend_serves_api_only(manager):
    with TestClient(create_app(serve_frontend=False)) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 404
