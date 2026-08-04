"""Configured S3 storage에 대한 비파괴 smoke test."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common import S3Storage, StorageError, create_storage, load_config


SMOKE_PREFIX = "experiments/uploading/smoke-tests/"
ALIAS_VALUE = "!python scripts/s3_smoke_test.py"


def _delete_object(storage: S3Storage, object_key: str) -> None:
    """업로드한 smoke-test object 하나만 삭제합니다."""
    storage.client.delete_object(Bucket=storage.bucket, Key=object_key)


def run_smoke_test(config_path: str | Path, *, keep: bool = False) -> dict:
    """작은 임시 JSON object의 S3 왕복과 prefix listing을 확인하고 정리합니다."""

    storage = create_storage(load_config(config_path))
    if not isinstance(storage, S3Storage):
        raise StorageError("smoke test에는 s3 storage backend 설정이 필요합니다.")

    object_key = f"{SMOKE_PREFIX}{uuid.uuid4().hex}.json"
    payload = {
        "purpose": "pill-object-detection-s3-smoke-test",
        "object_key": object_key,
    }
    object_uri: str | None = None
    listed_objects: list[str] = []

    try:
        with tempfile.TemporaryDirectory(prefix="pill-s3-smoke-") as temporary_directory:
            temporary_path = Path(temporary_directory)
            upload_path = temporary_path / "upload.json"
            download_path = temporary_path / "download.json"
            upload_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

            object_uri = storage.upload_file(upload_path, object_key)
            storage.download_file(object_uri, download_path)
            downloaded_payload = json.loads(download_path.read_text(encoding="utf-8"))
            if downloaded_payload != payload:
                raise StorageError("업로드한 JSON과 다운로드한 JSON이 다릅니다.")

            listed_objects = storage.list(SMOKE_PREFIX)
            if object_uri not in listed_objects:
                raise StorageError("업로드한 object가 smoke-test prefix listing에 없습니다.")
    finally:
        cleanup = _cleanup(storage, object_uri, keep=keep)

    return {
        "status": "ok" if cleanup["deleted"] or keep else "warning",
        "object_uri": object_uri,
        "listed_count": len(listed_objects),
        "message": "S3 upload, download, content 비교, prefix listing 확인 완료",
        "cleanup": cleanup,
    }


def _cleanup(storage: S3Storage, object_uri: str | None, *, keep: bool) -> dict:
    """임시 object를 삭제하고 결과를 알려줍니다. 실패해도 예외를 올리지 않습니다."""
    if object_uri is None:
        return {"deleted": False, "detail": "업로드 전에 중단되어 삭제할 object가 없습니다."}
    if keep:
        return {"deleted": False, "detail": "--keep 옵션으로 임시 object를 남겨두었습니다."}

    uploaded_key = unquote(urlsplit(object_uri).path.lstrip("/"))
    try:
        _delete_object(storage, uploaded_key)
        if storage.exists(object_uri):
            return {
                "deleted": False,
                "detail": "삭제 요청 후에도 object가 남아 있습니다. 직접 삭제하세요.",
            }
    except Exception as error:  # 정리 실패가 검증 결과를 가리지 않게 합니다.
        return {
            "deleted": False,
            "detail": f"임시 object 삭제 실패: {type(error).__name__}: {error}. 직접 삭제하세요.",
        }
    return {"deleted": True, "detail": "임시 object를 삭제했습니다."}


def install_alias() -> None:
    """이 저장소에 `git s3-smoke` alias를 설치합니다."""
    if shutil.which("git") is None:
        raise RuntimeError("Git을 찾을 수 없습니다.")
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "--local", "alias.s3-smoke", ALIAS_VALUE],
        check=True,
    )
    print("설치 완료: 이 저장소에서 'git s3-smoke'를 사용할 수 있습니다.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Amazon S3 storage smoke test")
    parser.add_argument("--config", help="AWS storage JSON config 경로")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="임시 object를 삭제하지 않고 남겨 둡니다.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="이 저장소에 git s3-smoke alias를 설치합니다.",
    )
    args = parser.parse_args(argv)

    if args.install:
        try:
            install_alias()
        except (RuntimeError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", None) or error
            print(f"alias 설치 실패: {str(detail).strip()}", file=sys.stderr)
            return 1
        return 0

    if not args.config:
        parser.error("--config가 필요합니다. 예: --config configs/env.aws.json")

    try:
        result = run_smoke_test(args.config, keep=args.keep)
    except (StorageError, OSError, ValueError) as error:
        print(f"S3 smoke test 실패: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
