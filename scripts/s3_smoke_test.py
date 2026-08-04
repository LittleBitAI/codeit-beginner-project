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
_ABSENT_CODES = {"NoSuchKey", "NoSuchVersion", "NotFound", "404"}


class SmokeTestCleanupError(StorageError):
    """검증과 임시 object 정리가 모두 실패한 경우입니다.

    원래 검증 오류, 정리 실패 내용, 남아 있는 object URI를 함께 전달합니다.
    """

    def __init__(
        self,
        message: str,
        *,
        original_error: BaseException,
        cleanup_detail: str,
        object_uri: str,
    ) -> None:
        super().__init__(message)
        self.original_error = original_error
        self.cleanup_detail = cleanup_detail
        self.object_uri = object_uri


def _error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return ""
    return str(response.get("Error", {}).get("Code", ""))


def _version_id(storage: S3Storage, object_key: str) -> str | None:
    """방금 업로드한 object의 VersionId를 확인합니다.

    버전 관리를 켜지 않은 bucket은 VersionId를 주지 않거나 'null'을 주며,
    그 경우 None을 돌려 버전 없이 삭제하게 합니다.
    """
    response = storage.client.head_object(Bucket=storage.bucket, Key=object_key)
    version_id = (response or {}).get("VersionId")
    if not version_id or version_id == "null":
        return None
    return str(version_id)


def _delete_object(
    storage: S3Storage, object_key: str, version_id: str | None = None
) -> None:
    """업로드한 smoke-test object 하나만 삭제합니다.

    VersionId를 알면 그 version을 정확히 지웁니다. 버전 관리 bucket에서 version
    없이 삭제하면 delete marker만 생기고 실제 data는 남기 때문입니다.
    """
    request = {"Bucket": storage.bucket, "Key": object_key}
    if version_id is not None:
        request["VersionId"] = version_id
    storage.client.delete_object(**request)


def _object_remains(
    storage: S3Storage, object_key: str, version_id: str | None
) -> bool:
    """삭제 대상이 아직 남아 있는지 확인합니다.

    version 관리 bucket에서는 delete marker 때문에 key 단위 조회가 404가 되므로,
    VersionId를 알면 반드시 그 version을 직접 조회합니다.
    """
    request = {"Bucket": storage.bucket, "Key": object_key}
    if version_id is not None:
        request["VersionId"] = version_id
    try:
        storage.client.head_object(**request)
    except Exception as error:  # botocore ClientError 및 backend별 예외
        if _error_code(error) in _ABSENT_CODES:
            return False
        raise
    return True


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
    version_id: str | None = None
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
            version_id = _version_id(storage, object_key)
            storage.download_file(object_uri, download_path)
            downloaded_payload = json.loads(download_path.read_text(encoding="utf-8"))
            if downloaded_payload != payload:
                raise StorageError("업로드한 JSON과 다운로드한 JSON이 다릅니다.")

            listed_objects = storage.list(SMOKE_PREFIX)
            if object_uri not in listed_objects:
                raise StorageError("업로드한 object가 smoke-test prefix listing에 없습니다.")
    except Exception as error:
        cleanup = _cleanup(storage, object_uri, version_id, keep=keep)
        if cleanup["deleted"] or object_uri is None or keep:
            raise
        raise SmokeTestCleanupError(
            "S3 smoke test 검증과 임시 object 정리가 모두 실패했습니다."
            f" | 검증 오류: {error}"
            f" | 정리 실패: {cleanup['detail']}"
            f" | 남은 object: {object_uri}"
            + (f" (VersionId: {version_id})" if version_id else ""),
            original_error=error,
            cleanup_detail=cleanup["detail"],
            object_uri=object_uri,
        ) from error
    else:
        cleanup = _cleanup(storage, object_uri, version_id, keep=keep)

    return {
        "status": "ok" if cleanup["deleted"] or keep else "warning",
        "object_uri": object_uri,
        "listed_count": len(listed_objects),
        "message": "S3 upload, download, content 비교, prefix listing 확인 완료",
        "cleanup": cleanup,
    }


def _cleanup(
    storage: S3Storage,
    object_uri: str | None,
    version_id: str | None = None,
    *,
    keep: bool,
) -> dict:
    """임시 object를 삭제하고 결과를 알려줍니다. 실패해도 예외를 올리지 않습니다."""
    if object_uri is None:
        return {
            "deleted": False,
            "version_id": None,
            "detail": "업로드 전에 중단되어 삭제할 object가 없습니다.",
        }
    if keep:
        return {
            "deleted": False,
            "version_id": version_id,
            "detail": "--keep 옵션으로 임시 object를 남겨두었습니다.",
        }

    uploaded_key = unquote(urlsplit(object_uri).path.lstrip("/"))
    try:
        _delete_object(storage, uploaded_key, version_id)
        if _object_remains(storage, uploaded_key, version_id):
            return {
                "deleted": False,
                "version_id": version_id,
                "detail": "삭제 요청 후에도 object가 남아 있습니다. 직접 삭제하세요.",
            }
    except Exception as error:  # 정리 실패가 검증 결과를 가리지 않게 합니다.
        return {
            "deleted": False,
            "version_id": version_id,
            "detail": f"임시 object 삭제 실패: {type(error).__name__}: {error}. 직접 삭제하세요.",
        }
    return {
        "deleted": True,
        "version_id": version_id,
        "detail": "임시 object를 삭제했습니다.",
    }


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
