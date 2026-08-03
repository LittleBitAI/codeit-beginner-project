"""Configured S3 storage에 대한 비파괴 smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common import S3Storage, StorageError, create_storage, load_config


SMOKE_PREFIX = "experiments/uploading/smoke-tests/"


def run_smoke_test(config_path: str | Path) -> dict:
    """작은 JSON object의 S3 왕복과 prefix listing을 확인합니다."""

    storage = create_storage(load_config(config_path))
    if not isinstance(storage, S3Storage):
        raise StorageError("smoke test에는 s3 storage backend 설정이 필요합니다.")

    object_key = f"{SMOKE_PREFIX}{uuid.uuid4().hex}.json"
    payload = {
        "purpose": "pill-object-detection-s3-smoke-test",
        "object_key": object_key,
    }

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

    return {
        "status": "ok",
        "object_uri": object_uri,
        "listed_count": len(listed_objects),
        "message": "S3 upload, download, content 비교, prefix listing 확인 완료",
        "cleanup": "자동 삭제하지 않았습니다. 생성된 smoke-test object URI를 확인하세요.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Amazon S3 storage smoke test")
    parser.add_argument("--config", required=True, help="AWS storage JSON config 경로")
    args = parser.parse_args(argv)

    try:
        result = run_smoke_test(args.config)
    except (StorageError, OSError, ValueError) as error:
        print(f"S3 smoke test 실패: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
