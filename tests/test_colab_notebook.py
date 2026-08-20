"""Colab notebook의 S3 전송비 안전장치를 검사합니다."""

import json
from pathlib import Path


NOTEBOOK = Path("docs/pill-detection-colab.ipynb")


def test_drive_cache_is_restored_before_s3_use_and_saved_after_training() -> None:
    """새 runtime은 S3 image를 받기 전에 Drive cache를 먼저 복원합니다."""

    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = [
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in cells
    ]

    credentials = next(i for i, source in enumerate(sources) if "AWS_DEFAULT_REGION" in source)
    restore = next(i for i, source in enumerate(sources) if "CACHE_ARCHIVE.is_file()" in source)
    first_s3_use = next(i for i, source in enumerate(sources) if "--list-datasets" in source)
    save = next(i for i, source in enumerate(sources) if "CACHE_ARCHIVE_PARTIAL" in source)

    assert "ap-southeast-2" in sources[credentials]
    assert credentials < restore < first_s3_use < save
