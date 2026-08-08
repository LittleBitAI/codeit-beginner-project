"""변환한 AI Hub 데이터를 S3 raw prefix로 올리는 도구의 계약을 확인합니다."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.upload_aihub_dataset import (
    UploadError,
    build_plan,
    upload,
)
from src.common.storage import S3Storage


BUCKET = "test-bucket"
DEST = "datasets/pill_detection/raw/v2/original/"
SOURCE = "datasets/pill_detection/raw/v1/original/"


def not_found() -> ClientError:
    return ClientError({"Error": {"Code": "404"}}, "HeadObject")


class FakeClient:
    """put/copy/head/list만 흉내내는 최소 S3 client입니다."""

    def __init__(self, *, existing: set[str] | None = None, listed: list[str] | None = None):
        self.existing = set(existing or ())
        self.listed = list(listed or ())
        self.puts: list[str] = []
        self.copies: list[tuple[str, str]] = []

    def put_object(self, Bucket: str, Key: str, Body: Any = None, **kwargs: Any) -> dict[str, Any]:
        if "IfNoneMatch" in kwargs and Key in self.existing:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.puts.append(Key)
        self.existing.add(Key)
        return {}

    def copy_object(self, Bucket: str, Key: str, CopySource: Any = None, **kwargs: Any) -> dict:
        self.copies.append((CopySource["Key"], Key))
        self.existing.add(Key)
        return {}

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key in self.existing:
            return {}
        raise not_found()

    def get_paginator(self, name: str) -> Any:
        listed = self.listed

        class Paginator:
            def paginate(self, Bucket: str, Prefix: str = "") -> list[dict[str, Any]]:
                return [
                    {"Contents": [{"Key": key} for key in listed if key.startswith(Prefix)]}
                ]

        return Paginator()


def _storage(**kwargs: Any) -> S3Storage:
    return S3Storage(BUCKET, client=FakeClient(**kwargs))


def _converted(root: Path, *, image_names: list[str]) -> Path:
    """변환 결과 tree를 흉내 냅니다. 문서 하나에 이미지 하나입니다."""

    for name in image_names:
        combo = name.split("_")[0]
        directory = root / "train_annotations" / f"{combo}_json" / "K-000250"
        target = directory / f"{Path(name).stem}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        image = {"file_name": name, "width": 976, "height": 1280, "id": 100001}
        annotation = {"id": 1, "image_id": 100001, "category_id": 250, "bbox": [1, 2, 3, 4]}
        target.write_text(
            json.dumps(
                {
                    "images": [image],
                    "annotations": [annotation],
                    "categories": [{"id": 250, "name": "마그밀정"}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    (root / "conversion_report.json").write_text("{}", encoding="utf-8")
    return root


def _images(root: Path, *, names: list[str], with_index: bool = True) -> Path:
    """중첩된 원천데이터 tree를 흉내 냅니다."""

    for name in names:
        combo = name.split("_")[0]
        directory = root / "원천데이터" / "TS_1_조합" / combo
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"png-bytes")
        if with_index:
            (directory / f"{combo}_index.png").write_bytes(b"index-sheet")
    return root


NAMES = ["K-000250-000573_0_2_70.png", "K-000250-000573_0_2_75.png"]


def test_plan_uploads_only_images_the_annotations_reference(tmp_path: Path) -> None:
    """올릴 목록은 annotation이 정합니다. 폴더를 훑지 않습니다.

    이 규칙 하나로 `_index.png` 3,503개(6.1 GiB)와 라벨이 없어 변환이 버린 이미지가
    함께 빠집니다.
    """

    source = _images(tmp_path / "src", names=NAMES)
    # 라벨이 없어 변환이 버린 이미지를 흉내 냅니다.
    orphan = source / "원천데이터" / "TS_1_조합" / "K-000250-000573" / "K-000250-000573_0_2_99.png"
    orphan.write_bytes(b"orphan")

    plan = build_plan(_converted(tmp_path / "conv", image_names=NAMES), source, dest_prefix=DEST)

    assert sorted(key for _, key in plan.images) == [f"{DEST}train_images/{name}" for name in NAMES]
    assert not any("_index" in key for _, key in plan.images)
    assert not any("_0_2_99" in key for _, key in plan.images)


def test_plan_keeps_the_nested_annotation_layout(tmp_path: Path) -> None:
    """annotation은 대회와 같은 `<조합>_json/<알약코드>/` 경로를 유지합니다."""

    plan = build_plan(
        _converted(tmp_path / "conv", image_names=NAMES),
        _images(tmp_path / "src", names=NAMES),
        dest_prefix=DEST,
    )

    assert len(plan.annotations) == 2
    assert all(
        key.startswith(f"{DEST}train_annotations/K-000250-000573_json/K-000250/")
        for _, key in plan.annotations
    )


def test_plan_fails_when_a_referenced_image_is_missing(tmp_path: Path) -> None:
    """참조된 이미지가 없으면 아무것도 올리기 전에 멈춥니다."""

    converted = _converted(tmp_path / "conv", image_names=NAMES)
    partial = _images(tmp_path / "src", names=NAMES[:1])

    with pytest.raises(UploadError, match="원천 이미지"):
        build_plan(converted, partial, dest_prefix=DEST)


def test_uploading_into_the_competition_prefix_is_refused(tmp_path: Path) -> None:
    """`raw/v1`은 대회가 준 원본이라 목적지로 쓸 수 없습니다."""

    with pytest.raises(UploadError, match="raw/v1"):
        build_plan(
            _converted(tmp_path / "conv", image_names=NAMES),
            _images(tmp_path / "src", names=NAMES),
            dest_prefix=SOURCE,
        )


def test_dry_run_uploads_nothing(tmp_path: Path) -> None:
    """`dry_run`은 계획만 세어 보고 객체를 만들지 않습니다."""

    storage = _storage()
    plan = build_plan(
        _converted(tmp_path / "conv", image_names=NAMES),
        _images(tmp_path / "src", names=NAMES),
        dest_prefix=DEST,
    )

    report = upload(storage, plan, source_prefix=SOURCE, dry_run=True)

    assert storage.client.puts == []
    assert storage.client.copies == []
    assert report["planned"]["images"] == 2
    assert report["uploaded"]["images"] == 0


def test_existing_objects_are_skipped_so_a_run_can_resume(tmp_path: Path) -> None:
    """이미 올라간 객체는 건너뛰어 중단된 업로드를 이어서 할 수 있습니다."""

    done = f"{DEST}train_images/{NAMES[0]}"
    # 목적지 listing 한 번으로 이미 올라간 것을 알아냅니다.
    storage = _storage(existing={done}, listed=[done])
    plan = build_plan(
        _converted(tmp_path / "conv", image_names=NAMES),
        _images(tmp_path / "src", names=NAMES),
        dest_prefix=DEST,
    )

    report = upload(storage, plan, source_prefix=SOURCE)

    assert done not in storage.client.puts
    assert f"{DEST}train_images/{NAMES[1]}" in storage.client.puts
    assert report["skipped"]["images"] == 1
    assert report["uploaded"]["images"] == 1


def test_competition_objects_are_copied_inside_the_bucket(tmp_path: Path) -> None:
    """대회 train/test 객체는 내려받지 않고 bucket 안에서 복사합니다."""

    listed = [
        f"{SOURCE}train_images/comp.png",
        f"{SOURCE}train_annotations/C_json/K-001900/comp.json",
        f"{SOURCE}test_images/1.png",
    ]
    storage = _storage(listed=listed)
    plan = build_plan(
        _converted(tmp_path / "conv", image_names=NAMES),
        _images(tmp_path / "src", names=NAMES),
        dest_prefix=DEST,
    )

    report = upload(storage, plan, source_prefix=SOURCE)

    assert sorted(storage.client.copies) == sorted(
        (key, key.replace(SOURCE, DEST)) for key in listed
    )
    assert report["copied"] == 3
