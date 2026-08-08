"""변환한 AI Hub 데이터를 대회 raw 형식의 새 S3 prefix로 올립니다.

`aihub_to_competition.py`가 만든 annotation tree와 AI Hub 원천 이미지를 합쳐,
data pipeline이 그대로 읽을 수 있는 prefix 하나를 만듭니다.

    <dest>/train_images/<이미지>.png              AI Hub 이미지를 **평탄하게** 올림
    <dest>/train_annotations/<조합>_json/...       변환 결과를 그대로 올림
    <dest>/train_images, train_annotations, test_images 중 대회 몫은 bucket 안에서 복사

세 가지를 그냥 sync하면 안 되기 때문에 이 도구가 있습니다.

1. AI Hub 원천데이터에는 조합마다 `_index.png`가 있는데, 실제 촬영 사진이 아니고
   참조하는 라벨도 없습니다. 통째로 올리면 6 GiB를 헛되게 씁니다. 그래서 올릴 목록은
   폴더를 훑어서가 아니라 **annotation이 가리키는 이미지**로 정합니다. 라벨이 없어
   변환이 버린 이미지도 같은 이유로 빠집니다.
2. 원천데이터는 `TS_*/<조합>/<이미지>.png`로 중첩인데 대회 `train_images/`는 평탄합니다.
3. `preparation.py`는 `train_images`, `train_annotations`, `test_images`가 **모두**
   같은 prefix에 있어야 실행합니다. 대회 몫은 내려받지 않고 서버 사이드로 복사합니다.

`raw/v1/`은 대회가 준 원본이라 목적지가 될 수 없고, 이미 있는 객체는 덮지 않습니다.
그래서 중단된 업로드를 같은 명령으로 이어서 할 수 있습니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common import S3Storage, Storage, StorageError


DEFAULT_DEST_PREFIX = "datasets/pill_detection/raw/v2/original/"
DEFAULT_SOURCE_PREFIX = "datasets/pill_detection/raw/v1/original/"

# 대회가 준 원본입니다. 목적지로 쓰면 다른 사람이 만든 artifact를 덮게 됩니다.
PROTECTED_PREFIX = "datasets/pill_detection/raw/v1/"

# 대회 prefix에서 새 prefix로 복사해 올 하위 경로입니다.
COPIED_DIRECTORIES = ("train_images/", "train_annotations/", "test_images/")

DEFAULT_WORKERS = 16


class UploadError(RuntimeError):
    """업로드를 시작할 수 없을 때 올리는 오류입니다."""


@dataclass(frozen=True)
class UploadPlan:
    """올릴 local file과 목적지 key의 짝입니다."""

    images: list[tuple[Path, str]] = field(default_factory=list)
    annotations: list[tuple[Path, str]] = field(default_factory=list)
    dest_prefix: str = DEFAULT_DEST_PREFIX

    @property
    def total_bytes(self) -> int:
        return sum(path.stat().st_size for path, _ in self.images + self.annotations)


def _normalized_prefix(value: str) -> str:
    text = str(value).strip().strip("/")
    if not text:
        raise UploadError("prefix가 비어 있습니다.")
    return f"{text}/"


def _referenced_image_names(converted: Path) -> list[str]:
    """변환 결과가 실제로 가리키는 이미지 이름을 모읍니다."""

    root = converted / "train_annotations"
    if not root.is_dir():
        raise UploadError(f"변환 결과에 train_annotations가 없습니다: {converted.as_posix()}")
    names: set[str] = set()
    for path in root.rglob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        names.add(document["images"][0]["file_name"])
    if not names:
        raise UploadError("변환 결과에서 annotation 문서를 찾지 못했습니다.")
    return sorted(names)


def _local_images(images_root: Path) -> dict[str, Path]:
    """원천데이터 tree의 PNG를 이름으로 찾을 수 있게 만듭니다.

    무엇을 올릴지는 annotation이 정합니다. 이 map은 이름을 경로로 바꾸는 색인일
    뿐이고, 참조되지 않은 file은 여기 있어도 계획에 들어가지 않습니다.
    """

    return {path.name: path for path in images_root.rglob("*.png")}


def build_plan(
    converted: Path,
    images_root: Path,
    *,
    dest_prefix: str = DEFAULT_DEST_PREFIX,
) -> UploadPlan:
    """올릴 목록을 만듭니다. 빠진 이미지가 있으면 여기서 멈춥니다."""

    prefix = _normalized_prefix(dest_prefix)
    if prefix.startswith(PROTECTED_PREFIX):
        raise UploadError(
            f"{PROTECTED_PREFIX}는 대회가 준 원본이라 목적지로 쓸 수 없습니다: {prefix}"
        )
    converted, images_root = Path(converted), Path(images_root)

    names = _referenced_image_names(converted)
    available = _local_images(images_root)
    missing = [name for name in names if name not in available]
    if missing:
        raise UploadError(
            f"변환 결과가 가리키는 원천 이미지 {len(missing)}개를 찾지 못했습니다 "
            f"(예: {', '.join(missing[:3])})"
        )

    annotation_root = converted / "train_annotations"
    annotations = sorted(
        (
            path,
            f"{prefix}train_annotations/"
            f"{path.relative_to(annotation_root).as_posix()}",
        )
        for path in annotation_root.rglob("*.json")
    )
    # 대회 train_images는 평탄합니다. 파일 이름이 전역적으로 유일해야 합니다.
    image_pairs = sorted((available[name], f"{prefix}train_images/{name}") for name in names)
    return UploadPlan(images=image_pairs, annotations=annotations, dest_prefix=prefix)


def _copy_within_bucket(storage: S3Storage, source_key: str, dest_key: str) -> None:
    """bucket 안에서 서버 사이드로 복사합니다. 내려받지 않습니다."""

    storage.client.copy_object(
        Bucket=storage.bucket,
        Key=dest_key,
        CopySource={"Bucket": storage.bucket, "Key": source_key},
    )


def _listed_keys(storage: Storage, prefix: str) -> list[str]:
    """`list`가 돌려준 URI에서 bucket 뒤의 key만 뽑습니다."""

    bucket = getattr(storage, "bucket", "")
    keys = []
    for uri in storage.list(prefix):
        text = str(uri)
        keys.append(text.split(f"{bucket}/", 1)[-1] if "://" in text else text)
    return keys


def _existing_keys(storage: Storage, prefix: str) -> set[str]:
    """목적지에 이미 있는 key를 한 번의 listing으로 모읍니다.

    객체마다 `exists`를 부르면 5만 개에 왕복이 두 배로 늘어납니다. 중단된 업로드를
    이어서 할 때도 이 집합만 보면 됩니다.
    """

    return set(_listed_keys(storage, prefix))


def _keys_to_copy(storage: Storage, source_prefix: str, dest_prefix: str) -> list[tuple[str, str]]:
    """대회 prefix에서 가져올 객체의 (원본 key, 목적지 key)를 모읍니다."""

    pairs: list[tuple[str, str]] = []
    for key in _listed_keys(storage, source_prefix):
        relative = key[len(source_prefix):]
        if not relative or key.endswith("/"):
            continue
        if not relative.startswith(COPIED_DIRECTORIES):
            continue
        pairs.append((key, f"{dest_prefix}{relative}"))
    return sorted(pairs)


def upload(
    storage: Storage,
    plan: UploadPlan,
    *,
    source_prefix: str = DEFAULT_SOURCE_PREFIX,
    workers: int = DEFAULT_WORKERS,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """계획대로 올리고, 대회 객체를 복사합니다. 이미 있는 것은 건너뜁니다."""

    source_prefix = _normalized_prefix(source_prefix)
    copies = _keys_to_copy(storage, source_prefix, plan.dest_prefix)
    planned = {
        "images": len(plan.images),
        "annotations": len(plan.annotations),
        "copies": len(copies),
        "bytes": plan.total_bytes,
    }
    uploaded: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    failures: list[str] = []

    if dry_run:
        return {
            "dry_run": True,
            "dest_prefix": plan.dest_prefix,
            "planned": planned,
            "uploaded": {"images": 0, "annotations": 0},
            "skipped": {"images": 0, "annotations": 0},
            "copied": 0,
            "failures": [],
        }

    existing = set() if overwrite else _existing_keys(storage, plan.dest_prefix)

    def send(kind: str, item: tuple[Path, str]) -> None:
        path, key = item
        try:
            if key in existing:
                skipped[kind] += 1
                return
            storage.upload_file(path, key, overwrite=overwrite)
            uploaded[kind] += 1
        except StorageError as error:
            failures.append(f"{key}: {error}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for kind, items in (("images", plan.images), ("annotations", plan.annotations)):
            list(executor.map(lambda item, kind=kind: send(kind, item), items))

    copied = 0
    for source_key, dest_key in copies:
        try:
            if dest_key in existing:
                skipped["copies"] += 1
                continue
            _copy_within_bucket(storage, source_key, dest_key)
            copied += 1
        except (StorageError, AttributeError, KeyError) as error:
            failures.append(f"{dest_key}: {error}")

    return {
        "dry_run": False,
        "dest_prefix": plan.dest_prefix,
        "planned": planned,
        "uploaded": {"images": uploaded["images"], "annotations": uploaded["annotations"]},
        "skipped": {
            "images": skipped["images"],
            "annotations": skipped["annotations"],
            "copies": skipped["copies"],
        },
        "copied": copied,
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="변환한 AI Hub 데이터를 대회 형식의 새 S3 prefix로 올립니다."
    )
    parser.add_argument("--converted", required=True, help="aihub_to_competition.py의 출력 위치")
    parser.add_argument("--images", required=True, help="AI Hub 1.Training 디렉터리")
    parser.add_argument("--dest-prefix", default=DEFAULT_DEST_PREFIX)
    parser.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    parser.add_argument("--bucket", help="기본값은 PILL_STORAGE_S3_BUCKET")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default="artifacts/aihub-upload-report.json")
    arguments = parser.parse_args(argv)

    bucket = arguments.bucket or os.environ.get("PILL_STORAGE_S3_BUCKET", "")
    if not bucket:
        print("업로드 실패: --bucket 또는 PILL_STORAGE_S3_BUCKET이 필요합니다.")
        return 1

    try:
        plan = build_plan(
            Path(arguments.converted), Path(arguments.images), dest_prefix=arguments.dest_prefix
        )
        report = upload(
            S3Storage(bucket),
            plan,
            source_prefix=arguments.source_prefix,
            workers=arguments.workers,
            dry_run=arguments.dry_run,
            overwrite=arguments.overwrite,
        )
    except (UploadError, StorageError) as error:
        print(f"업로드 실패: {error}")
        return 1

    planned = report["planned"]
    gib = planned["bytes"] / 2**30
    print(f"목적지 s3://{bucket}/{report['dest_prefix']}")
    print(
        f"계획: 이미지 {planned['images']}개({gib:.2f} GiB), "
        f"annotation {planned['annotations']}개, 대회 복사 {planned['copies']}개"
    )
    if report["dry_run"]:
        print("dry-run이므로 아무것도 올리지 않았습니다.")
        return 0
    print(
        f"올림: 이미지 {report['uploaded']['images']}개, "
        f"annotation {report['uploaded']['annotations']}개, 복사 {report['copied']}개"
    )
    print(
        f"건너뜀(이미 있음): 이미지 {report['skipped']['images']}개, "
        f"annotation {report['skipped']['annotations']}개, 복사 {report['skipped']['copies']}개"
    )
    output = Path(arguments.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    if report["failures"]:
        print(f"실패 {len(report['failures'])}건. 예: {report['failures'][0]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
