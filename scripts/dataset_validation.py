"""Read-only integrity validation for the pill detection dataset in Amazon S3."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common import S3Storage, StorageError


RAW_DATASET_PREFIX = "datasets/pill_detection/raw/v1/"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXPECTED_PATHS = ("train_images/", "train_annotations/", "test_images/")
MAX_REPORTED_EXAMPLES = 100


def _normalise_prefix(prefix: str) -> str:
    return prefix.strip("/") + "/"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def _append_limited(target: list[Any], value: Any) -> None:
    if len(target) < MAX_REPORTED_EXAMPLES:
        target.append(value)


def _object_records(storage: S3Storage, prefix: str) -> list[dict[str, Any]]:
    """Return object keys and sizes with S3's paginated listing API."""
    paginator = storage.client.get_paginator("list_objects_v2")
    records = [
        {"key": item["Key"], "size": int(item.get("Size", 0))}
        for page in paginator.paginate(Bucket=storage.bucket, Prefix=prefix)
        for item in page.get("Contents", [])
    ]
    return sorted(records, key=lambda record: record["key"])


def _relative_path(key: str, directory_name: str) -> str:
    marker = f"{directory_name}/"
    return key.split(marker, 1)[1] if marker in key else key


def _image_objects(records: Iterable[Mapping[str, Any]], directory_name: str) -> list[str]:
    return [
        str(record["key"])
        for record in records
        if f"{directory_name}/" in str(record["key"])
        and Path(str(record["key"])).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]


def _annotation_objects(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(record["key"])
        for record in records
        if "train_annotations/" in str(record["key"])
        and Path(str(record["key"])).suffix.lower() == ".json"
    ]


def _duplicates(values: Iterable[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _image_key_for_filename(filename: Any, image_keys: Iterable[str]) -> str | None:
    if not isinstance(filename, str) or not filename.strip():
        return None

    normalised = filename.replace("\\", "/").lstrip("./")
    candidates = list(image_keys)
    direct_matches = [
        key
        for key in candidates
        if _relative_path(key, "train_images") == normalised
        or key.endswith(f"/train_images/{normalised}")
    ]
    if len(direct_matches) == 1:
        return direct_matches[0]

    basename_matches = [key for key in candidates if Path(key).name == Path(normalised).name]
    return basename_matches[0] if len(basename_matches) == 1 else None


def _validate_coco(
    document: Any,
    *,
    source_key: str,
    train_image_keys: list[str],
    report: dict[str, Any],
    object_counts: list[int],
    over_four_examples: list[dict[str, Any]],
    referenced_image_keys: set[str],
    category_counts: Counter[str],
    category_ids: set[Any],
) -> None:
    """Validate one COCO JSON document and aggregate its results into report."""
    if not isinstance(document, Mapping):
        report["errors"].append(f"COCO JSON root must be an object: {source_key}")
        return

    required = ("images", "annotations", "categories")
    if any(not isinstance(document.get(name), list) for name in required):
        missing = [name for name in required if not isinstance(document.get(name), list)]
        report["errors"].append(
            f"COCO required list fields are missing or invalid in {source_key}: {', '.join(missing)}"
        )
        return

    images = document["images"]
    annotations = document["annotations"]
    categories = document["categories"]
    image_entries = [entry for entry in images if isinstance(entry, Mapping)]
    annotation_entries = [entry for entry in annotations if isinstance(entry, Mapping)]
    category_entries = [entry for entry in categories if isinstance(entry, Mapping)]

    if len(image_entries) != len(images):
        report["errors"].append(f"Non-object image entry found in {source_key}")
    if len(annotation_entries) != len(annotations):
        report["errors"].append(f"Non-object annotation entry found in {source_key}")
    if len(category_entries) != len(categories):
        report["errors"].append(f"Non-object category entry found in {source_key}")

    image_ids = [entry.get("id") for entry in image_entries if "id" in entry]
    annotation_ids = [entry.get("id") for entry in annotation_entries if "id" in entry]
    duplicated_image_ids = _duplicates(image_ids)
    duplicated_annotation_ids = _duplicates(annotation_ids)
    if duplicated_image_ids:
        report["duplicate_ids"]["image_ids"][source_key] = duplicated_image_ids
    if duplicated_annotation_ids:
        report["duplicate_ids"]["annotation_ids"][source_key] = duplicated_annotation_ids

    images_by_id = {entry.get("id"): entry for entry in image_entries if "id" in entry}
    categories_by_id = {entry.get("id"): entry for entry in category_entries if "id" in entry}
    category_ids.update(categories_by_id)
    category_names = {
        category_id: str(entry.get("name", category_id))
        for category_id, entry in categories_by_id.items()
    }

    for image in image_entries:
        image_key = _image_key_for_filename(image.get("file_name"), train_image_keys)
        if image_key is None:
            _append_limited(
                report["missing_images"],
                {"annotation_file": source_key, "file_name": image.get("file_name")},
            )
        else:
            referenced_image_keys.add(image_key)

    objects_by_image_id: Counter[Any] = Counter()
    for index, annotation in enumerate(annotation_entries):
        annotation_label = f"{source_key}#annotation[{index}]"
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if image_id not in images_by_id:
            report["errors"].append(
                f"{annotation_label} references a missing image_id: {image_id!r}"
            )
        else:
            objects_by_image_id[image_id] += 1
        if category_id not in categories_by_id:
            report["errors"].append(
                f"{annotation_label} references a missing category_id: {category_id!r}"
            )
        else:
            category_counts[category_names[category_id]] += 1

        bbox = annotation.get("bbox")
        invalid_reason: str | None = None
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(value) for value in bbox):
            invalid_reason = "bbox must be a four-item numeric list"
        else:
            x, y, width, height = bbox
            if width <= 0 or height <= 0:
                invalid_reason = "bbox width and height must be greater than zero"
            elif image_id in images_by_id:
                image = images_by_id[image_id]
                image_width = image.get("width")
                image_height = image.get("height")
                if not _is_number(image_width) or not _is_number(image_height):
                    report["warnings"].append(
                        f"Cannot check bbox bounds without numeric image dimensions: {annotation_label}"
                    )
                elif x < 0 or y < 0 or x + width > image_width or y + height > image_height:
                    invalid_reason = "bbox is outside its image bounds"
        if invalid_reason:
            _append_limited(
                report["invalid_bboxes"],
                {"annotation": annotation_label, "bbox": bbox, "reason": invalid_reason},
            )
            report["errors"].append(f"Invalid bbox in {annotation_label}: {invalid_reason}")

    for image_id in images_by_id:
        object_count = objects_by_image_id.get(image_id, 0)
        object_counts.append(object_count)
        if object_count > 4:
            _append_limited(
                over_four_examples,
                {
                    "annotation_file": source_key,
                    "image_id": image_id,
                    "object_count": object_count,
                },
            )
    report["annotation_count"] += len(annotation_entries)


def _objects_per_image_summary(object_counts: list[int]) -> dict[str, Any]:
    if not object_counts:
        return {"image_count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "histogram": {}}
    histogram = Counter(object_counts)
    return {
        "image_count": len(object_counts),
        "min": min(object_counts),
        "max": max(object_counts),
        "mean": round(sum(object_counts) / len(object_counts), 4),
        "median": median(object_counts),
        "histogram": {str(value): histogram[value] for value in sorted(histogram)},
    }


def _validate_downloads(
    storage: S3Storage,
    *,
    train_image_keys: list[str],
    test_image_keys: list[str],
    annotation_keys: list[str],
    report: dict[str, Any],
) -> None:
    """Download only the requested samples and verify that they can be opened."""
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency is pinned
        report["errors"].append(f"Pillow is required for image sample checks: {error}")
        return

    selected_images = [("train", key) for key in train_image_keys[:3]] + [
        ("test", key) for key in test_image_keys[:1]
    ]
    report["sample_downloads"] = {"images": [], "annotation_json": None}
    with tempfile.TemporaryDirectory(prefix="pill-dataset-validation-") as directory:
        temporary_directory = Path(directory)
        for split, key in selected_images:
            destination = temporary_directory / f"{split}-{Path(key).name}"
            try:
                storage.download_file(_uri(storage.bucket, key), destination)
                with Image.open(destination) as image:
                    image.verify()
                report["sample_downloads"]["images"].append({"key": key, "status": "ok"})
            except (OSError, StorageError) as error:
                report["errors"].append(f"Sample image could not be opened ({key}): {error}")

        if annotation_keys:
            key = annotation_keys[0]
            destination = temporary_directory / Path(key).name
            try:
                storage.download_file(_uri(storage.bucket, key), destination)
                json.loads(destination.read_text(encoding="utf-8"))
                report["sample_downloads"]["annotation_json"] = {"key": key, "status": "ok"}
            except (OSError, StorageError, UnicodeDecodeError, json.JSONDecodeError) as error:
                report["errors"].append(f"Sample annotation JSON could not be opened ({key}): {error}")


def _read_annotation_documents(
    storage: S3Storage, annotation_keys: list[str]
) -> list[tuple[str, Any | Exception]]:
    """Read small per-image COCO documents concurrently without writing to S3."""
    def read_one(key: str) -> Any:
        return storage.read_json(_uri(storage.bucket, key))

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {key: executor.submit(read_one, key) for key in annotation_keys}
        results: list[tuple[str, Any | Exception]] = []
        for key in annotation_keys:
            try:
                results.append((key, futures[key].result()))
            except StorageError as error:
                results.append((key, error))
    return results


def validate_dataset(
    *,
    bucket: str | None = None,
    prefix: str = RAW_DATASET_PREFIX,
    region: str | None = None,
    profile: str | None = None,
    s3_client: Any | None = None,
    sts_client: Any | None = None,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Run a read-only dataset validation and return a JSON-serialisable report."""
    resolved_bucket = bucket or os.environ.get("PILL_STORAGE_S3_BUCKET")
    if not resolved_bucket:
        raise ValueError("PILL_STORAGE_S3_BUCKET must be set")
    resolved_region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    resolved_profile = profile or os.environ.get("AWS_PROFILE") or None
    resolved_prefix = _normalise_prefix(prefix)
    storage = S3Storage(
        resolved_bucket,
        profile=resolved_profile,
        region=resolved_region,
        client=s3_client,
    )
    report: dict[str, Any] = {
        "status": "fail",
        "bucket": resolved_bucket,
        "prefix": resolved_prefix,
        "checked_at": checked_at or _utc_now(),
        "train_image_count": 0,
        "test_image_count": 0,
        "annotation_file_count": 0,
        "category_count": 0,
        "annotation_count": 0,
        "missing_images": [],
        "unreferenced_images": [],
        "invalid_bboxes": [],
        "duplicate_ids": {"image_ids": {}, "annotation_ids": {}},
        "objects_per_image_summary": {},
        "class_distribution": {},
        "warnings": [],
        "errors": [],
        "connection": {"principal_arn": None, "region": resolved_region, "bucket_access": False},
        "expected_paths": {},
        "actual_top_level_entries": {},
        "extension_counts": {},
        "total_object_count": 0,
        "total_size_bytes": 0,
        "manifest": {"present": False},
    }

    try:
        if sts_client is None:
            import boto3

            sts_client = boto3.Session(
                profile_name=resolved_profile, region_name=resolved_region
            ).client("sts")
        report["connection"]["principal_arn"] = sts_client.get_caller_identity()["Arn"]
        storage.client.head_bucket(Bucket=resolved_bucket)
        report["connection"]["bucket_access"] = True
        records = _object_records(storage, resolved_prefix)
    except Exception as error:  # SDK error details are recorded without credentials.
        report["errors"].append(f"AWS connection or prefix listing failed: {type(error).__name__}: {error}")
        return report

    keys = [str(record["key"]) for record in records]
    report["total_object_count"] = len(records)
    report["total_size_bytes"] = sum(int(record["size"]) for record in records)
    report["extension_counts"] = dict(
        sorted(Counter(Path(key).suffix.lower().lstrip(".") or "[no extension]" for key in keys).items())
    )
    report["actual_top_level_entries"] = dict(
        sorted(
            Counter(
                key[len(resolved_prefix) :].split("/", 1)[0]
                for key in keys
                if key.startswith(resolved_prefix) and key[len(resolved_prefix) :]
            ).items()
        )
    )
    for expected_path in EXPECTED_PATHS:
        exists = any(key.startswith(f"{resolved_prefix}{expected_path}") for key in keys)
        report["expected_paths"][expected_path] = exists
        if not exists:
            report["warnings"].append(f"Expected path is absent at the raw prefix: {expected_path}")

    manifest_key = f"{resolved_prefix}manifest.json"
    if manifest_key in keys:
        report["manifest"]["present"] = True
        try:
            manifest = storage.read_json(_uri(resolved_bucket, manifest_key))
            report["manifest"]["content"] = manifest
        except StorageError as error:
            report["errors"].append(f"Manifest JSON parsing failed: {error}")

    train_image_keys = _image_objects(records, "train_images")
    test_image_keys = _image_objects(records, "test_images")
    annotation_keys = _annotation_objects(records)
    report["train_image_count"] = len(train_image_keys)
    report["test_image_count"] = len(test_image_keys)
    report["annotation_file_count"] = len(annotation_keys)
    manifest_content = report["manifest"].get("content")
    if isinstance(manifest_content, Mapping):
        for count_name, actual_count in (
            ("train_image_count", report["train_image_count"]),
            ("test_image_count", report["test_image_count"]),
        ):
            expected_count = manifest_content.get(count_name)
            if isinstance(expected_count, int) and expected_count != actual_count:
                report["errors"].append(
                    f"Manifest {count_name} is {expected_count}, but {actual_count} objects were listed."
                )
    if not annotation_keys:
        report["errors"].append("No annotation JSON files were found under train_annotations/")

    object_counts: list[int] = []
    over_four_examples: list[dict[str, Any]] = []
    referenced_image_keys: set[str] = set()
    category_counts: Counter[str] = Counter()
    category_ids: set[Any] = set()
    for annotation_key, document in _read_annotation_documents(storage, annotation_keys):
        if isinstance(document, Exception):
            report["errors"].append(
                f"Annotation JSON parsing failed ({annotation_key}): {document}"
            )
            continue
        _validate_coco(
            document,
            source_key=annotation_key,
            train_image_keys=train_image_keys,
            report=report,
            object_counts=object_counts,
            over_four_examples=over_four_examples,
            referenced_image_keys=referenced_image_keys,
            category_counts=category_counts,
            category_ids=category_ids,
        )

    for image_key in train_image_keys:
        if image_key not in referenced_image_keys:
            _append_limited(report["unreferenced_images"], image_key)

    report["category_count"] = len(category_ids)
    report["class_distribution"] = dict(sorted(category_counts.items()))
    report["objects_per_image_summary"] = _objects_per_image_summary(object_counts)
    if over_four_examples:
        report["warnings"].append("At least one image contains more than four objects.")
    report["images_over_four_objects"] = over_four_examples

    _validate_downloads(
        storage,
        train_image_keys=train_image_keys,
        test_image_keys=test_image_keys,
        annotation_keys=annotation_keys,
        report=report,
    )
    if report["errors"]:
        report["status"] = "fail"
    elif report["warnings"]:
        report["status"] = "warning"
    else:
        report["status"] = "pass"
    return report


def write_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write the locally generated report as UTF-8 without BOM and LF line endings."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only S3 pill dataset integrity validation")
    parser.add_argument("--bucket", help="S3 bucket (defaults to PILL_STORAGE_S3_BUCKET)")
    parser.add_argument("--prefix", default=RAW_DATASET_PREFIX)
    parser.add_argument("--output", default="artifacts/dataset_validation_report.json")
    args = parser.parse_args(argv)

    try:
        report = validate_dataset(bucket=args.bucket, prefix=args.prefix)
        output_path = write_report(report, args.output)
    except (OSError, ValueError) as error:
        print(f"Dataset validation could not start: {error}", file=sys.stderr)
        return 2

    summary = {
        "status": report["status"],
        "principal_arn": report["connection"]["principal_arn"],
        "region": report["connection"]["region"],
        "bucket": report["bucket"],
        "prefix": report["prefix"],
        "train_image_count": report["train_image_count"],
        "test_image_count": report["test_image_count"],
        "annotation_file_count": report["annotation_file_count"],
        "annotation_count": report["annotation_count"],
        "category_count": report["category_count"],
        "total_object_count": report["total_object_count"],
        "total_size_bytes": report["total_size_bytes"],
        "warnings": len(report["warnings"]),
        "errors": len(report["errors"]),
        "report_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
