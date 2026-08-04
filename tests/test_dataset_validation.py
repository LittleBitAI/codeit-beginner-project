import io
import json
from unittest.mock import Mock

from scripts.dataset_validation import validate_dataset, write_report


def _object(key, body):
    return {"Key": key, "Size": len(body)}


def test_dataset_validation_uses_mocked_s3_and_reports_coco_issues(tmp_path):
    prefix = "datasets/pill_detection/raw/v1/"
    annotation_key = f"{prefix}original/train_annotations/instances.json"
    document = {
        "images": [
            {"id": 1, "file_name": "one.png", "width": 10, "height": 10},
            {"id": 1, "file_name": "two.png", "width": 10, "height": 10},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
            {"id": 1, "image_id": 9, "category_id": 2, "bbox": [0, 0, 0, 1]},
        ],
        "categories": [{"id": 1, "name": "pill"}],
    }
    image_bytes = io.BytesIO()
    from PIL import Image

    Image.new("RGB", (1, 1)).save(image_bytes, format="PNG")
    image_body = image_bytes.getvalue()
    bodies = {
        f"{prefix}manifest.json": json.dumps({"version": "v1"}).encode(),
        annotation_key: json.dumps(document).encode(),
        f"{prefix}original/train_images/one.png": image_body,
        f"{prefix}original/train_images/two.png": image_body,
        f"{prefix}original/train_images/unreferenced.png": image_body,
        f"{prefix}original/test_images/test.png": image_body,
    }
    client = Mock()
    paginator = Mock()
    paginator.paginate.return_value = [{"Contents": [_object(key, body) for key, body in bodies.items()]}]
    client.get_paginator.return_value = paginator
    client.get_object.side_effect = lambda Bucket, Key: {"Body": io.BytesIO(bodies[Key])}
    client.download_file.side_effect = lambda bucket, key, destination: open(destination, "wb").write(bodies[key])
    sts_client = Mock()
    sts_client.get_caller_identity.return_value = {"Arn": "arn:aws:iam::123456789012:user/test"}

    report = validate_dataset(
        bucket="test-bucket",
        prefix=prefix,
        s3_client=client,
        sts_client=sts_client,
        checked_at="2026-08-04T00:00:00Z",
    )

    assert report["status"] == "fail"
    assert report["train_image_count"] == 3
    assert report["test_image_count"] == 1
    assert report["annotation_file_count"] == 1
    assert report["category_count"] == 1
    assert report["annotation_count"] == 2
    assert report["class_distribution"] == {"pill": 1}
    assert report["duplicate_ids"]["image_ids"][annotation_key] == [1]
    assert report["duplicate_ids"]["annotation_ids"][annotation_key] == [1]
    assert len(report["invalid_bboxes"]) == 1
    assert any("missing image_id" in error for error in report["errors"])
    assert any("missing category_id" in error for error in report["errors"])
    assert report["sample_downloads"]["annotation_json"]["status"] == "ok"
    assert len(report["sample_downloads"]["images"]) == 4

    output = write_report(report, tmp_path / "report.json")
    assert json.loads(output.read_text(encoding="utf-8"))["bucket"] == "test-bucket"
    client.head_bucket.assert_called_once_with(Bucket="test-bucket")
