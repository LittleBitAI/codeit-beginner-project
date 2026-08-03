from unittest.mock import Mock, patch

from scripts import s3_smoke_test
from src.common.storage import S3Storage


def test_smoke_test_uses_only_unique_test_object_and_does_not_delete():
    storage = S3Storage("test-bucket", client=Mock())
    storage.upload_file = Mock(
        return_value=(
            "s3://test-bucket/experiments/uploading/smoke-tests/fixed-id.json"
        )
    )

    def download_file(source, destination):
        destination.write_text(
            '{\n'
            '  "purpose": "pill-object-detection-s3-smoke-test",\n'
            '  "object_key": '
            '"experiments/uploading/smoke-tests/fixed-id.json"\n'
            '}\n',
            encoding="utf-8",
            newline="\n",
        )

    storage.download_file = Mock(side_effect=download_file)
    storage.list = Mock(
        return_value=[
            "s3://test-bucket/experiments/uploading/smoke-tests/fixed-id.json"
        ]
    )

    with (
        patch.object(s3_smoke_test, "load_config", return_value={}),
        patch.object(s3_smoke_test, "create_storage", return_value=storage),
        patch.object(s3_smoke_test.uuid, "uuid4") as uuid4,
    ):
        uuid4.return_value.hex = "fixed-id"
        result = s3_smoke_test.run_smoke_test("configs/env.aws.json")

    assert result["status"] == "ok"
    storage.upload_file.assert_called_once()
    assert storage.upload_file.call_args.args[1] == (
        "experiments/uploading/smoke-tests/fixed-id.json"
    )
    storage.list.assert_called_once_with("experiments/uploading/smoke-tests/")
    assert not hasattr(storage, "delete")
