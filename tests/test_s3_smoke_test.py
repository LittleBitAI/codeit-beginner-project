from unittest.mock import Mock, patch

import pytest

from scripts import s3_smoke_test
from src.common.storage import S3Storage, StorageError


SMOKE_URI = "s3://test-bucket/experiments/uploading/smoke-tests/fixed-id.json"
SMOKE_KEY = "experiments/uploading/smoke-tests/fixed-id.json"


def build_storage(*, listed=None, exists_after_delete=False):
    """왕복이 성공하도록 준비한 S3Storage mock을 만듭니다."""
    storage = S3Storage("test-bucket", client=Mock())
    storage.upload_file = Mock(return_value=SMOKE_URI)

    def download_file(source, destination):
        destination.write_text(
            '{\n'
            '  "purpose": "pill-object-detection-s3-smoke-test",\n'
            f'  "object_key": "{SMOKE_KEY}"\n'
            '}\n',
            encoding="utf-8",
            newline="\n",
        )

    storage.download_file = Mock(side_effect=download_file)
    storage.list = Mock(return_value=[SMOKE_URI] if listed is None else listed)
    storage.exists = Mock(return_value=exists_after_delete)
    return storage


def run(storage, **kwargs):
    with (
        patch.object(s3_smoke_test, "load_config", return_value={}),
        patch.object(s3_smoke_test, "create_storage", return_value=storage),
        patch.object(s3_smoke_test.uuid, "uuid4") as uuid4,
    ):
        uuid4.return_value.hex = "fixed-id"
        return s3_smoke_test.run_smoke_test("configs/env.aws.json", **kwargs)


def test_smoke_test_uploads_reads_and_deletes_the_temporary_object():
    storage = build_storage()

    result = run(storage)

    assert result["status"] == "ok"
    storage.upload_file.assert_called_once()
    assert storage.upload_file.call_args.args[1] == SMOKE_KEY
    storage.list.assert_called_once_with("experiments/uploading/smoke-tests/")
    storage.client.delete_object.assert_called_once_with(
        Bucket="test-bucket", Key=SMOKE_KEY
    )
    assert result["cleanup"] == {
        "deleted": True,
        "detail": "임시 object를 삭제했습니다.",
    }


def test_keep_option_leaves_the_object_in_place():
    storage = build_storage()

    result = run(storage, keep=True)

    assert result["status"] == "ok"
    storage.client.delete_object.assert_not_called()
    assert result["cleanup"]["deleted"] is False
    assert "--keep" in result["cleanup"]["detail"]


def test_object_still_present_after_delete_is_reported_as_warning():
    storage = build_storage(exists_after_delete=True)

    result = run(storage)

    assert result["status"] == "warning"
    assert result["cleanup"]["deleted"] is False
    assert "직접 삭제" in result["cleanup"]["detail"]


def test_delete_failure_does_not_hide_a_successful_round_trip():
    storage = build_storage()
    storage.client.delete_object.side_effect = RuntimeError("권한 없음")

    result = run(storage)

    assert result["status"] == "warning"
    assert "임시 object 삭제 실패" in result["cleanup"]["detail"]


def test_verification_failure_still_deletes_the_temporary_object():
    storage = build_storage(listed=[])

    with pytest.raises(StorageError, match="prefix listing에 없습니다"):
        run(storage)

    storage.client.delete_object.assert_called_once_with(
        Bucket="test-bucket", Key=SMOKE_KEY
    )


def test_failure_before_upload_has_nothing_to_delete():
    storage = build_storage()
    storage.upload_file.side_effect = StorageError("업로드 실패")

    with pytest.raises(StorageError, match="업로드 실패"):
        run(storage)

    storage.client.delete_object.assert_not_called()


def test_non_s3_backend_is_rejected():
    with (
        patch.object(s3_smoke_test, "load_config", return_value={}),
        patch.object(s3_smoke_test, "create_storage", return_value=object()),
    ):
        with pytest.raises(StorageError, match="s3 storage backend"):
            s3_smoke_test.run_smoke_test("configs/env.local.json")


def test_config_is_required_when_not_installing(capsys):
    with pytest.raises(SystemExit):
        s3_smoke_test.main([])

    assert "--config가 필요합니다" in capsys.readouterr().err
