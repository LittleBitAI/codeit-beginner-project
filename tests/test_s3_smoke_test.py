from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from scripts import s3_smoke_test
from src.common.storage import S3Storage, StorageError


SMOKE_URI = "s3://test-bucket/experiments/uploading/smoke-tests/fixed-id.json"
SMOKE_KEY = "experiments/uploading/smoke-tests/fixed-id.json"
PREFIXED_SMOKE_URI = (
    "s3://test-bucket/team/dev/experiments/uploading/smoke-tests/fixed-id.json"
)
PREFIXED_SMOKE_KEY = "team/dev/experiments/uploading/smoke-tests/fixed-id.json"
PAYLOAD_TEXT = (
    '{\n'
    '  "purpose": "pill-object-detection-s3-smoke-test",\n'
    f'  "object_key": "{SMOKE_KEY}"\n'
    '}\n'
)


def not_found() -> ClientError:
    return ClientError({"Error": {"Code": "404"}}, "HeadObject")


class FakeClient:
    """VersionId와 delete marker 동작을 흉내내는 최소 S3 client입니다."""

    def __init__(self, *, version_id=None, remains_after_delete=False, delete_error=None):
        self.version_id = version_id
        self.remains_after_delete = remains_after_delete
        self.delete_error = delete_error
        self.delete_calls = []
        self.head_calls = []
        self.deleted = False

    def head_object(self, Bucket, Key, VersionId=None):
        self.head_calls.append({"Key": Key, "VersionId": VersionId})
        if not self.deleted:
            return {"VersionId": self.version_id} if self.version_id else {}
        if self.remains_after_delete:
            return {"VersionId": self.version_id} if self.version_id else {}
        raise not_found()

    def delete_object(self, Bucket, Key, VersionId=None):
        self.delete_calls.append({"Key": Key, "VersionId": VersionId})
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted = True
        return {}


def build_storage(*, listed=None, client=None, **client_kwargs):
    """왕복이 성공하도록 준비한 S3Storage mock을 만듭니다."""
    storage = S3Storage("test-bucket", client=client or FakeClient(**client_kwargs))
    storage.upload_file = Mock(return_value=SMOKE_URI)

    def download_file(source, destination):
        destination.write_text(PAYLOAD_TEXT, encoding="utf-8", newline="\n")

    storage.download_file = Mock(side_effect=download_file)
    storage.list = Mock(return_value=[SMOKE_URI] if listed is None else listed)
    # delete marker 때문에 key 단위 조회는 성공해도 False가 됩니다.
    storage.exists = Mock(return_value=False)
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
    assert storage.client.delete_calls == [{"Key": SMOKE_KEY, "VersionId": None}]
    assert result["cleanup"]["deleted"] is True


def test_versioned_bucket_deletes_the_uploaded_version():
    storage = build_storage(version_id="v-uploaded-1")

    result = run(storage)

    assert result["status"] == "ok"
    assert storage.client.delete_calls == [
        {"Key": SMOKE_KEY, "VersionId": "v-uploaded-1"}
    ]
    assert result["cleanup"]["version_id"] == "v-uploaded-1"
    # 삭제 확인도 해당 version을 직접 조회해야 합니다.
    assert storage.client.head_calls[-1] == {"Key": SMOKE_KEY, "VersionId": "v-uploaded-1"}


def test_version_lookup_uses_the_prefixed_key_from_the_uploaded_uri():
    storage = build_storage(version_id="v-uploaded-1")
    storage.upload_file.return_value = PREFIXED_SMOKE_URI
    storage.list.return_value = [PREFIXED_SMOKE_URI]

    result = run(storage)

    assert result["status"] == "ok"
    assert storage.client.head_calls[0] == {
        "Key": PREFIXED_SMOKE_KEY,
        "VersionId": None,
    }
    assert storage.client.delete_calls == [
        {"Key": PREFIXED_SMOKE_KEY, "VersionId": "v-uploaded-1"}
    ]


def test_unversioned_bucket_null_version_deletes_without_version_id():
    storage = build_storage(version_id="null")

    result = run(storage)

    assert result["status"] == "ok"
    assert storage.client.delete_calls == [{"Key": SMOKE_KEY, "VersionId": None}]
    assert result["cleanup"]["version_id"] is None


def test_delete_marker_does_not_count_as_successful_cleanup():
    """회귀: 버전이 남아 있으면 key 조회가 404여도 정리 성공으로 보지 않습니다."""
    storage = build_storage(version_id="v-uploaded-1", remains_after_delete=True)

    result = run(storage)

    assert result["status"] == "warning"
    assert result["cleanup"]["deleted"] is False
    assert "직접 삭제" in result["cleanup"]["detail"]
    # storage.exists()는 delete marker 때문에 False를 주므로 근거로 쓰면 안 됩니다.
    storage.exists.assert_not_called()


def test_verification_and_cleanup_failure_are_reported_together():
    """회귀: 둘 다 실패하면 원래 오류, 정리 오류, 남은 URI를 함께 알립니다."""
    storage = build_storage(
        listed=[],
        version_id="v-uploaded-1",
        delete_error=RuntimeError("삭제 권한 없음"),
    )

    with pytest.raises(s3_smoke_test.SmokeTestCleanupError) as error:
        run(storage)

    message = str(error.value)
    assert "prefix listing에 없습니다" in message
    assert "삭제 권한 없음" in message
    assert SMOKE_URI in message
    assert "v-uploaded-1" in message

    assert isinstance(error.value.original_error, StorageError)
    assert "삭제 권한 없음" in error.value.cleanup_detail
    assert error.value.object_uri == SMOKE_URI
    assert isinstance(error.value.__cause__, StorageError)


def test_cleanup_error_is_a_storage_error():
    assert issubclass(s3_smoke_test.SmokeTestCleanupError, StorageError)


def test_verification_failure_with_successful_cleanup_raises_original_error():
    storage = build_storage(listed=[])

    with pytest.raises(StorageError, match="prefix listing에 없습니다") as error:
        run(storage)

    assert not isinstance(error.value, s3_smoke_test.SmokeTestCleanupError)
    assert storage.client.delete_calls == [{"Key": SMOKE_KEY, "VersionId": None}]


def test_keep_option_leaves_the_object_in_place():
    storage = build_storage(version_id="v-uploaded-1")

    result = run(storage, keep=True)

    assert result["status"] == "ok"
    assert storage.client.delete_calls == []
    assert result["cleanup"]["deleted"] is False
    assert result["cleanup"]["version_id"] == "v-uploaded-1"
    assert "--keep" in result["cleanup"]["detail"]


def test_delete_failure_does_not_hide_a_successful_round_trip():
    storage = build_storage(delete_error=RuntimeError("권한 없음"))

    result = run(storage)

    assert result["status"] == "warning"
    assert "임시 object 삭제 실패" in result["cleanup"]["detail"]


def test_failure_before_upload_has_nothing_to_delete():
    storage = build_storage()
    storage.upload_file.side_effect = StorageError("업로드 실패")

    with pytest.raises(StorageError, match="업로드 실패"):
        run(storage)

    assert storage.client.delete_calls == []


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
