import io
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from src.common.storage import (
    BucketNotFoundError,
    CredentialsNotFoundError,
    LocalStorage,
    ObjectAlreadyExistsError,
    ObjectNotFoundError,
    S3Storage,
    StorageAccessDeniedError,
    StorageConfigurationError,
    create_storage,
)


def client_error(code: str, operation: str = "TestOperation") -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": "mocked AWS error"}},
        operation,
    )


def test_local_file_upload_download_and_overwrite_protection(tmp_path):
    storage = LocalStorage(tmp_path / "storage")
    source = tmp_path / "source.txt"
    source.write_text("알약 storage test\n", encoding="utf-8", newline="\n")

    stored_path = Path(storage.upload_file(source, "datasets/source.txt"))
    assert stored_path.read_text(encoding="utf-8") == "알약 storage test\n"
    assert storage.exists("datasets/source.txt")

    with pytest.raises(ObjectAlreadyExistsError):
        storage.upload_file(source, "datasets/source.txt")

    source.write_text("updated\n", encoding="utf-8", newline="\n")
    storage.upload_file(source, "datasets/source.txt", overwrite=True)
    destination = tmp_path / "downloaded.txt"
    storage.download_file("datasets/source.txt", destination)
    assert destination.read_text(encoding="utf-8") == "updated\n"

    with pytest.raises(ObjectAlreadyExistsError):
        storage.download_file("datasets/source.txt", destination)


def test_local_json_listing_and_missing_object(tmp_path):
    storage = LocalStorage(tmp_path / "storage")
    value = {"name": "알약", "count": 2}

    stored_path = storage.write_json("registry/items.json", value)
    storage.write_json("registry-other/items.json", value)

    assert storage.read_json("registry/items.json") == value
    assert storage.list("registry/") == [stored_path]
    assert len(storage.list()) == 2
    with pytest.raises(ObjectAlreadyExistsError):
        storage.write_json("registry/items.json", value)
    with pytest.raises(ObjectNotFoundError):
        storage.read_json("registry/missing.json")


def test_local_storage_rejects_paths_outside_root(tmp_path):
    storage = LocalStorage(tmp_path / "storage")

    with pytest.raises(StorageConfigurationError, match="root 밖"):
        storage.exists("../outside.json")


def test_create_storage_uses_environment_before_config(tmp_path):
    config = {
        "storage": {
            "backend": "s3",
            "s3": {"bucket": "config-bucket", "region": "ap-northeast-1"},
        }
    }
    storage = create_storage(
        config,
        environ={
            "PILL_STORAGE_BACKEND": "local",
            "PILL_STORAGE_LOCAL_ROOT": str(tmp_path),
        },
    )

    assert isinstance(storage, LocalStorage)
    assert storage.root == tmp_path.resolve()


def test_create_s3_storage_uses_standard_profile_and_region_variables():
    mock_client = Mock()
    storage = create_storage(
        {"storage": {"backend": "s3", "s3": {"bucket": "config-bucket"}}},
        environ={
            "PILL_STORAGE_S3_BUCKET": "environment-bucket",
            "PILL_STORAGE_S3_PREFIX": "team-space",
            "AWS_PROFILE": "temporary-profile",
            "AWS_REGION": "ap-northeast-2",
        },
        s3_client=mock_client,
    )

    assert isinstance(storage, S3Storage)
    assert storage.bucket == "environment-bucket"
    assert storage.prefix == "team-space"
    assert storage.profile == "temporary-profile"
    assert storage.region == "ap-northeast-2"


def test_s3_file_upload_uses_conditional_write_and_handles_s3_uri(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"smoke-test")
    captured = {}
    mock_client = Mock()

    def put_object(**request):
        captured.update(request)
        captured["content"] = request["Body"].read()

    mock_client.put_object.side_effect = put_object
    storage = S3Storage("test-bucket", client=mock_client)

    uri = storage.upload_file(source, "s3://test-bucket/datasets/source.txt")

    assert uri == "s3://test-bucket/datasets/source.txt"
    assert captured["Bucket"] == "test-bucket"
    assert captured["Key"] == "datasets/source.txt"
    assert captured["IfNoneMatch"] == "*"
    assert captured["content"] == b"smoke-test"


def test_s3_download_json_exists_and_prefix_listing(tmp_path):
    mock_client = Mock()
    mock_client.download_file.side_effect = (
        lambda bucket, key, destination: Path(destination).write_bytes(b"downloaded")
    )
    mock_client.get_object.return_value = {
        "Body": io.BytesIO('{"name": "알약"}'.encode("utf-8"))
    }
    paginator = Mock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "team/registry/b.json"},
                {"Key": "team/registry/a.json"},
            ]
        }
    ]
    mock_client.get_paginator.return_value = paginator
    storage = S3Storage("test-bucket", prefix="team", client=mock_client)

    destination = tmp_path / "download.txt"
    assert storage.download_file("datasets/source.txt", destination) == destination
    assert destination.read_bytes() == b"downloaded"
    assert storage.read_json("registry/item.json") == {"name": "알약"}
    assert storage.exists("registry/item.json")
    assert storage.list("registry/") == [
        "s3://test-bucket/team/registry/a.json",
        "s3://test-bucket/team/registry/b.json",
    ]
    mock_client.head_object.assert_called_once_with(
        Bucket="test-bucket", Key="team/registry/item.json"
    )
    paginator.paginate.assert_called_once_with(
        Bucket="test-bucket", Prefix="team/registry/"
    )


def test_s3_write_json_requires_explicit_overwrite():
    mock_client = Mock()
    storage = S3Storage("test-bucket", client=mock_client)

    storage.write_json("registry/item.json", {"name": "알약"})
    request = mock_client.put_object.call_args.kwargs
    assert request["IfNoneMatch"] == "*"
    assert json.loads(request["Body"].decode("utf-8")) == {"name": "알약"}

    storage.write_json("registry/item.json", {"name": "알약"}, overwrite=True)
    assert "IfNoneMatch" not in mock_client.put_object.call_args.kwargs


@pytest.mark.parametrize(
    ("error", "expected_exception"),
    (
        (NoCredentialsError(), CredentialsNotFoundError),
        (client_error("AccessDenied"), StorageAccessDeniedError),
        (client_error("NoSuchBucket"), BucketNotFoundError),
        (client_error("NoSuchKey"), ObjectNotFoundError),
        (client_error("PreconditionFailed"), ObjectAlreadyExistsError),
    ),
)
def test_s3_errors_are_converted_to_clear_storage_errors(
    tmp_path, error, expected_exception
):
    source = tmp_path / "source.txt"
    source.write_text("test", encoding="utf-8")
    mock_client = Mock()
    mock_client.put_object.side_effect = error
    storage = S3Storage("test-bucket", client=mock_client)

    with pytest.raises(expected_exception):
        storage.upload_file(source, "datasets/source.txt")


def test_s3_exists_returns_false_for_missing_object():
    mock_client = Mock()
    mock_client.head_object.side_effect = client_error("404", "HeadObject")
    storage = S3Storage("test-bucket", client=mock_client)

    assert storage.exists("datasets/missing.jpg") is False


def test_s3_uri_must_match_configured_bucket():
    storage = S3Storage("expected-bucket", client=Mock())

    with pytest.raises(StorageConfigurationError, match="bucket이 다릅니다"):
        storage.exists("s3://other-bucket/datasets/item.jpg")
