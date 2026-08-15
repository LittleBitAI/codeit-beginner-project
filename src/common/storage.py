"""Local filesystem과 Amazon S3를 위한 공통 storage interface."""

from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
)


LOGICAL_S3_PREFIXES = (
    "datasets/",
    "experiments/uploading/",
    "experiments/completed/",
    "experiments/failed/",
    "registry/",
    "submissions/",
    "final-models/",
)


class StorageError(RuntimeError):
    """Storage 작업이 실패했을 때 사용하는 기본 예외입니다."""


class StorageConfigurationError(StorageError):
    """Storage 설정이 없거나 잘못된 경우입니다."""


class StorageDependencyError(StorageError):
    """선택한 backend의 dependency가 설치되지 않은 경우입니다."""


class CredentialsNotFoundError(StorageError):
    """AWS credential을 default credential chain에서 찾지 못한 경우입니다."""


class StorageAccessDeniedError(StorageError):
    """Storage 작업 권한이 거부된 경우입니다."""


class BucketNotFoundError(StorageError):
    """설정한 S3 bucket이 없는 경우입니다."""


class ObjectNotFoundError(StorageError):
    """요청한 local file 또는 S3 object가 없는 경우입니다."""


class ObjectAlreadyExistsError(StorageError):
    """명시적인 overwrite 허용 없이 대상이 이미 존재하는 경우입니다."""


class Storage(ABC):
    """LocalStorage와 S3Storage가 제공하는 공통 interface입니다."""

    @abstractmethod
    def upload_file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> str:
        """Local file을 storage에 업로드하고 대상 URI/path를 반환합니다."""

    @abstractmethod
    def download_file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Storage object를 local file로 다운로드합니다."""

    @abstractmethod
    def read_json(self, source: str | Path) -> Any:
        """UTF-8 JSON object를 읽습니다."""

    @abstractmethod
    def write_json(
        self,
        destination: str | Path,
        value: Any,
        *,
        overwrite: bool = False,
    ) -> str:
        """값을 UTF-8 JSON object로 저장합니다."""

    @abstractmethod
    def exists(self, location: str | Path) -> bool:
        """Object가 존재하는지 확인합니다."""

    @abstractmethod
    def list(self, prefix: str | Path = "") -> list[str]:
        """Prefix 아래 object URI/path를 반환합니다."""

    @abstractmethod
    def identity(self, location: str | Path) -> tuple[str, ...]:
        """이 backend가 **위치 표기를 해석한 결과**를 돌려줍니다.

        쓰기 전에 "두 산출물이 같은 곳에 저장되는가"를 판정하려는 쪽이, 스스로
        규칙을 지어내는 대신 이것을 물어보게 하려는 것입니다. 규칙을 밖에서 다시
        만들면 backend가 실제로 하는 해석과 어긋납니다.

        **약속하는 것:** 읽고 쓸 때와 같은 해석을 거친 뒤에도 같은 자리를 가리키는
        표기는 같은 값이 됩니다. backend가 다르면 절대 겹치지 않습니다.

        **이미 있는 파일**은 파일 시스템에 직접 물어봅니다. 그래서 hard link로 이어진
        두 경로와 volume마다 다르게 설정된 대소문자 규칙까지 가립니다. 덮어쓰기를 켠
        실행이 이 갈래로 들어옵니다.

        **아직 없는 파일**은 물어볼 대상이 없어 표기로만 판정합니다. 대소문자는 그
        platform의 기본 규칙(`os.path.normcase`)만 따르므로, volume 설정이 다르면
        가리지 못합니다.

        값의 모양은 약속하지 않습니다. 사람에게 보여 줄 이름이 아닙니다.
        """


class LocalStorage(Storage):
    """설정된 root 안에서 동작하는 local filesystem storage입니다."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root).expanduser().resolve()

    def _resolve(self, location: str | Path) -> Path:
        raw_location = os.fspath(location)
        if raw_location.lower().startswith("s3://"):
            raise StorageConfigurationError(
                "local backend에서는 s3:// URI를 사용할 수 없습니다."
            )

        candidate = Path(raw_location).expanduser()
        resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise StorageConfigurationError(
                f"local storage root 밖의 경로는 사용할 수 없습니다: {location}"
            ) from error
        return resolved

    @staticmethod
    def _raise_local_error(error: OSError, location: Path) -> None:
        if isinstance(error, FileNotFoundError):
            raise ObjectNotFoundError(f"local object가 없습니다: {location}") from error
        if isinstance(error, PermissionError):
            raise StorageAccessDeniedError(
                f"local object 접근 권한이 없습니다: {location}"
            ) from error
        raise StorageError(f"local storage 작업에 실패했습니다: {location}") from error

    def upload_file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> str:
        source_path = Path(source).expanduser()
        destination_path = self._resolve(destination)
        if not source_path.is_file():
            raise ObjectNotFoundError(f"업로드할 local file이 없습니다: {source_path}")
        if destination_path.exists() and not overwrite:
            raise ObjectAlreadyExistsError(
                f"local object가 이미 있습니다: {destination_path}"
            )

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.resolve() != destination_path:
                shutil.copy2(source_path, destination_path)
        except OSError as error:
            self._raise_local_error(error, destination_path)
        return str(destination_path)

    def download_file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        source_path = self._resolve(source)
        destination_path = Path(destination).expanduser()
        if not source_path.is_file():
            raise ObjectNotFoundError(f"local object가 없습니다: {source_path}")
        if destination_path.exists() and not overwrite:
            raise ObjectAlreadyExistsError(
                f"download 대상이 이미 있습니다: {destination_path}"
            )

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path != destination_path.resolve():
                shutil.copy2(source_path, destination_path)
        except OSError as error:
            self._raise_local_error(error, destination_path)
        return destination_path

    def read_json(self, source: str | Path) -> Any:
        source_path = self._resolve(source)
        try:
            return json.loads(source_path.read_text(encoding="utf-8"))
        except OSError as error:
            self._raise_local_error(error, source_path)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StorageError(f"유효한 UTF-8 JSON이 아닙니다: {source_path}") from error

    def write_json(
        self,
        destination: str | Path,
        value: Any,
        *,
        overwrite: bool = False,
    ) -> str:
        destination_path = self._resolve(destination)
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if overwrite else "x"
            with destination_path.open(mode, encoding="utf-8", newline="\n") as output:
                json.dump(value, output, ensure_ascii=False, indent=2)
                output.write("\n")
        except FileExistsError as error:
            raise ObjectAlreadyExistsError(
                f"local object가 이미 있습니다: {destination_path}"
            ) from error
        except OSError as error:
            self._raise_local_error(error, destination_path)
        return str(destination_path)

    def exists(self, location: str | Path) -> bool:
        return self._resolve(location).is_file()

    def identity(self, location: str | Path) -> tuple[str, ...]:
        # 읽고 쓸 때와 **같은** `_resolve`를 씁니다.
        path = self._resolve(location)
        try:
            status = path.stat()
        except OSError:
            # 아직 없는 파일은 물어볼 대상이 없어 표기로만 판정합니다. `normcase`는
            # 그 platform의 기본 대소문자 규칙입니다.
            return ("local-path", os.path.normcase(str(path)))
        # 이미 있는 파일은 filesystem에 직접 묻습니다. hard link로 이어진 두 경로와
        # volume마다 다르게 설정된 대소문자 규칙은 여기서만 가릴 수 있습니다.
        # `overwrite`를 켠 실행은 이미 있는 파일에 쓰므로 이 갈래로 들어옵니다.
        return ("local-file", str(status.st_dev), str(status.st_ino))

    def list(self, prefix: str | Path = "") -> list[str]:
        prefix_path = self._resolve(prefix)
        try:
            raw_prefix = os.fspath(prefix)
            if prefix_path == self.root:
                relative_prefix = ""
            else:
                relative_prefix = prefix_path.relative_to(self.root).as_posix()
                if raw_prefix.endswith(("/", "\\")):
                    relative_prefix += "/"
            if prefix_path.is_file():
                return [str(prefix_path)]
            return sorted(
                str(path)
                for path in self.root.rglob("*")
                if path.is_file()
                and path.relative_to(self.root).as_posix().startswith(relative_prefix)
            )
        except OSError as error:
            self._raise_local_error(error, prefix_path)


class S3Storage(Storage):
    """Default AWS credential chain을 사용하는 Amazon S3 storage입니다."""

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        profile: str | None = None,
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not bucket or "/" in bucket:
            raise StorageConfigurationError("유효한 S3 bucket 이름이 필요합니다.")
        self.bucket = bucket
        self.prefix = self._normalize_key(prefix, allow_empty=True).rstrip("/")
        self.profile = profile or None
        self.region = region or None
        self._provided_client = client
        self._cached_client: Any | None = None

    @staticmethod
    def _normalize_key(key: str, *, allow_empty: bool = False) -> str:
        normalized = key.replace("\\", "/").lstrip("/")
        parts = normalized.split("/") if normalized else []
        if any(part in {".", ".."} for part in parts):
            raise StorageConfigurationError("S3 key에는 '.' 또는 '..' segment를 사용할 수 없습니다.")
        if not normalized and not allow_empty:
            raise StorageConfigurationError("S3 object key가 필요합니다.")
        return normalized

    @property
    def client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        if self._cached_client is not None:
            return self._cached_client

        try:
            import boto3
        except ImportError as error:
            raise StorageDependencyError(
                "S3 backend 사용을 위해 requirements.txt의 boto3를 설치해야 합니다."
            ) from error

        try:
            session = boto3.Session(
                profile_name=self.profile,
                region_name=self.region,
            )
            self._cached_client = session.client("s3")
        except ProfileNotFound as error:
            raise CredentialsNotFoundError(
                "설정한 AWS profile을 찾을 수 없습니다. AWS_PROFILE과 SSO login 상태를 확인하세요."
            ) from error
        return self._cached_client

    def _resolve(self, location: str | Path, *, allow_empty: bool = False) -> tuple[str, str]:
        raw_location = os.fspath(location)
        if raw_location.lower().startswith("s3://"):
            parsed = urlsplit(raw_location)
            if parsed.scheme.lower() != "s3" or not parsed.netloc:
                raise StorageConfigurationError(f"유효한 S3 URI가 아닙니다: {location}")
            if parsed.query or parsed.fragment:
                raise StorageConfigurationError("S3 URI에는 query 또는 fragment를 사용할 수 없습니다.")
            if parsed.netloc != self.bucket:
                raise StorageConfigurationError(
                    "설정한 bucket과 S3 URI의 bucket이 다릅니다."
                )
            return parsed.netloc, self._normalize_key(
                unquote(parsed.path.lstrip("/")), allow_empty=allow_empty
            )

        key = self._normalize_key(raw_location, allow_empty=allow_empty)
        if self.prefix:
            key = f"{self.prefix}/{key}" if key else f"{self.prefix}/"
        return self.bucket, key

    @staticmethod
    def _uri(bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key}"

    @staticmethod
    def _raise_s3_error(
        error: Exception,
        *,
        bucket: str,
        key: str | None = None,
    ) -> None:
        location = f"s3://{bucket}/{key}" if key else f"s3://{bucket}"
        if isinstance(error, (NoCredentialsError, PartialCredentialsError, ProfileNotFound)):
            raise CredentialsNotFoundError(
                "AWS credential을 찾을 수 없습니다. AWS_PROFILE과 IAM Identity Center login 상태를 확인하세요."
            ) from error
        if isinstance(error, ClientError):
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"AccessDenied", "AccessDeniedException", "AllAccessDisabled", "403"}:
                raise StorageAccessDeniedError(f"S3 접근 권한이 거부되었습니다: {location}") from error
            if code == "NoSuchBucket":
                raise BucketNotFoundError(f"S3 bucket이 없습니다: {bucket}") from error
            if code in {"NoSuchKey", "NotFound", "404"}:
                raise ObjectNotFoundError(f"S3 object가 없습니다: {location}") from error
            if code in {"PreconditionFailed", "ConditionalRequestConflict", "412"}:
                raise ObjectAlreadyExistsError(f"S3 object가 이미 있습니다: {location}") from error
        raise StorageError(f"S3 작업에 실패했습니다: {location}") from error

    def upload_file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> str:
        source_path = Path(source).expanduser()
        if not source_path.is_file():
            raise ObjectNotFoundError(f"업로드할 local file이 없습니다: {source_path}")
        bucket, key = self._resolve(destination)
        request: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if not overwrite:
            request["IfNoneMatch"] = "*"
        try:
            with source_path.open("rb") as body:
                self.client.put_object(Body=body, **request)
        except (ClientError, NoCredentialsError, PartialCredentialsError, ProfileNotFound) as error:
            self._raise_s3_error(error, bucket=bucket, key=key)
        return self._uri(bucket, key)

    def download_file(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        bucket, key = self._resolve(source)
        destination_path = Path(destination).expanduser()
        if destination_path.exists() and not overwrite:
            raise ObjectAlreadyExistsError(
                f"download 대상이 이미 있습니다: {destination_path}"
            )
        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket, key, str(destination_path))
        except PermissionError as error:
            raise StorageAccessDeniedError(
                f"download 대상 경로에 쓸 권한이 없습니다: {destination_path}"
            ) from error
        except OSError as error:
            raise StorageError(
                f"download 대상 경로를 준비하지 못했습니다: {destination_path}"
            ) from error
        except (ClientError, NoCredentialsError, PartialCredentialsError, ProfileNotFound) as error:
            self._raise_s3_error(error, bucket=bucket, key=key)
        return destination_path

    def read_json(self, source: str | Path) -> Any:
        bucket, key = self._resolve(source)
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            raw_value = response["Body"].read()
            return json.loads(raw_value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise StorageError(f"유효한 UTF-8 JSON이 아닙니다: {self._uri(bucket, key)}") from error
        except (ClientError, NoCredentialsError, PartialCredentialsError, ProfileNotFound) as error:
            self._raise_s3_error(error, bucket=bucket, key=key)

    def write_json(
        self,
        destination: str | Path,
        value: Any,
        *,
        overwrite: bool = False,
    ) -> str:
        bucket, key = self._resolve(destination)
        body = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        request: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": body,
            "ContentType": "application/json; charset=utf-8",
        }
        if not overwrite:
            request["IfNoneMatch"] = "*"
        try:
            self.client.put_object(**request)
        except (ClientError, NoCredentialsError, PartialCredentialsError, ProfileNotFound) as error:
            self._raise_s3_error(error, bucket=bucket, key=key)
        return self._uri(bucket, key)

    def identity(self, location: str | Path) -> tuple[str, ...]:
        # 읽고 쓸 때와 **같은** `_resolve`를 씁니다. scheme 대소문자, percent
        # encoding, prefix 붙이기가 거기서 이미 정리되므로 여기서 다시 규칙을
        # 만들지 않습니다. `//`처럼 S3에서 실제로 다른 key는 다르게 남습니다.
        return ("s3", *self._resolve(location))

    def exists(self, location: str | Path) -> bool:
        bucket, key = self._resolve(location)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "NotFound", "404"}:
                return False
            self._raise_s3_error(error, bucket=bucket, key=key)
        except (NoCredentialsError, PartialCredentialsError, ProfileNotFound) as error:
            self._raise_s3_error(error, bucket=bucket, key=key)

    def list(self, prefix: str | Path = "") -> list[str]:
        bucket, key_prefix = self._resolve(prefix, allow_empty=True)
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)
            return sorted(
                self._uri(bucket, item["Key"])
                for page in pages
                for item in page.get("Contents", [])
            )
        except (ClientError, NoCredentialsError, PartialCredentialsError, ProfileNotFound) as error:
            self._raise_s3_error(error, bucket=bucket, key=key_prefix or None)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise StorageConfigurationError(f"{name} 설정은 object여야 합니다.")
    return value


def _value(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def create_storage(
    config: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    s3_client: Any | None = None,
) -> Storage:
    """Config와 환경 변수로 선택한 storage backend를 생성합니다.

    환경 변수가 config보다 우선합니다. AWS credential 자체는 받거나 저장하지
    않고 boto3의 default credential chain에 맡깁니다.
    """

    environment = os.environ if environ is None else environ
    root_config = _mapping(config, "config")
    storage_config = _mapping(root_config.get("storage"), "storage")
    local_config = _mapping(storage_config.get("local"), "storage.local")
    s3_config = _mapping(storage_config.get("s3"), "storage.s3")

    backend = (
        _value(environment.get("PILL_STORAGE_BACKEND"), storage_config.get("backend"))
        or "local"
    ).lower()
    if backend == "local":
        root = _value(
            environment.get("PILL_STORAGE_LOCAL_ROOT"),
            local_config.get("root"),
        ) or "artifacts"
        return LocalStorage(root)
    if backend != "s3":
        raise StorageConfigurationError(f"지원하지 않는 storage backend입니다: {backend}")

    bucket = _value(
        environment.get("PILL_STORAGE_S3_BUCKET"),
        s3_config.get("bucket"),
    )
    if bucket is None:
        raise StorageConfigurationError(
            "S3 backend에는 PILL_STORAGE_S3_BUCKET 또는 storage.s3.bucket이 필요합니다."
        )
    prefix = _value(
        environment.get("PILL_STORAGE_S3_PREFIX"),
        s3_config.get("prefix"),
    ) or ""
    profile = _value(environment.get("AWS_PROFILE"), s3_config.get("profile"))
    region = _value(
        environment.get("AWS_REGION"),
        environment.get("AWS_DEFAULT_REGION"),
        s3_config.get("region"),
    )
    return S3Storage(
        bucket,
        prefix=prefix,
        profile=profile,
        region=region,
        client=s3_client,
    )
