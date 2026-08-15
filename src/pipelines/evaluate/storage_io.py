"""Evaluate pipeline이 artifact를 읽고 쓸 때 사용하는 입출력 helper입니다.

`s3://`로 시작하는 URI는 `src/common/storage.py`의 공통 storage interface로
처리하고, 그 외 경로는 저장소 root 기준 상대 경로로 보고 local filesystem에서
직접 처리합니다. 개별 pipeline은 boto3를 직접 사용하지 않습니다.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.common import LocalStorage, Storage, StorageError, create_storage

from .errors import ArtifactWriteError, InputArtifactError


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def is_remote_uri(uri: str | os.PathLike[str]) -> bool:
    """S3 URI인지 확인합니다."""
    return os.fspath(uri).lower().startswith("s3://")


def join_uri(base: str, name: str) -> str:
    """URI 또는 상대 경로 뒤에 file name을 붙입니다."""
    return f"{base.rstrip('/')}/{name}"


class ArtifactStore:
    """Local 경로와 S3 URI를 같은 방식으로 다루는 얇은 입출력 계층입니다."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        storage: Storage | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        self._config = dict(config or {})
        self._storage = storage
        self.repository_root = (
            Path(repository_root).resolve() if repository_root is not None else REPOSITORY_ROOT
        )

    @property
    def storage(self) -> Storage:
        """S3 URI를 처리할 때만 생성하는 공통 storage backend입니다."""
        if self._storage is None:
            try:
                self._storage = create_storage(self._config)
            except StorageError as error:
                raise InputArtifactError(f"S3 storage 설정을 준비하지 못했습니다: {error}") from error
        return self._storage

    def local_path(self, uri: str) -> Path:
        """Local URI를 저장소 root 기준으로 해석하고 root 밖 경로를 막습니다."""
        candidate = Path(uri).expanduser()
        absolute = candidate if candidate.is_absolute() else self.repository_root / candidate
        resolved = absolute.resolve()
        root = self.repository_root.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise InputArtifactError(
                f"저장소 root 밖의 local 경로는 사용할 수 없습니다: {uri}"
            ) from error
        return resolved

    def normalize_uri(self, uri: str | Path) -> str:
        """저장 결과를 계약 형식(S3 URI 또는 저장소 기준 상대 경로)으로 바꿉니다."""
        raw_uri = os.fspath(uri)
        if is_remote_uri(raw_uri):
            return raw_uri
        path = Path(raw_uri).expanduser()
        absolute = path if path.is_absolute() else (self.repository_root / path)
        try:
            return absolute.resolve().relative_to(self.repository_root).as_posix()
        except ValueError:
            return absolute.resolve().as_posix()

    def artifact_identity(self, uri: str) -> tuple[str, ...]:
        """저장 계층이 보는 대상을 그대로 돌려줍니다.

        두 산출물이 같은 파일에 저장되는지 쓰기 전에 판정하려는 쪽이 씁니다. 표기
        규칙을 여기서 다시 만들지 않고 backend에 물어봅니다 — 밖에서 지어낸 규칙은
        backend가 실제로 하는 해석과 어긋납니다.

        local은 이 저장소 root 기준으로 봅니다. 쓸 때 `local_path`가 쓰는 것과 같은
        기준이라야 판정과 실제 저장이 어긋나지 않습니다. 저장소 밖 경로는 다른
        입출력과 똑같이 `local_path`가 먼저 막습니다.
        """
        if is_remote_uri(uri):
            try:
                return self.storage.identity(uri)
            except StorageError as error:
                raise InputArtifactError(
                    f"저장 위치를 확인하지 못했습니다 ({uri}): {error}"
                ) from error

        path = self.local_path(uri)
        try:
            return LocalStorage(self.repository_root).identity(path)
        except StorageError as error:
            raise InputArtifactError(
                f"저장 위치를 확인하지 못했습니다 ({uri}): {error}"
            ) from error

    def exists(self, uri: str) -> bool:
        """대상 artifact가 이미 있는지 확인합니다."""
        if is_remote_uri(uri):
            try:
                return bool(self.storage.exists(uri))
            except StorageError as error:
                raise InputArtifactError(f"S3 object 확인에 실패했습니다 ({uri}): {error}") from error
        return self.local_path(uri).is_file()

    def read_text(self, uri: str) -> str:
        """UTF-8 text artifact를 읽습니다."""
        if is_remote_uri(uri):
            with tempfile.TemporaryDirectory(prefix="pill-evaluate-") as directory:
                return self._read_local_text(self.ensure_local_file(uri, directory), uri)
        return self._read_local_text(self.local_path(uri), uri)

    @staticmethod
    def _read_local_text(path: Path, uri: str) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise InputArtifactError(f"artifact가 없습니다: {uri}") from error
        except UnicodeDecodeError as error:
            raise InputArtifactError(f"UTF-8 text가 아닙니다: {uri}") from error
        except OSError as error:
            raise InputArtifactError(f"artifact를 읽지 못했습니다 ({uri}): {error}") from error

    def read_json(self, uri: str) -> Any:
        """UTF-8 JSON artifact를 읽습니다."""
        raw_text = self.read_text(uri)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise InputArtifactError(f"유효한 JSON이 아닙니다: {uri}") from error

    def ensure_local_file(self, uri: str, download_dir: str | Path) -> Path:
        """Local file 경로를 반환하고, S3 object는 임시 directory로 내려받습니다."""
        if is_remote_uri(uri):
            destination = Path(download_dir) / Path(uri.split("/")[-1] or "artifact").name
            try:
                return self.storage.download_file(uri, destination, overwrite=True)
            except StorageError as error:
                raise InputArtifactError(f"artifact를 내려받지 못했습니다 ({uri}): {error}") from error

        path = self.local_path(uri)
        if not path.is_file():
            raise InputArtifactError(f"artifact가 없습니다: {uri}")
        return path

    def write_json(self, uri: str, value: Any, *, overwrite: bool = False) -> str:
        """JSON artifact를 UTF-8 without BOM, LF로 저장하고 계약 형식 URI를 반환합니다."""
        if is_remote_uri(uri):
            try:
                return self.storage.write_json(uri, value, overwrite=overwrite)
            except StorageError as error:
                raise ArtifactWriteError(f"artifact를 저장하지 못했습니다 ({uri}): {error}") from error

        path = self.local_path(uri)
        if path.exists() and not overwrite:
            raise ArtifactWriteError(
                f"artifact가 이미 있습니다. overwrite를 허용해야 덮어씁니다: {self.normalize_uri(uri)}"
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as output:
                json.dump(value, output, ensure_ascii=False, indent=2)
                output.write("\n")
        except OSError as error:
            raise ArtifactWriteError(f"artifact를 저장하지 못했습니다 ({uri}): {error}") from error
        return self.normalize_uri(path)

    def write_text(self, uri: str, value: str, *, overwrite: bool = False) -> str:
        """Text artifact를 UTF-8 without BOM, LF로 저장합니다."""
        if is_remote_uri(uri):
            with tempfile.TemporaryDirectory(prefix="pill-evaluate-upload-") as directory:
                source = Path(directory) / Path(uri.split("/")[-1] or "artifact.txt").name
                try:
                    source.write_text(value, encoding="utf-8", newline="\n")
                    return self.storage.upload_file(source, uri, overwrite=overwrite)
                except StorageError as error:
                    raise ArtifactWriteError(
                        f"artifact를 저장하지 못했습니다 ({uri}): {error}"
                    ) from error
                except OSError as error:
                    raise ArtifactWriteError(
                        f"임시 text artifact를 만들지 못했습니다 ({uri}): {error}"
                    ) from error

        path = self.local_path(uri)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if overwrite else "x"
            with path.open(mode, encoding="utf-8", newline="\n") as output:
                output.write(value)
        except FileExistsError as error:
            raise ArtifactWriteError(
                "artifact가 이미 있습니다. overwrite를 허용해야 덮어씁니다: "
                f"{self.normalize_uri(uri)}"
            ) from error
        except OSError as error:
            raise ArtifactWriteError(f"artifact를 저장하지 못했습니다 ({uri}): {error}") from error
        return self.normalize_uri(path)

    def remove_local(self, uri: str) -> None:
        """이번 실행에서 만든 local artifact만 정리합니다. S3 object는 삭제하지 않습니다."""
        if is_remote_uri(uri):
            return
        try:
            self.local_path(uri).unlink(missing_ok=True)
        except (OSError, InputArtifactError):
            return


__all__ = ["ArtifactStore", "REPOSITORY_ROOT", "is_remote_uri", "join_uri"]
