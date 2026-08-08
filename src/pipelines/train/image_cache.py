"""Dataset version으로 무효화되는 실행 간 S3 이미지 cache입니다."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit
from uuid import uuid4

from PIL import Image

from src.common import Storage

from .errors import TrainError


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPOSITORY_ROOT / "artifacts" / "train-image-cache" / "v1"
TEMPORARY_ROOT = REPOSITORY_ROOT / "artifacts"
CACHE_TTL_SECONDS = 14 * 24 * 60 * 60
MAX_CACHE_BYTES = 50 * 1024**3
_VERSION_SEGMENT = re.compile(r"v[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FILES = ("train_manifest.json", "validation_manifest.json")


def _dataset_fingerprint(summary: Mapping[str, Any]) -> str | None:
    """Version과 두 manifest checksum이 완전할 때만 영구 cache key를 만듭니다."""

    source_prefix = summary.get("source_prefix")
    if not isinstance(source_prefix, str) or not any(
        _VERSION_SEGMENT.fullmatch(part) for part in source_prefix.strip("/").split("/")
    ):
        return None
    split = summary.get("split")
    checksums = split.get("checksums") if isinstance(split, Mapping) else None
    if not isinstance(checksums, Mapping) or checksums.get("algorithm") != "sha256":
        return None
    manifest_hashes: dict[str, str] = {}
    for name in _MANIFEST_FILES:
        record = checksums.get(name)
        digest = record.get("sha256") if isinstance(record, Mapping) else None
        size = record.get("bytes") if isinstance(record, Mapping) else None
        if (
            not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            return None
        manifest_hashes[name] = digest
    payload = {"source_prefix": source_prefix, "manifests": manifest_hashes}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    """Windows와 POSIX에서 cache 정리를 process 간 직렬화합니다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


class ImageCacheSession:
    """한 train run 동안 cache lease와 다운로드 publish를 관리합니다."""

    def __init__(
        self,
        dataset_summary: Mapping[str, Any],
        *,
        cache_root: Path = CACHE_ROOT,
        temporary_root: Path = TEMPORARY_ROOT,
        ttl_seconds: float = CACHE_TTL_SECONDS,
        max_cache_bytes: int = MAX_CACHE_BYTES,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._fingerprint = _dataset_fingerprint(dataset_summary)
        self._cache_root = cache_root
        self._temporary_root = temporary_root
        self._ttl_seconds = ttl_seconds
        self._max_cache_bytes = max_cache_bytes
        self._now = now
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._lease: Path | None = None
        self._persistent = False
        self.namespace = temporary_root

    def __enter__(self) -> ImageCacheSession:
        if self._fingerprint is None:
            self._start_temporary()
            return self
        trash: list[Path] = []
        try:
            self._cache_root.mkdir(parents=True, exist_ok=True)
            with _process_lock(self._cache_root / ".cleanup.lock"):
                self.namespace = self._cache_root / self._fingerprint
                active = self.namespace / ".active"
                active.mkdir(parents=True, exist_ok=True)
                self._lease = active / f"{os.getpid()}-{uuid4().hex}.lease"
                self._lease.touch(exist_ok=False)
                (self.namespace / ".last_used").touch()
                self._persistent = True
                trash = self._cleanup_locked()
        except OSError:
            try:
                self._remove_lease()
            except OSError:
                pass
            self._start_temporary()
            return self
        for path in trash:
            shutil.rmtree(path, ignore_errors=True)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
            return
        try:
            with _process_lock(self._cache_root / ".cleanup.lock"):
                self._remove_lease()
                (self.namespace / ".last_used").touch()
        except OSError:
            # 남은 lease는 TTL 이후 stale lease로 정리됩니다.
            return

    def _start_temporary(self) -> None:
        self._persistent = False
        self._temporary_root.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="train-images-", dir=self._temporary_root
        )
        self.namespace = Path(self._temporary.name)

    def _remove_lease(self) -> None:
        if self._lease is not None:
            self._lease.unlink(missing_ok=True)
            self._lease = None

    def _cleanup_locked(self) -> list[Path]:
        now = self._now()
        cutoff = now - self._ttl_seconds
        trash = [
            path
            for path in self._cache_root.iterdir()
            if path.is_dir() and path.name.startswith(".trash-")
        ]
        candidates: list[tuple[float, int, Path]] = []
        for namespace in self._cache_root.iterdir():
            if not namespace.is_dir() or namespace.name.startswith("."):
                continue
            active = namespace / ".active"
            if active.is_dir():
                for lease in active.glob("*.lease"):
                    if lease.stat().st_mtime < cutoff:
                        lease.unlink(missing_ok=True)
                if any(active.glob("*.lease")):
                    continue
            marker = namespace / ".last_used"
            used_at = marker.stat().st_mtime if marker.exists() else namespace.stat().st_mtime
            candidates.append((used_at, _directory_size(namespace), namespace))

        remaining: list[tuple[float, int, Path]] = []
        for used_at, size, namespace in candidates:
            if used_at < cutoff:
                trash.append(self._move_to_trash(namespace))
            else:
                remaining.append((used_at, size, namespace))

        total = sum(
            _directory_size(path)
            for path in self._cache_root.iterdir()
            if path.is_dir() and not path.name.startswith(".trash-")
        )
        for _, size, namespace in sorted(remaining):
            if total <= self._max_cache_bytes:
                break
            trash.append(self._move_to_trash(namespace))
            total -= size
        return trash

    def _move_to_trash(self, namespace: Path) -> Path:
        destination = self._cache_root / f".trash-{uuid4().hex}"
        namespace.rename(destination)
        return destination

    def _touch(self) -> None:
        if not self._persistent:
            return
        now = self._now()
        for marker in (self._lease, self.namespace / ".last_used"):
            if marker is None:
                continue
            try:
                os.utime(marker, (now, now))
            except OSError:
                continue

    def fetch(self, location: str, storage: Storage) -> Path:
        """S3 image를 완성된 directory 단위로 원자적으로 publish합니다."""

        self._touch()
        suffix = Path(urlsplit(location).path).suffix or ".image"
        digest = hashlib.sha256(location.encode("utf-8")).hexdigest()
        objects = self.namespace / "objects"
        entry = objects / digest
        destination = entry / f"image{suffix}"
        if destination.is_file():
            return destination

        objects.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{digest}-", suffix=".tmp", dir=objects)
        )
        downloaded = temporary / f"image{suffix}"
        try:
            storage.download_file(location, downloaded)
            with Image.open(downloaded) as image:
                image.verify()
            try:
                temporary.rename(entry)
            except OSError as error:
                if not destination.is_file():
                    raise TrainError("image cache publish failed") from error
            if not destination.is_file():
                raise TrainError("image cache publish failed")
            return destination
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
