"""Dataset version으로 무효화되는 실행 간 S3 이미지 cache입니다."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from PIL import Image

from src.common import S3Storage, Storage, StorageError

from .errors import TrainError


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPOSITORY_ROOT / "artifacts" / "train-image-cache" / "v1"
TEMPORARY_ROOT = REPOSITORY_ROOT / "artifacts"
CACHE_TTL_SECONDS = 14 * 24 * 60 * 60
MAX_CACHE_BYTES = 50 * 1024**3
# cache 한 칸이 dataset 한 벌을 통째로 담습니다. version을 바꾸면 두 벌이 되고,
# 디스크는 그만큼 없습니다. 쓰고 있는 것만 남기고 나머지는 첫 이미지를 받기 전에
# 비웁니다. 지웠던 version으로 돌아가면 그때 다시 받습니다.
MAX_CACHE_DATASETS = 1
# 다 채운 cache를 묶어 두는 자리입니다. 이름이 fingerprint라 dataset이 바뀌면 다른
# 묶음이 되고, 옛 묶음을 실수로 쓰는 일이 없습니다.
ARCHIVE_PREFIX = "datasets/pill_detection/image-cache"
ARCHIVE_NAME = "image-cache.tar"
_VERSION_SEGMENT = re.compile(r"v[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_FILES = ("train_manifest.json", "validation_manifest.json")
_DOWNLOAD_IDENTITY_ATTEMPTS = 3


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


def _s3_object_identity(location: str, storage: Storage) -> str | None:
    """S3 객체 내용 변경을 구분할 수 있는 metadata를 정규화합니다."""

    if not isinstance(storage, S3Storage):
        return None
    parsed = urlsplit(location)
    if parsed.scheme.lower() != "s3" or parsed.netloc != storage.bucket:
        raise TrainError("image cache S3 URI does not match the configured bucket")
    key = unquote(parsed.path.lstrip("/"))
    try:
        metadata = storage.client.head_object(Bucket=parsed.netloc, Key=key)
    except Exception as error:
        raise TrainError("S3 image metadata lookup failed") from error

    identity = {
        name: metadata[name]
        for name in (
            "VersionId",
            "ETag",
            "ChecksumSHA256",
            "ChecksumSHA1",
            "ChecksumCRC32C",
            "ChecksumCRC32",
        )
        if isinstance(metadata.get(name), str) and metadata[name]
    }
    if not identity:
        raise TrainError("S3 image metadata has no content identity")
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _lock_stream(stream: Any, *, blocking: bool) -> bool:
    """열어 둔 file 하나를 process 사이에서 잠급니다. 닫으면 풀립니다.

    ``blocking``이 False면 다른 process가 이미 잡고 있을 때 기다리지 않고 ``False``를
    돌려줍니다. 그래서 "저 lease의 주인이 아직 살아 있는가"를 묻는 데 쓸 수 있습니다.
    """

    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(
                stream.fileno(),
                msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK,
                1,
            )
        else:
            import fcntl

            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
    except OSError:
        if blocking:
            raise
        return False
    return True


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    """Windows와 POSIX에서 cache 정리를 process 간 직렬화합니다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        _lock_stream(stream, blocking=True)
        yield


def _lease_is_abandoned(lease: Path) -> bool:
    """주인 process가 사라진 lease인지 봅니다.

    실행 중인 run은 자기 lease file을 연 채로 잠가 둡니다. 그래서 지금 잠글 수 있다는
    것은 그 run이 끝났거나 죽었다는 뜻입니다. 마지막으로 만진 시각을 대신 보면
    안 됩니다. 살아 있는 run도 첫 이미지를 받기 전 준비 작업에서 얼마든지 오래 조용할
    수 있고, 그 사이 다른 실행이 이 cache를 지워 버립니다.

    확인하지 못하면 살아 있다고 봅니다. 지우지 않아서 생기는 손해는 자리뿐입니다.
    """

    try:
        with lease.open("a+b") as stream:
            return _lock_stream(stream, blocking=False)
    except OSError:
        return False


def _safe_member_name(name: str) -> str:
    """묶음 안의 이름이 cache directory 밖을 가리키지 않는지 확인합니다.

    묶음은 bucket에서 옵니다. 이름 하나가 ``../``이면 푸는 쪽 파일이 조용히
    덮어써지므로, 하나라도 이상하면 전부 거부하고 평소대로 이미지를 받습니다.
    """

    if not name or name.startswith(("/", "\\")) or ":" in name:
        raise TrainError(f"image cache archive member escapes the cache: {name}")
    parts = name.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise TrainError(f"image cache archive member escapes the cache: {name}")
    return name


def _extract_archive(archive: Path, destination: Path) -> None:
    """cache 묶음을 풉니다. 파일과 directory 말고는 하나도 받지 않습니다."""

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r") as bundle:
        members = bundle.getmembers()
        for member in members:
            # symlink는 자기 밖을 가리킬 수 있고, 그 뒤에 풀리는 파일이 그 링크를
            # 통해 cache 밖에 써집니다. 이름 검사만으로는 막지 못합니다.
            if not (member.isfile() or member.isdir()):
                raise TrainError(
                    f"image cache archive member escapes the cache: {member.name}"
                )
            _safe_member_name(member.name)
        bundle.extractall(destination, members=members)


def _directory_size(path: Path) -> int:
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _is_valid_image(path: Path) -> bool:
    """형식 검사와 pixel 전체 decode가 모두 성공한 image인지 확인합니다."""

    try:
        with Image.open(path) as image:
            image.verify()
        # verify()는 JPEG pixel을 decode하지 않습니다. 같은 파일을 다시 열어야
        # 끝이 잘린 JPEG처럼 verify만 통과하는 손상도 잡을 수 있습니다.
        with Image.open(path) as image:
            image.load()
    except (
        OSError,
        SyntaxError,
        ValueError,
        EOFError,
        IndexError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return False
    return True


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
        max_datasets: int = MAX_CACHE_DATASETS,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._fingerprint = _dataset_fingerprint(dataset_summary)
        self._cache_root = cache_root
        self._temporary_root = temporary_root
        self._ttl_seconds = ttl_seconds
        self._max_cache_bytes = max_cache_bytes
        self._max_datasets = max_datasets
        self._now = now
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._lease: Path | None = None
        self._lease_handle: Any | None = None
        self._persistent = False
        self._object_identities: dict[str, str] = {}
        self._verified_entries: set[str] = set()
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
                # lease를 연 채로 잠급니다. 이 process가 어떻게 끝나든 OS가 풀어 주므로,
                # 다른 실행은 "이 run이 살아 있는가"를 시각이 아니라 잠금으로 묻습니다.
                self._lease_handle = self._lease.open("a+b")
                if not _lock_stream(self._lease_handle, blocking=False):
                    raise OSError(f"image cache lease is already held: {self._lease}")
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
        trash: list[Path] = []
        try:
            with _process_lock(self._cache_root / ".cleanup.lock"):
                self._remove_lease()
                (self.namespace / ".last_used").touch()
                trash = self._cleanup_locked()
        except OSError:
            # 남은 lease는 이 process가 끝날 때 OS가 잠금을 풀어 주므로, 다음 실행의
            # 정리가 주인 없는 lease로 보고 거둬 갑니다.
            return
        for path in trash:
            shutil.rmtree(path, ignore_errors=True)

    @property
    def archive_uri(self) -> str | None:
        """다 채운 cache가 통째로 올라가는 자리입니다.

        실행마다 새로 만드는 임시 cache는 다음 실행이 쓸 수 없으므로 ``None``입니다.
        """

        if not self._persistent or self._fingerprint is None:
            return None
        return f"{ARCHIVE_PREFIX}/{self._fingerprint}.tar"

    def _entry_names(self) -> list[str]:
        """이미 받아 둔 image entry 이름입니다. 받는 중인 임시 폴더는 뺍니다."""

        objects = self.namespace / "objects"
        if not objects.is_dir():
            return []
        return sorted(
            path.name
            for path in objects.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    def seed_from_archive(self, storage: Storage) -> bool:
        """묶음 하나를 받아 cache를 채웁니다.

        런타임이 바뀌면 디스크가 비어 있어 이미지를 한 장씩 다시 받게 됩니다. 같은
        1.9 GB라도 객체 하나로 받는 편이 훨씬 빠릅니다.

        여기서 실패해도 학습은 그대로 진행됩니다. 이건 빠른 길일 뿐이고, 없으면
        평소처럼 이미지를 한 장씩 받습니다.
        """

        location = self.archive_uri
        if location is None or not isinstance(storage, S3Storage):
            return False
        if self._entry_names():
            return False  # 이미 받아 둔 것이 있으면 건드리지 않습니다.
        staging = self.namespace / f".seed-{uuid4().hex}"
        try:
            if not storage.exists(location):
                return False
            self._temporary_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="train-cache-archive-", dir=self._temporary_root
            ) as scratch:
                archive = Path(scratch) / ARCHIVE_NAME
                storage.download_file(location, archive)
                _extract_archive(archive, staging)
            objects = self.namespace / "objects"
            objects.mkdir(parents=True, exist_ok=True)
            for entry in staging.iterdir():
                try:
                    entry.rename(objects / entry.name)
                except OSError:
                    # 같은 cache를 쓰는 다른 실행이 먼저 놓았습니다. 내용은 같습니다.
                    continue
            return True
        except (TrainError, StorageError, OSError, tarfile.TarError):
            return False
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def publish_archive(self, storage: Storage, *, expected_entries: int) -> bool:
        """다 채운 cache를 묶어 한 번만 올립니다.

        ``expected_entries``는 이 dataset의 이미지 수입니다. 그만큼 받기 전에 올리면
        다음 실행이 나머지가 영영 없는 묶음을 믿게 되므로, 다 찼을 때만 올립니다.
        """

        location = self.archive_uri
        if location is None or not isinstance(storage, S3Storage):
            return False
        names = self._entry_names()
        if not names or len(names) != expected_entries:
            return False
        objects = self.namespace / "objects"
        try:
            if storage.exists(location):
                return False  # 이미 다른 실행이 올렸습니다.
            self._temporary_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="train-cache-archive-", dir=self._temporary_root
            ) as scratch:
                archive = Path(scratch) / ARCHIVE_NAME
                with tarfile.open(archive, "w") as bundle:
                    for name in names:
                        bundle.add(objects / name, arcname=name)
                # 먼저 올린 실행의 묶음을 덮어쓰면, 그것을 받고 있던 쪽이 반쪽짜리를
                # 보게 됩니다. 진 쪽은 그냥 올리지 않습니다.
                storage.upload_file(archive, location, overwrite=False)
            return True
        except (StorageError, OSError, tarfile.TarError):
            return False

    def _start_temporary(self) -> None:
        self._persistent = False
        self._temporary_root.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="train-images-", dir=self._temporary_root
        )
        self.namespace = Path(self._temporary.name)

    def __getstate__(self) -> dict[str, Any]:
        """DataLoader worker에게 보낼 때 lease 손잡이는 빼고 보냅니다.

        열린 file은 pickle되지 않습니다. lease는 부모 process가 계속 들고 있으므로
        worker가 따로 들 이유도 없습니다.
        """

        return {**self.__dict__, "_lease_handle": None}

    def _remove_lease(self) -> None:
        # 잠금은 손잡이를 닫아야 풀립니다. 닫기 전에 지우면 Windows에서 지워지지도
        # 않고, 지워지더라도 잠금이 남아 다음 실행이 살아 있다고 읽습니다.
        if self._lease_handle is not None:
            self._lease_handle.close()
            self._lease_handle = None
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
        in_use = 0
        candidates: list[tuple[float, int, Path]] = []
        for namespace in self._cache_root.iterdir():
            if not namespace.is_dir() or namespace.name.startswith("."):
                continue
            active = namespace / ".active"
            if active.is_dir():
                for lease in active.glob("*.lease"):
                    if _lease_is_abandoned(lease):
                        lease.unlink(missing_ok=True)
                if any(active.glob("*.lease")):
                    in_use += 1
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
        # 오래 안 쓴 것부터 버립니다. 쓰고 있는 dataset은 lease가 지켜 주므로 여기
        # 남는 것은 아무도 쓰지 않는 것들뿐입니다.
        datasets = in_use + len(remaining)
        for _, size, namespace in sorted(remaining):
            if total <= self._max_cache_bytes and datasets <= self._max_datasets:
                break
            trash.append(self._move_to_trash(namespace))
            total -= size
            datasets -= 1
        return trash

    def _move_to_trash(self, namespace: Path) -> Path:
        destination = self._cache_root / f".trash-{uuid4().hex}"
        namespace.rename(destination)
        return destination

    def _touch(self) -> None:
        # 살아 있다는 표시는 lease 잠금이 합니다. 여기서 적는 시각은 오래 안 쓴
        # dataset부터 버리기 위한 순서일 뿐입니다.
        if not self._persistent:
            return
        now = self._now()
        try:
            os.utime(self.namespace / ".last_used", (now, now))
        except OSError:
            return

    def fetch(self, location: str, storage: Storage) -> Path:
        """S3 image를 완성된 directory 단위로 원자적으로 publish합니다."""

        self._touch()
        suffix = Path(urlsplit(location).path).suffix or ".image"
        objects = self.namespace / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        last_download_was_corrupt = False
        for _ in range(_DOWNLOAD_IDENTITY_ATTEMPTS):
            identity = self._object_identities.get(location)
            if identity is None:
                identity = _s3_object_identity(location, storage)
                if identity is not None:
                    self._object_identities[location] = identity
            cache_key = json.dumps(
                {"location": location, "object_identity": identity},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest = hashlib.sha256(cache_key).hexdigest()
            entry = objects / digest
            destination = entry / f"image{suffix}"
            if destination.is_file():
                if digest in self._verified_entries or _is_valid_image(destination):
                    self._verified_entries.add(digest)
                    return destination

            temporary = Path(
                tempfile.mkdtemp(prefix=f".{digest}-", suffix=".tmp", dir=objects)
            )
            downloaded = temporary / f"image{suffix}"
            try:
                storage.download_file(location, downloaded)
                if not _is_valid_image(downloaded):
                    last_download_was_corrupt = True
                    continue
                if identity is not None:
                    downloaded_identity = _s3_object_identity(location, storage)
                    if downloaded_identity != identity:
                        last_download_was_corrupt = False
                        if downloaded_identity is not None:
                            self._object_identities[location] = downloaded_identity
                        continue
                try:
                    temporary.rename(entry)
                except OSError:
                    if destination.is_file() and _is_valid_image(destination):
                        self._verified_entries.add(digest)
                        return destination
                    try:
                        downloaded.replace(destination)
                    except OSError as replace_error:
                        if not (
                            destination.is_file() and _is_valid_image(destination)
                        ):
                            raise TrainError("image cache repair failed") from replace_error
                if not destination.is_file():
                    raise TrainError("image cache publish failed")
                self._verified_entries.add(digest)
                return destination
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary, ignore_errors=True)
        if last_download_was_corrupt:
            raise TrainError("S3 image remained corrupt after repeated downloads")
        raise TrainError("S3 image changed repeatedly during cache download")
