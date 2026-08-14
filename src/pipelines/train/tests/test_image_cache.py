from __future__ import annotations

import gc
import io
import os
import pickle
import struct
import tarfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest
from PIL import Image

from src.common import S3Storage, StorageError
from src.pipelines.train import image_cache as image_cache_module
from src.pipelines.train.errors import TrainError
from src.pipelines.train.image_cache import ImageCacheSession


def _summary(
    version: int = 1,
    *,
    train_checksum: str = "a" * 64,
    validation_checksum: str = "b" * 64,
) -> dict:
    return {
        "source_prefix": f"datasets/pill_detection/raw/v{version}/",
        "split": {
            "checksums": {
                "algorithm": "sha256",
                "train_manifest.json": {"sha256": train_checksum, "bytes": 100},
                "validation_manifest.json": {
                    "sha256": validation_checksum,
                    "bytes": 80,
                },
            }
        },
    }


def _storage() -> Mock:
    storage = Mock()

    def download(source, destination, *, overwrite=False):
        assert source == "s3://bucket/images/pill.png"
        assert overwrite is False
        Image.new("RGB", (3, 2), color="red").save(destination)
        return Path(destination)

    storage.download_file.side_effect = download
    return storage


def _s3_cache_storage() -> S3Storage:
    """이미지 한 장과 cache 묶음 하나를 다루는 가짜 bucket입니다.

    ``published``에 올라간 것만 존재합니다. 실제 실행처럼 묶음도 이미지도 같은
    ``download_file``로 내려받습니다.
    """

    client = Mock()
    client.head_object.return_value = {"ETag": '"pill-etag"'}
    storage = S3Storage("bucket", client=client)
    published: dict[str, bytes] = {}

    def download_file(source, destination, *, overwrite=False):
        destination = Path(destination)
        payload = published.get(str(source))
        if payload is None:
            Image.new("RGB", (3, 2), color="red").save(destination, format="PNG")
        else:
            destination.write_bytes(payload)
        return destination

    def upload_file(source, destination, *, overwrite=False):
        if not overwrite and str(destination) in published:
            raise StorageError(f"이미 있습니다: {destination}")
        published[str(destination)] = Path(source).read_bytes()
        return f"s3://bucket/{destination}"

    storage.exists = lambda location: str(location) in published
    storage.download_file = Mock(side_effect=download_file)
    storage.upload_file = Mock(side_effect=upload_file)
    storage.published = published
    return storage


IMAGES = (
    "s3://bucket/images/pill.png",
    "s3://bucket/images/tablet.png",
)


def _warm(cache_root: Path, temporary_root: Path, storage: S3Storage) -> None:
    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as cache:
        for location in IMAGES:
            cache.fetch(location, storage)
        cache.publish_archive(storage, expected_entries=len(IMAGES))


def test_publish_never_replaces_an_archive_another_run_already_made(tmp_path):
    storage = _s3_cache_storage()
    _warm(tmp_path / "first", tmp_path / "temporary", storage)
    uploads = storage.upload_file.call_count

    _warm(tmp_path / "second", tmp_path / "temporary", storage)

    # 덮어쓰면 아직 그 묶음을 받고 있던 실행이 반쪽짜리를 보게 됩니다.
    assert storage.upload_file.call_count == uploads
    assert all(
        call.kwargs["overwrite"] is False for call in storage.upload_file.call_args_list
    )


def test_half_filled_cache_is_never_published(tmp_path):
    storage = _s3_cache_storage()

    with ImageCacheSession(
        _summary(),
        cache_root=tmp_path / "cache",
        temporary_root=tmp_path / "temporary",
    ) as cache:
        cache.fetch(IMAGES[0], storage)
        # 두 장짜리 dataset인데 한 장만 받았습니다. 이대로 올리면 다음 실행은
        # 나머지 한 장이 영영 없는 묶음을 믿게 됩니다.
        assert cache.publish_archive(storage, expected_entries=len(IMAGES)) is False

    assert storage.published == {}


def _png_with_corrupt_idat() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (20, 20), color="red").save(stream, format="PNG")
    payload = bytearray(stream.getvalue())
    position = 8
    while position < len(payload):
        length = struct.unpack(">I", payload[position : position + 4])[0]
        if payload[position + 4 : position + 8] == b"IDAT":
            payload[position + 8 + length // 2] ^= 1
            return bytes(payload)
        position += length + 12
    raise AssertionError("PNG fixture has no IDAT chunk")


def _png_without_idat() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (20, 20), color="red").save(stream, format="PNG")
    payload = stream.getvalue()
    position = 8
    while position < len(payload):
        length = struct.unpack(">I", payload[position : position + 4])[0]
        if payload[position + 4 : position + 8] == b"IDAT":
            return payload[:position] + payload[position + length + 12 :]
        position += length + 12
    raise AssertionError("PNG fixture has no IDAT chunk")


def _truncated_jpeg() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (100, 100), color="blue").save(stream, format="JPEG")
    return stream.getvalue()[:-2]


def test_repeated_corrupt_s3_download_never_publishes_a_cache_entry(tmp_path):
    storage = _storage()

    def download(source, destination, *, overwrite=False):
        Path(destination).write_bytes(_truncated_jpeg())
        return Path(destination)

    storage.download_file.side_effect = download
    with ImageCacheSession(
        _summary(),
        cache_root=tmp_path / "cache",
        temporary_root=tmp_path / "temporary",
    ) as cache:
        with pytest.raises(TrainError, match="remained corrupt"):
            cache.fetch(IMAGES[0], storage)
        objects = cache.namespace / "objects"
        assert not [path for path in objects.iterdir() if path.is_dir()]

    assert storage.download_file.call_count == image_cache_module._DOWNLOAD_IDENTITY_ATTEMPTS


@pytest.mark.parametrize(
    "corrupt_payload",
    (_png_with_corrupt_idat(), _png_without_idat(), _truncated_jpeg()),
    ids=("png-syntax-error", "png-index-error", "jpeg-load-error"),
)
def test_shared_archive_image_that_cannot_be_fully_decoded_is_downloaded_again(
    tmp_path, corrupt_payload
):
    storage = _s3_cache_storage()
    temporary_root = tmp_path / "temporary"

    with ImageCacheSession(
        _summary(), cache_root=tmp_path / "producer", temporary_root=temporary_root
    ) as producer:
        cached = producer.fetch(IMAGES[0], storage)
        cached.write_bytes(corrupt_payload)
        assert producer.publish_archive(storage, expected_entries=1) is True

    with ImageCacheSession(
        _summary(), cache_root=tmp_path / "consumer", temporary_root=temporary_root
    ) as consumer:
        assert consumer.seed_from_archive(storage) is True
        repaired = consumer.fetch(IMAGES[0], storage)
        with Image.open(repaired) as image:
            image.load()

    source_downloads = [
        call
        for call in storage.download_file.call_args_list
        if call.args[0] == IMAGES[0]
    ]
    assert len(source_downloads) == 2


def test_corrupt_s3_download_is_retried_before_cache_publish(tmp_path):
    storage = _storage()

    def download(source, destination, *, overwrite=False):
        assert source == IMAGES[0]
        assert overwrite is False
        if storage.download_file.call_count == 1:
            Path(destination).write_bytes(_truncated_jpeg())
        else:
            Image.new("RGB", (3, 2), color="red").save(destination, format="PNG")
        return Path(destination)

    storage.download_file.side_effect = download
    with ImageCacheSession(
        _summary(),
        cache_root=tmp_path / "cache",
        temporary_root=tmp_path / "temporary",
    ) as cache:
        repaired = cache.fetch(IMAGES[0], storage)
        with Image.open(repaired) as image:
            image.load()

    assert storage.download_file.call_count == 2


@pytest.mark.parametrize(
    "name",
    ["../escaped.png", "/etc/escaped.png", "C:/escaped.png", "..\\escaped.png"],
)
def test_archive_member_named_outside_the_cache_is_refused(tmp_path, name):
    archive = tmp_path / "image-cache.tar"
    payload = io.BytesIO(b"payload")
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo(name)
        info.size = len(payload.getvalue())
        bundle.addfile(info, payload)

    # 묶음은 bucket에서 옵니다. 이름 하나로 cache 밖 파일이 덮어써질 수 있습니다.
    with pytest.raises(TrainError, match="escapes"):
        image_cache_module._extract_archive(archive, tmp_path / "objects")

    assert not (tmp_path / "escaped.png").exists()


def test_archive_symlink_member_is_refused(tmp_path):
    archive = tmp_path / "image-cache.tar"
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo("objects/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../escaped"
        bundle.addfile(info)

    # 이름은 멀쩡합니다. 링크가 가리키는 곳이 cache 밖이라 이름 검사로는 못 막습니다.
    with pytest.raises(TrainError, match="escapes"):
        image_cache_module._extract_archive(archive, tmp_path / "objects")


def test_temporary_cache_has_no_archive_to_share(tmp_path):
    storage = _s3_cache_storage()

    # fingerprint를 만들 수 없는 dataset은 실행마다 다른 임시 폴더를 씁니다.
    # 다음 실행이 쓸 수 없는 묶음을 올리면 bucket만 커집니다.
    with ImageCacheSession(
        {"train_images": 1},
        cache_root=tmp_path / "cache",
        temporary_root=tmp_path / "temporary",
    ) as cache:
        cache.fetch(IMAGES[0], storage)
        assert cache.archive_uri is None
        assert cache.seed_from_archive(storage) is False
        assert cache.publish_archive(storage, expected_entries=1) is False

    assert storage.published == {}


def test_versioned_s3_cache_is_reused_and_version_change_invalidates_it(tmp_path):
    storage = _storage()
    cache_root = tmp_path / "persistent"
    temporary_root = tmp_path / "temporary"

    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as first:
        first_path = first.fetch("s3://bucket/images/pill.png", storage)
    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as second:
        second_path = second.fetch("s3://bucket/images/pill.png", storage)
    with ImageCacheSession(
        _summary(version=2), cache_root=cache_root, temporary_root=temporary_root
    ) as changed:
        changed_path = changed.fetch("s3://bucket/images/pill.png", storage)

    assert first_path == second_path
    assert changed_path != first_path
    assert storage.download_file.call_count == 2


def test_manifest_checksum_change_invalidates_the_cache(tmp_path):
    storage = _storage()
    cache_root = tmp_path / "persistent"
    temporary_root = tmp_path / "temporary"

    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as first:
        first.fetch("s3://bucket/images/pill.png", storage)
    with ImageCacheSession(
        _summary(train_checksum="c" * 64),
        cache_root=cache_root,
        temporary_root=temporary_root,
    ) as changed:
        changed.fetch("s3://bucket/images/pill.png", storage)

    assert storage.download_file.call_count == 2


def test_s3_object_identity_change_invalidates_the_cached_image(tmp_path):
    client = Mock()
    client.head_object.side_effect = [
        {"ETag": '"old-etag"', "VersionId": "old-version"},
        {"ETag": '"old-etag"', "VersionId": "old-version"},
        {"ETag": '"old-etag"', "VersionId": "old-version"},
        {"ETag": '"new-etag"', "VersionId": "new-version"},
        {"ETag": '"new-etag"', "VersionId": "new-version"},
    ]

    def download(bucket, key, destination):
        color = "red" if client.download_file.call_count == 1 else "blue"
        Image.new("RGB", (3, 2), color=color).save(destination, format="BMP")

    client.download_file.side_effect = download
    storage = S3Storage("bucket", client=client)
    cache_root = tmp_path / "persistent"
    temporary_root = tmp_path / "temporary"

    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as first:
        first_path = first.fetch("s3://bucket/images/pill.png", storage)
    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as unchanged:
        unchanged_path = unchanged.fetch("s3://bucket/images/pill.png", storage)
    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as changed:
        changed_path = changed.fetch("s3://bucket/images/pill.png", storage)

    assert first_path == unchanged_path
    assert changed_path != first_path
    assert first_path.stat().st_size == changed_path.stat().st_size
    with Image.open(first_path) as first_image, Image.open(changed_path) as changed_image:
        assert first_image.getpixel((0, 0)) == (255, 0, 0)
        assert changed_image.getpixel((0, 0)) == (0, 0, 255)
    assert client.download_file.call_count == 2


def test_legacy_summary_uses_a_fresh_temporary_cache_each_run(tmp_path):
    storage = _storage()
    cache_root = tmp_path / "persistent"
    temporary_root = tmp_path / "temporary"

    with ImageCacheSession(
        {"train_images": 1},
        cache_root=cache_root,
        temporary_root=temporary_root,
    ) as first:
        first_path = first.fetch("s3://bucket/images/pill.png", storage)
        assert first_path.is_file()
    assert not first_path.exists()
    with ImageCacheSession(
        {"train_images": 1},
        cache_root=cache_root,
        temporary_root=temporary_root,
    ) as second:
        second.fetch("s3://bucket/images/pill.png", storage)

    assert storage.download_file.call_count == 2
    assert not cache_root.exists()


def test_cache_handle_is_picklable_for_dataloader_workers(tmp_path):
    storage = _storage()

    with ImageCacheSession(
        {"train_images": 1},
        cache_root=tmp_path / "persistent",
        temporary_root=tmp_path / "temporary",
    ) as cache:
        worker_cache = pickle.loads(pickle.dumps(cache))
        path = worker_cache.fetch("s3://bucket/images/pill.png", storage)
        del worker_cache
        gc.collect()
        assert cache.namespace.exists()

    assert path.name == "image.png"


def test_a_leased_cache_handle_is_picklable_for_dataloader_workers(tmp_path):
    """lease를 든 session도 worker에게 보낼 수 있어야 합니다.

    열린 file은 pickle되지 않습니다. 보낼 때 손잡이를 빼지 않으면 num_workers를 올린
    학습이 첫 batch에서 죽습니다. 위 test는 fingerprint가 없는 임시 cache를 쓰므로
    손잡이가 아예 생기지 않아 이 경우를 잡지 못합니다.
    """

    storage = _storage()
    with ImageCacheSession(
        _summary(),
        cache_root=tmp_path / "persistent",
        temporary_root=tmp_path / "temporary",
    ) as cache:
        worker_cache = pickle.loads(pickle.dumps(cache))
        assert worker_cache.fetch(IMAGES[0], storage).is_file()


def test_process_lock_failure_falls_back_to_temporary_cache(tmp_path, monkeypatch):
    class BrokenLock:
        def __enter__(self):
            raise OSError("lock unavailable")

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(image_cache_module, "_process_lock", lambda path: BrokenLock())
    cache_root = tmp_path / "persistent"

    with ImageCacheSession(
        _summary(),
        cache_root=cache_root,
        temporary_root=tmp_path / "temporary",
    ) as cache:
        assert cache.namespace.parent == tmp_path / "temporary"

    assert not cache.namespace.exists()


def test_failed_download_never_publishes_a_partial_cache_entry(tmp_path):
    storage = _storage()

    def fail(source, destination, *, overwrite=False):
        Path(destination).write_bytes(b"partial")
        raise StorageError("download interrupted")

    storage.download_file.side_effect = fail
    cache_root = tmp_path / "persistent"
    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=tmp_path / "temporary"
    ) as cache:
        with pytest.raises(StorageError, match="interrupted"):
            cache.fetch("s3://bucket/images/pill.png", storage)

        objects = cache.namespace / "objects"
        assert not [path for path in objects.iterdir() if path.is_dir()]


def test_concurrent_downloads_publish_one_complete_entry(tmp_path):
    barrier = Barrier(2)
    storage = _storage()

    def download(source, destination, *, overwrite=False):
        barrier.wait(timeout=5)
        Image.new("RGB", (3, 2), color="blue").save(destination)
        return Path(destination)

    storage.download_file.side_effect = download
    cache_root = tmp_path / "persistent"
    temporary_root = tmp_path / "temporary"

    def fetch() -> Path:
        with ImageCacheSession(
            _summary(), cache_root=cache_root, temporary_root=temporary_root
        ) as cache:
            return cache.fetch("s3://bucket/images/pill.png", storage)

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(lambda _: fetch(), range(2)))

    assert paths[0] == paths[1]
    with Image.open(paths[0]) as image:
        assert image.size == (3, 2)
    assert storage.download_file.call_count == 2
    assert not list((paths[0].parents[1]).glob(".*.tmp"))


def _inactive_namespace(root: Path, name: str, *, used_at: float, size: int) -> Path:
    namespace = root / name
    namespace.mkdir(parents=True)
    (namespace / ".last_used").touch()
    os.utime(namespace / ".last_used", (used_at, used_at))
    (namespace / "payload").write_bytes(b"x" * size)
    return namespace


def test_cleanup_removes_expired_cache_but_protects_an_active_namespace(tmp_path):
    now = 20 * 24 * 60 * 60.0
    cache_root = tmp_path / "persistent"
    expired = _inactive_namespace(cache_root, "expired", used_at=0.0, size=4)
    active = _inactive_namespace(cache_root, "active", used_at=0.0, size=4)

    with _held_lease(active):
        with ImageCacheSession(
            _summary(),
            cache_root=cache_root,
            temporary_root=tmp_path / "temporary",
            now=lambda: now,
        ):
            pass

    assert not expired.exists()
    assert active.exists()


@contextmanager
def _held_lease(namespace: Path, name: str = "other-run.lease"):
    """다른 실행이 들고 있는 lease를 흉내 냅니다.

    같은 process의 다른 손잡이도 서로 잠금을 막으므로, 살아 있는 run을 test 안에서
    그대로 만들 수 있습니다.
    """

    active = namespace / ".active"
    active.mkdir(parents=True, exist_ok=True)
    lease = active / name
    with lease.open("a+b") as stream:
        assert image_cache_module._lock_stream(stream, blocking=False)
        yield lease


def test_a_lease_no_process_holds_stops_protecting_the_cache(tmp_path):
    """주인이 사라진 lease는 자리를 지켜 주지 않습니다.

    실행 중인 run은 자기 lease file을 연 채로 잠가 둡니다. 잠글 수 있다는 것은
    그 run이 끝났거나 죽었다는 뜻입니다.
    """

    now = 20 * 24 * 60 * 60.0
    cache_root = tmp_path / "persistent"
    abandoned = _inactive_namespace(cache_root, "abandoned", used_at=0.0, size=4)
    (abandoned / ".active").mkdir()
    (abandoned / ".active" / "dead-run.lease").touch()

    with ImageCacheSession(
        _summary(),
        cache_root=cache_root,
        temporary_root=tmp_path / "temporary",
        now=lambda: now,
    ):
        pass

    assert not abandoned.exists()


def test_a_held_lease_protects_its_cache_however_long_it_is_quiet(tmp_path):
    """살아 있는 run은 오래 조용해도 지켜집니다.

    첫 이미지를 받기 전 준비 작업은 얼마든지 오래 걸릴 수 있습니다. lease를 마지막으로
    만진 시각으로 판단하면 그 사이에 시작한 다른 실행이 살아 있는 cache를 지웁니다.
    """

    now = 20 * 24 * 60 * 60.0
    cache_root = tmp_path / "persistent"
    working = _inactive_namespace(cache_root, "working", used_at=0.0, size=4)

    with _held_lease(working) as lease:
        os.utime(lease, (0.0, 0.0))  # 아주 오래 조용했지만 주인은 살아 있습니다.
        with ImageCacheSession(
            _summary(),
            cache_root=cache_root,
            temporary_root=tmp_path / "temporary",
            now=lambda: now,
        ) as cache:
            # 정리가 터져 임시 cache로 물러나면 이 test가 엉뚱한 이유로 통과합니다.
            assert cache.namespace.parent == cache_root
        assert lease.exists()

    assert working.exists()


def test_starting_a_run_keeps_only_the_dataset_it_uses(tmp_path):
    """dataset을 바꾸면 앞 version의 cache는 첫 이미지를 받기 전에 사라집니다."""

    now = 1_000.0
    cache_root = tmp_path / "persistent"
    previous = _inactive_namespace(cache_root, "previous", used_at=now, size=4)

    with ImageCacheSession(
        _summary(version=6),
        cache_root=cache_root,
        temporary_root=tmp_path / "temporary",
        ttl_seconds=1_000_000.0,
        now=lambda: now,
    ) as cache:
        assert not previous.exists()
        assert cache.namespace.is_dir()


def test_cleanup_enforces_size_limit_oldest_first(tmp_path):
    cache_root = tmp_path / "persistent"
    oldest = _inactive_namespace(cache_root, "oldest", used_at=100.0, size=8)
    newer = _inactive_namespace(cache_root, "newer", used_at=200.0, size=8)

    with ImageCacheSession(
        _summary(),
        cache_root=cache_root,
        temporary_root=tmp_path / "temporary",
        max_cache_bytes=10,
        max_datasets=3,
        ttl_seconds=1_000.0,
        now=lambda: 500.0,
    ):
        pass

    assert not oldest.exists()
    assert newer.exists()


def test_cleanup_enforces_size_limit_after_current_lease_is_released(tmp_path):
    cache_root = tmp_path / "persistent"

    with ImageCacheSession(
        _summary(),
        cache_root=cache_root,
        temporary_root=tmp_path / "temporary",
        max_cache_bytes=10,
        ttl_seconds=1_000.0,
        now=lambda: 500.0,
    ) as cache:
        namespace = cache.namespace
        (namespace / "payload").write_bytes(b"x" * 20)
        assert namespace.exists()

    assert not namespace.exists()
