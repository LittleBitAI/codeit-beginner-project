from __future__ import annotations

import gc
import io
import os
import pickle
import struct
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
        assert source in IMAGES
        assert overwrite is False
        Image.new("RGB", (3, 2), color="red").save(destination)
        return Path(destination)

    storage.download_file.side_effect = download
    return storage


IMAGES = (
    "s3://bucket/images/pill.png",
    "s3://bucket/images/tablet.png",
)


def test_prefetch_fills_the_cache_and_skips_what_it_already_has(tmp_path):
    """학습 전에 전부 받아 두되, 이어서 하는 실행은 남은 것만 받습니다."""

    storage = _storage()
    cache_root = tmp_path / "persistent"
    temporary_root = tmp_path / "temporary"

    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as interrupted:
        interrupted.fetch(IMAGES[0], storage)

    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as resumed:
        assert resumed.prefetch(IMAGES, storage) == len(IMAGES)
        # 두 장 중 한 장은 앞 실행이 이미 받아 두었습니다.
        assert storage.download_file.call_count == len(IMAGES)
        for location in IMAGES:
            assert resumed.fetch(location, storage).is_file()
        assert storage.download_file.call_count == len(IMAGES)


def test_prefetch_downloads_more_than_one_image_at_a_time(tmp_path):
    """한 장씩 차례로 받으면 첫 epoch이 이미지 수만큼 기다립니다."""

    # 두 장이 동시에 오지 않으면 여기서 기다리다 시간이 초과됩니다.
    barrier = Barrier(len(IMAGES), timeout=5)
    storage = _storage()

    def download(source, destination, *, overwrite=False):
        barrier.wait()
        Image.new("RGB", (3, 2), color="red").save(destination, format="PNG")
        return Path(destination)

    storage.download_file.side_effect = download
    with ImageCacheSession(
        _summary(),
        cache_root=tmp_path / "persistent",
        temporary_root=tmp_path / "temporary",
    ) as cache:
        assert cache.prefetch(IMAGES, storage) == len(IMAGES)


def test_prefetch_leaves_an_image_it_cannot_get_to_the_training_loop(tmp_path):
    """한 장을 못 받아도 학습은 시작합니다. 그 자리에서 다시 받아 보게 둡니다."""

    storage = _storage()

    def download(source, destination, *, overwrite=False):
        if source == IMAGES[1]:
            raise StorageError("이미지를 받지 못했습니다")
        Image.new("RGB", (3, 2), color="red").save(destination, format="PNG")
        return Path(destination)

    storage.download_file.side_effect = download
    with ImageCacheSession(
        _summary(),
        cache_root=tmp_path / "persistent",
        temporary_root=tmp_path / "temporary",
    ) as cache:
        assert cache.prefetch(IMAGES, storage) == 1
        assert cache.fetch(IMAGES[0], storage).is_file()


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
def test_cached_image_that_cannot_be_fully_decoded_is_downloaded_again(
    tmp_path, corrupt_payload
):
    storage = _storage()
    cache_root = tmp_path / "persistent"
    temporary_root = tmp_path / "temporary"

    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as first:
        # 받아 둔 뒤 디스크에서 깨진 파일입니다. 그대로 쓰면 학습이 그 batch에서
        # 멈추므로, 다음 실행이 알아보고 다시 받아 고쳐야 합니다.
        first.fetch(IMAGES[0], storage).write_bytes(corrupt_payload)

    with ImageCacheSession(
        _summary(), cache_root=cache_root, temporary_root=temporary_root
    ) as second:
        repaired = second.fetch(IMAGES[0], storage)
        with Image.open(repaired) as image:
            image.load()

    assert storage.download_file.call_count == 2


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


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork는 POSIX에만 있습니다")
def test_a_forked_worker_closing_its_copy_does_not_release_the_lease(tmp_path):
    """worker가 끝나도 부모 run의 lease는 풀리지 않아야 합니다.

    POSIX DataLoader는 fork로 worker를 만들고, worker는 부모가 열어 둔 lease fd를
    복제해 갖습니다. worker가 끝나며 그 복제본을 닫을 때 잠금까지 풀린다면, 그 순간
    시작한 다른 실행이 살아 있는 cache를 통째로 지웁니다.
    """

    lease = tmp_path / "run.lease"
    with lease.open("a+b") as stream:
        assert image_cache_module._lock_stream(stream, blocking=False)
        child = os.fork()
        if child == 0:  # worker: 물려받은 fd를 그대로 둔 채 끝납니다.
            os._exit(0)
        os.waitpid(child, 0)

        assert not image_cache_module._lease_is_abandoned(lease)

    # 대조군입니다. 주인이 놓으면 같은 검사가 버려진 것으로 읽어야 합니다.
    assert image_cache_module._lease_is_abandoned(lease)


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
