from __future__ import annotations

import gc
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock

import pytest
from PIL import Image

from src.common import StorageError
from src.pipelines.train import image_cache as image_cache_module
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
    (active / ".active").mkdir()
    (active / ".active" / "other-run.lease").touch()
    os.utime(active / ".active" / "other-run.lease", (now, now))

    with ImageCacheSession(
        _summary(),
        cache_root=cache_root,
        temporary_root=tmp_path / "temporary",
        now=lambda: now,
    ):
        pass

    assert not expired.exists()
    assert active.exists()


def test_cleanup_enforces_size_limit_oldest_first(tmp_path):
    cache_root = tmp_path / "persistent"
    oldest = _inactive_namespace(cache_root, "oldest", used_at=100.0, size=8)
    newer = _inactive_namespace(cache_root, "newer", used_at=200.0, size=8)

    with ImageCacheSession(
        _summary(),
        cache_root=cache_root,
        temporary_root=tmp_path / "temporary",
        max_cache_bytes=10,
        ttl_seconds=1_000.0,
        now=lambda: 500.0,
    ):
        pass

    assert not oldest.exists()
    assert newer.exists()
