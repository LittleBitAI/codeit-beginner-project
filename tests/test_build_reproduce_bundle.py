"""발표용 재현 번들을 모으는 script의 test입니다.

이 script가 조용히 잘못 만들면, 받은 사람은 자격 증명이 없어 원인을 확인할 길이 없는
자리에서 막힙니다. 그래서 **자격 증명 없이 열리는가**를 정하는 두 가지 — 경로를 바꿔
적는 것과 표본이 class를 덮는 것 — 만 확인합니다. S3에서 파일을 받아 오는 부분은
여기서 돌리지 않습니다.
"""

from __future__ import annotations

import json
import re

import pytest

from scripts.build_reproduce_bundle import (
    COVER_TIMES,
    FUSION_RUNS,
    REPRODUCE_DIR,
    BundleError,
    _inside_repository,
    choose_demo_groups,
    download_many,
    main,
    write_derived,
    retarget_fusion_inputs,
    write_test_manifest,
)


BUCKET = "s3://team-bucket"
RAW = f"{BUCKET}/datasets/pill_detection/raw/v5/original"


class FakeStore:
    """`read_json`만 있는 최소 storage입니다."""

    bucket = "team-bucket"

    def __init__(self, document: object) -> None:
        self._document = document

    def read_json(self, source: str) -> object:  # noqa: ARG002 - 하나만 읽습니다.
        return self._document


def manifest() -> dict[str, object]:
    return {
        "info": {"description": "test"},
        "images": [
            {"id": 1, "file_name": f"{RAW}/test_images/1.png", "width": 976, "height": 1280},
            {"id": 2, "file_name": f"{RAW}/test_images/2.png", "width": 976, "height": 1280},
        ],
        "annotations": [],
        "categories": [{"id": 250, "name": "가나정", "supercategory": "pill"}],
    }


def test_the_test_manifest_points_at_the_images_the_bundle_carries(tmp_path):
    """`s3://`가 적힌 채로 두면 자격 증명이 없는 사람은 시험지를 열지 못합니다."""

    written = write_test_manifest(FakeStore(manifest()), tmp_path, rebuilding=False)

    document = json.loads(written.read_text(encoding="utf-8"))
    assert [image["file_name"] for image in document["images"]] == [
        "../raw/v90/test_images/1.png",
        "../raw/v90/test_images/2.png",
    ]
    # 위치만 바꿉니다. id나 크기가 바뀌면 융합 입력과 이어지지 않습니다.
    assert [image["id"] for image in document["images"]] == [1, 2]
    assert document["images"][0]["width"] == 976
    assert document["categories"] == manifest()["categories"]
    # 실제로 그 자리에서 열립니다.
    assert (written.parent / document["images"][0]["file_name"]).as_posix().endswith(
        "raw/v90/test_images/1.png"
    )


def test_a_manifest_without_images_stops_the_build(tmp_path):
    with pytest.raises(BundleError):
        write_test_manifest(FakeStore({"images": []}), tmp_path, rebuilding=False)


def test_the_fusion_inputs_point_at_the_bundled_manifest(tmp_path):
    """evaluate는 합칠 예측이 적어 둔 시험지를 **실제로 읽어** 견줍니다.

    `s3://`가 적혀 있으면 그 확인에서 멈춥니다. 위치만 바꾸고 나머지는 그대로 둡니다.
    """

    directory = tmp_path / REPRODUCE_DIR / "fused"
    directory.mkdir(parents=True)
    for run in FUSION_RUNS:
        (directory / f"{run}.json").write_text(
            json.dumps(
                {
                    "run_id": run,
                    "test_manifest_uri": f"{BUCKET}/datasets/x/test_manifest.json",
                    "predictions": [{"image_id": 1, "score": 0.5}],
                }
            ),
            encoding="utf-8",
        )

    changed = retarget_fusion_inputs(tmp_path)

    assert len(changed) == len(FUSION_RUNS)
    for run in FUSION_RUNS:
        document = json.loads((directory / f"{run}.json").read_text(encoding="utf-8"))
        assert document["test_manifest_uri"] == f"{REPRODUCE_DIR}/test_manifest.json"
        assert document["run_id"] == run
        assert document["predictions"] == [{"image_id": 1, "score": 0.5}]

    # 두 번 돌려도 같은 결과이고, 바꿀 것이 없으면 다시 쓰지 않습니다.
    assert retarget_fusion_inputs(tmp_path) == []


def _entries(*names: str) -> list[str]:
    return [f"{RAW}/train_annotations/{name}_json/K-000001/{name}_0_2_0_2_70_000_200.json" for name in names]


def test_every_class_comes_from_more_than_one_group(tmp_path):
    """조합 하나뿐인 class는 검증에 못 갑니다. 나누는 단위가 조합 통째이기 때문입니다."""

    chosen = choose_demo_groups(
        _entries(
            "K-000001-000002-000003-000004",
            "K-000001-000002-000003-000005",
            "K-000004-000005-000006-000007",
            "K-000006-000007-000008-000009",
            "K-000008-000009-000001-000002",
            "K-000003-000004-000005-000006",
        ),
        wanted=0,
    )

    seen: dict[str, int] = {}
    for name in chosen:
        for code in re.findall(r"\d{6}", name):
            seen[code] = seen.get(code, 0) + 1
    assert seen, "조합을 하나도 고르지 못했습니다."
    assert min(seen.values()) >= COVER_TIMES


def test_a_source_without_any_combination_stops_the_build():
    with pytest.raises(BundleError):
        choose_demo_groups([f"{RAW}/train_annotations/notes.txt"], wanted=3)


@pytest.mark.parametrize("outside", ("../sabrefish-bundle", "../밖", "/tmp/번들"))
def test_a_destination_outside_the_repository_is_refused(outside: str):
    """`<repo>-bundle`은 `<repo>`로 시작하지만 저장소 밖입니다.

    문자열 접두사로 견주면 그 둘이 갈리지 않아, 번들이 저장소 밖에 쓰입니다.
    """
    with pytest.raises(BundleError):
        _inside_repository(outside, label="번들 폴더")


def test_a_destination_inside_the_repository_passes():
    assert _inside_repository("artifacts/reproduce-bundle", label="번들 폴더").is_absolute()


def test_a_file_already_there_stops_the_build_until_a_person_chooses(tmp_path):
    """건너뛰면 남의 파일이 서명을 받고, 덮어쓰면 있던 것이 사라집니다.

    둘 다 위험하므로 기본은 멈추고 사람이 고르게 합니다.
    """

    target = tmp_path / "이미있음.json"
    target.write_text("{}", encoding="utf-8")
    fetched: list[tuple[str, bool]] = []

    class Store:
        bucket = "team-bucket"

        def download_file(self, source, destination, *, overwrite=False):
            fetched.append((str(source), overwrite))

    pairs = [("key.json", target)]

    with pytest.raises(BundleError):
        download_many(Store(), pairs, label="test", resume=False, overwrite=False)
    assert fetched == []

    download_many(Store(), pairs, label="test", resume=True, overwrite=False)
    assert fetched == []

    download_many(Store(), pairs, label="test", resume=False, overwrite=True)
    assert fetched == [("s3://team-bucket/key.json", True)]


def test_a_missing_file_is_fetched_without_any_flag(tmp_path):
    fetched: list[str] = []

    class Store:
        bucket = "team-bucket"

        def download_file(self, source, destination, *, overwrite=False):
            fetched.append(str(source))

    download_many(
        Store(),
        [("key.json", tmp_path / "없음.json")],
        label="test",
        resume=False,
        overwrite=False,
    )

    assert fetched == ["s3://team-bucket/key.json"]


def test_a_derived_file_already_there_also_stops_the_build(tmp_path):
    """내려받는 파일만 막으면 이 둘이 그 검사를 지나 기본 실행이 지웁니다."""

    target = tmp_path / "SHA256SUMS"
    target.write_text("남의 것\n", encoding="utf-8")

    with pytest.raises(BundleError):
        write_derived(target, "새 것\n", rebuilding=False)
    assert target.read_text(encoding="utf-8") == "남의 것\n"

    # 다시 만들기로 한 실행은 옛 목록을 남기지 않습니다.
    write_derived(target, "새 것\n", rebuilding=True)
    assert target.read_text(encoding="utf-8") == "새 것\n"


def test_the_two_flags_cannot_be_given_together():
    """하나는 있는 것을 믿고 하나는 버립니다. 조용히 고르면 파괴적인 쪽이 이깁니다."""

    assert main(["--resume", "--overwrite"]) == 1
