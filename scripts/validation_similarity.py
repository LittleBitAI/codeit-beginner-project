#!/usr/bin/env python3
"""검증 세트를 믿어도 되는지 숫자로 답합니다.

전처리된 dataset의 validation crop 하나하나에 대해, **같은 class의 train crop 중
가장 비슷한 것까지의 거리**를 잽니다. 그 거리가 0에 가까우면 model은 validation에서
새로운 것을 보지 않습니다. 그런 검증 세트의 점수는 높게 나오지만 대회 점수와
무관합니다.

    python scripts/validation_similarity.py --dataset v4-seed42-8020-group
    python scripts/validation_similarity.py --dataset v3-seed42-8020-group --sample 400

**이미지를 내려받습니다.** 전수로 돌리면 dataset 전체를 받으므로 시간과 전송 비용이
듭니다. 먼저 ``--sample``로 감을 잡는 편이 낫습니다. 받은 이미지는 crop 특징만 남기고
바로 지우므로 디스크에 쌓이지 않습니다.

거리는 0~255 척도의 평균 픽셀 차이입니다. 3보다 작으면 사람 눈으로 두 그림을 구분할
수 없습니다.
"""

from __future__ import annotations

import argparse
import collections
import json
import posixpath
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common import StorageError, create_storage  # noqa: E402


PROCESSED_PREFIX = "datasets/pill_detection/processed"

#: crop을 이 크기로 맞춰 비교합니다. 알약이 놓인 위치와 크기가 조금 달라도 같은
#: 그림인지 보려는 것이므로, 원본 해상도로 비교하면 안 됩니다.
THUMBNAIL_SIZE = 32

#: 0~255 척도의 평균 픽셀 차이입니다. 3 아래는 눈으로 구분되지 않습니다.
DEFAULT_THRESHOLDS = (1.0, 3.0, 5.0, 10.0)

#: 이 거리보다 가까운 validation crop이 이 비율을 넘으면 검증 세트를 믿을 수 없습니다.
VERDICT_THRESHOLD = 3.0
VERDICT_RATIO = 0.5

DOWNLOAD_WORKERS = 16


class ValidationSimilarityError(RuntimeError):
    """팀원이 그대로 읽고 고칠 수 있는 오류입니다."""


def thumbnail(image: Image.Image) -> np.ndarray:
    """crop 하나를 위치·크기와 무관한 비교용 벡터로 만듭니다."""

    resized = image.convert("RGB").resize(
        (THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32).reshape(-1)


def crop_thumbnail(image: Image.Image, bbox: Sequence[float]) -> np.ndarray:
    """annotation 하나가 가리키는 자리를 잘라 비교용 벡터로 만듭니다."""

    x, y, width, height = (int(round(float(value))) for value in bbox)
    if width <= 0 or height <= 0:
        raise ValidationSimilarityError(f"넓이가 0인 bounding box입니다: {list(bbox)}")
    return thumbnail(image.crop((x, y, x + width, y + height)))


def nearest_same_class(
    validation: Sequence[Mapping[str, Any]], train: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """validation crop마다 같은 class의 가장 비슷한 train crop까지의 거리입니다.

    다른 class와 비슷한 것은 누수가 아니라 그냥 닮은 알약입니다. 같은 class 안에서만
    봅니다. train에 그 class가 없으면 잴 수 없으므로 ``inf``로 둡니다. 0으로 두면
    누수가 심한 것처럼 보입니다.
    """

    banks: dict[Any, np.ndarray] = {}
    grouped: dict[Any, list[np.ndarray]] = collections.defaultdict(list)
    for row in train:
        grouped[row["category_id"]].append(np.asarray(row["thumb"], dtype=np.float32))
    for category, vectors in grouped.items():
        banks[category] = np.stack(vectors)

    distances = np.full(len(validation), np.inf, dtype=np.float32)
    for index, row in enumerate(validation):
        bank = banks.get(row["category_id"])
        if bank is None:
            continue
        vector = np.asarray(row["thumb"], dtype=np.float32)
        distances[index] = np.abs(bank - vector).mean(axis=1).min()
    return distances


def build_report(
    distances: np.ndarray,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    sampled: bool = False,
) -> dict[str, Any]:
    """거리 분포를 사람이 읽는 판정으로 바꿉니다.

    ``sampled``는 일부 이미지만 보고 잰 결과라는 뜻입니다. 그때 나온 비율은 **하한**
    입니다. 비교 대상 train crop이 줄면 가장 가까운 것도 멀어지기 때문입니다. 실제로
    v4를 250장으로 재면 37%, 전수로 재면 79%가 나왔습니다. 그래서 표본 결과로는
    "써도 된다"고 말하지 않습니다.
    """

    comparable = distances[np.isfinite(distances)]
    ratios: dict[str, float | None] = {}
    for threshold in thresholds:
        ratios[str(float(threshold))] = (
            float((comparable < threshold).mean()) if comparable.size else None
        )
    near = ratios.get(str(float(VERDICT_THRESHOLD)))
    if near is None:
        verdict = "train에 같은 class가 없어 비교할 수 없습니다."
    elif near >= VERDICT_RATIO:
        verdict = (
            f"validation crop의 {near:.0%}가 train의 같은 class 그림과 구분되지 않습니다. "
            "이 검증 점수로 model을 고르면 안 됩니다."
        )
    elif sampled:
        verdict = (
            f"표본에서 {near:.0%}입니다. 표본은 비교 대상이 적어 실제보다 낮게 나오므로 "
            "이 값은 하한입니다. 판정하려면 --sample 없이 전수로 다시 재세요."
        )
    else:
        verdict = (
            f"validation crop의 {near:.0%}만 train 그림과 구분되지 않습니다. "
            "검증 점수를 비교에 쓸 수 있습니다."
        )
    return {
        "validation_crops": int(distances.size),
        "comparable_crops": int(comparable.size),
        "near_duplicate_ratio": ratios,
        "median_distance": float(np.median(comparable)) if comparable.size else None,
        "p90_distance": (
            float(np.percentile(comparable, 90)) if comparable.size else None
        ),
        "verdict": verdict,
    }


def collect_crops(
    storage: Any, manifest_uri: str, *, sample: int | None, label: str
) -> list[dict[str, Any]]:
    """manifest 한 개의 crop 특징을 모읍니다. 이미지는 받는 즉시 지웁니다."""

    manifest = storage.read_json(manifest_uri)
    locations = {item["id"]: str(item["file_name"]) for item in manifest["images"]}
    by_image: dict[Any, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for annotation in manifest["annotations"]:
        by_image[annotation["image_id"]].append(annotation)

    image_ids = sorted(locations)
    if sample is not None and sample < len(image_ids):
        # 앞에서부터 자르면 한 조합에 몰립니다. 고르게 건너뛰며 고릅니다.
        step = len(image_ids) / sample
        image_ids = [image_ids[int(index * step)] for index in range(sample)]

    rows: list[dict[str, Any]] = []
    lock = threading.Lock()
    progress = {"done": 0}

    def handle(image_id: Any) -> None:
        uri = locations[image_id]
        with tempfile.TemporaryDirectory(prefix="similarity-") as scratch:
            local = Path(scratch) / posixpath.basename(uri)
            storage.download_file(uri, local)
            with Image.open(local) as opened:
                opened.load()
                found = [
                    {
                        "category_id": annotation["category_id"],
                        "thumb": crop_thumbnail(opened, annotation["bbox"]),
                    }
                    for annotation in by_image[image_id]
                ]
        with lock:
            rows.extend(found)
            progress["done"] += 1
            if progress["done"] % 200 == 0:
                print(f"  {label} {progress['done']}/{len(image_ids)}장", flush=True)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        list(pool.map(handle, image_ids))
    print(f"  {label}: 이미지 {len(image_ids)}장에서 crop {len(rows)}개", flush=True)
    return rows


def describe(dataset: str, report: Mapping[str, Any]) -> str:
    """읽을 사람이 결론부터 보게 씁니다."""

    lines = [
        f"dataset: {dataset}",
        "",
        f"  {report['verdict']}",
        "",
        f"  validation crop      {report['validation_crops']}개"
        f" (비교 가능 {report['comparable_crops']}개)",
        f"  같은 class 최근접 거리  중앙 {report['median_distance']:.2f}"
        f" | p90 {report['p90_distance']:.2f}",
        "",
        "  거리별 비율 (0~255 척도의 평균 픽셀 차이, 3 아래는 눈으로 구분 불가)",
    ]
    for threshold, ratio in report["near_duplicate_ratio"].items():
        shown = "잴 수 없음" if ratio is None else f"{ratio:.1%}"
        lines.append(f"    < {threshold:>5}  {shown}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="전처리된 dataset의 validation이 train과 얼마나 비슷한지 잽니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="전처리 dataset 이름")
    parser.add_argument(
        "--sample",
        type=int,
        help="각 split에서 볼 이미지 수. 주지 않으면 전수입니다.",
    )
    parser.add_argument("--output", help="결과를 JSON으로 남길 경로")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    bucket = (os.environ.get("PILL_STORAGE_S3_BUCKET") or "").strip()
    if not bucket:
        print(
            "오류: PILL_STORAGE_S3_BUCKET 환경 변수가 없습니다.", file=sys.stderr
        )
        return 1
    if args.sample is not None and args.sample < 1:
        print("오류: --sample은 1 이상이어야 합니다.", file=sys.stderr)
        return 1

    base = f"s3://{bucket}/{PROCESSED_PREFIX}/{args.dataset}"
    try:
        storage = create_storage({"storage": {"backend": "s3", "s3": {"prefix": ""}}})
        train = collect_crops(
            storage, f"{base}/train_manifest.json", sample=args.sample, label="train"
        )
        validation = collect_crops(
            storage,
            f"{base}/validation_manifest.json",
            sample=args.sample,
            label="validation",
        )
    except StorageError as error:
        print(
            f"오류: dataset을 읽지 못했습니다({type(error).__name__}). "
            "이름과 AWS 자격 증명을 확인해 주세요.",
            file=sys.stderr,
        )
        return 1
    except ValidationSimilarityError as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1

    report = build_report(
        nearest_same_class(validation, train), sampled=args.sample is not None
    )
    report["dataset"] = args.dataset
    report["train_crops"] = len(train)
    report["sampled_images"] = args.sample
    print()
    print(describe(args.dataset, report))
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n결과를 남겼습니다: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
