#!/usr/bin/env python3
"""Colab에서 학습을 돌릴 config 파일 하나를 만듭니다.

팀원이 JSON을 손으로 쓰지 않게 하는 것이 목적입니다. 고른 값을 인자로 받아
dataset이 S3에 실제로 있는지 확인한 뒤 config를 쓰고, 다음에 칠 명령까지
알려 줍니다.

    python scripts/make_colab_config.py --list-datasets
    python scripts/make_colab_config.py --dataset v1-seed42-8020 --epochs 30 --batch-size 2

**train의 기본값을 여기에 베끼지 않습니다.** 고른 값과 Colab 실행에 필요한 값만
적고 나머지는 train이 가진 기본값에 맡깁니다. 두 곳에 같은 숫자를 두면 언젠가
어긋나고, 화면에는 실제로 쓰인 값과 다른 값이 보이게 됩니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common import StorageError, create_storage  # noqa: E402


# 전처리 결과가 올라가는 자리입니다. data pipeline이 정한 규칙을 그대로 씁니다.
PROCESSED_PREFIX = "datasets/pill_detection/processed"

# train이 config.inputs.data에서 요구하는 네 가지입니다. 왼쪽이 S3 파일 이름,
# 오른쪽이 config key입니다.
ARTIFACT_KEYS = {
    "train_manifest": "train_manifest_uri",
    "validation_manifest": "validation_manifest_uri",
    "class_map": "class_map_uri",
    "dataset_summary": "dataset_summary_uri",
}
REQUIRED_ARTIFACTS = tuple(ARTIFACT_KEYS)

# checkpoint가 올라갈 자리입니다. registry record와 실험 목록이 이 배치를 씁니다.
DEFAULT_OUTPUT_PREFIX = "experiments/completed"

# train의 run-id 규칙과 같습니다. 어긋나면 train이 실행을 거부합니다.
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ColabConfigError(RuntimeError):
    """팀원이 그대로 읽고 고칠 수 있는 오류입니다."""


def resolve_bucket(environment: Mapping[str, str]) -> str:
    """어느 bucket을 쓸지 정합니다. 없으면 무엇을 설정해야 하는지 알려 줍니다."""

    bucket = (environment.get("PILL_STORAGE_S3_BUCKET") or "").strip()
    if not bucket:
        raise ColabConfigError(
            "PILL_STORAGE_S3_BUCKET 환경 변수가 없습니다. 팀 bucket 이름을 넣어 주세요.\n"
            '  import os; os.environ["PILL_STORAGE_S3_BUCKET"] = "<팀 bucket>"'
        )
    return bucket


def generate_run_id() -> str:
    """Colab에서 돈 실행임을 이름으로 알 수 있게 만듭니다.

    train은 같은 run_id의 결과가 이미 있으면 덮어쓰지 않고 거부합니다. 시각만으로는
    Windows의 시계 해상도가 거칠어 연달아 부르면 같은 값이 나오므로 짧은 무작위
    꼬리를 붙입니다. 이름일 뿐이라 재현성은 seed가 정합니다.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"colab-{stamp}-{uuid4().hex[:4]}"


def dataset_artifact_uris(bucket: str, dataset: str) -> dict[str, str]:
    """dataset 이름 하나에서 필요한 artifact URI 네 개를 만듭니다."""

    base = f"s3://{bucket}/{PROCESSED_PREFIX}/{dataset}"
    return {name: f"{base}/{name}.json" for name in REQUIRED_ARTIFACTS}


def available_datasets(storage: Any, bucket: str) -> list[str]:
    """bucket에 올라와 있는 전처리 dataset 이름을 모읍니다."""

    prefix = f"s3://{bucket}/{PROCESSED_PREFIX}/"
    names = set()
    for location in storage.list(prefix):
        remainder = location[len(prefix) :].strip("/")
        if "/" in remainder:
            names.add(remainder.split("/", 1)[0])
    return sorted(names)


def inspect_datasets(storage: Any, bucket: str) -> list[tuple[str, list[str]]]:
    """dataset 이름마다 빠진 필수 artifact를 함께 돌려줍니다.

    prefix만 있고 artifact가 없는 이름(예: 상위 폴더)이 목록에 섞이면 팀원이
    그것을 고르게 됩니다. 무엇이 없어서 못 쓰는지 함께 보여 주기 위해서입니다.
    """

    report = []
    for name in available_datasets(storage, bucket):
        uris = dataset_artifact_uris(bucket, name)
        missing = [key for key, uri in uris.items() if not storage.exists(uri)]
        report.append((name, missing))
    return report


def build_config(
    *,
    storage: Any,
    bucket: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    architecture: str,
    optimizer: str,
    output: Path,
    device: str = "cuda",
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
    learning_rate: float | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """확인을 마친 config를 만듭니다. 파일로 쓰지는 않습니다."""

    uris = dataset_artifact_uris(bucket, dataset)
    missing = [uri for uri in uris.values() if not storage.exists(uri)]
    if missing:
        listed = "\n".join(f"  - {uri}" for uri in sorted(missing))
        raise ColabConfigError(
            f"dataset '{dataset}'의 artifact를 S3에서 찾지 못했습니다.\n{listed}\n"
            "--list-datasets로 올라와 있는 이름을 확인해 주세요."
        )

    train: dict[str, Any] = {
        "run_id": run_id or generate_run_id(),
        "architecture": architecture,
        "optimizer": optimizer,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "output_prefix": output_prefix,
    }
    if learning_rate is not None:
        train["learning_rate"] = learning_rate

    if not RUN_ID_PATTERN.match(train["run_id"]):
        raise ColabConfigError(
            f"run_id '{train['run_id']}'는 train이 받는 형식이 아닙니다. "
            "영문·숫자로 시작하고 . _ - 만 쓸 수 있습니다."
        )

    return {
        "project": {"name": "pill-object-detection"},
        "execution": {"mode": "colab"},
        "storage": {"backend": "s3", "s3": {"prefix": ""}},
        "train": train,
        "inputs": {"data": {ARTIFACT_KEYS[name]: uris[name] for name in REQUIRED_ARTIFACTS}},
    }


def write_config(destination: Path, config: Mapping[str, Any]) -> None:
    """config를 UTF-8(BOM 없음)로 씁니다."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def describe(destination: Path, config: Mapping[str, Any]) -> str:
    """무엇이 만들어졌고 다음에 무엇을 쳐야 하는지 알려 줍니다."""

    train = config["train"]
    lines = [
        f"config를 만들었습니다: {destination}",
        "",
        f"  실행 이름   {train['run_id']}",
        f"  모델        {train['architecture']}",
        f"  Optimizer   {train['optimizer']}",
        f"  Epochs      {train['epochs']}",
        f"  Batch size  {train['batch_size']}",
        f"  Device      {train['device']}",
        f"  결과 위치   s3://.../{train['output_prefix']}/{train['run_id']}/",
        "",
        "여기 없는 값은 train의 기본값을 씁니다.",
        "",
        "다음 명령으로 학습을 시작하세요.",
        f"  python -m src.main_pipeline --only train --config {destination}",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab 학습용 config를 만듭니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="S3에 올라와 있는 전처리 dataset 이름만 보여 주고 끝냅니다.",
    )
    parser.add_argument("--dataset", help="쓸 전처리 dataset 이름 (예: v1-seed42-8020)")
    parser.add_argument("--epochs", type=int, help="전체 학습 데이터를 몇 번 반복할지")
    parser.add_argument("--batch-size", type=int, help="한 번에 처리할 이미지 수")
    parser.add_argument(
        "--architecture",
        help="쓸 detection architecture. 지원 목록은 src/pipelines/train/model.py에 있습니다.",
    )
    parser.add_argument("--optimizer", help="AdamW, SGD, Adam 중 하나")
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="주지 않으면 train이 optimizer별 기본값을 씁니다.",
    )
    parser.add_argument("--run-id", help="비워 두면 colab-<시각>으로 자동으로 만듭니다.")
    parser.add_argument("--device", default="cuda", help="기본값 cuda")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--output",
        default="artifacts/colab/train.json",
        help="만들 config 경로. 기본값 artifacts/colab/train.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    try:
        bucket = resolve_bucket(os.environ)
        storage = create_storage({"storage": {"backend": "s3", "s3": {"prefix": ""}}})

        if args.list_datasets:
            report = inspect_datasets(storage, bucket)
            if not report:
                print(f"s3://{bucket}/{PROCESSED_PREFIX}/ 아래에 전처리 dataset이 없습니다.")
                return 1
            usable = [name for name, missing in report if not missing]
            broken = [(name, missing) for name, missing in report if missing]
            if usable:
                print("쓸 수 있는 dataset:")
                for name in usable:
                    print(f"  {name}")
            else:
                print("쓸 수 있는 dataset이 없습니다.")
            if broken:
                print("\n쓸 수 없는 이름 (필수 artifact가 없습니다):")
                for name, missing in broken:
                    print(f"  {name} — {', '.join(sorted(missing))} 없음")
            return 0 if usable else 1

        required = {
            "--dataset": args.dataset,
            "--epochs": args.epochs,
            "--batch-size": args.batch_size,
            "--architecture": args.architecture,
            "--optimizer": args.optimizer,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            raise ColabConfigError(
                f"다음 값이 필요합니다: {', '.join(missing)}\n"
                "결과를 바꾸는 값이라 조용한 기본값을 두지 않습니다."
            )

        destination = Path(args.output)
        config = build_config(
            storage=storage,
            bucket=bucket,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            architecture=args.architecture,
            optimizer=args.optimizer,
            output=destination,
            device=args.device,
            output_prefix=args.output_prefix,
            learning_rate=args.learning_rate,
            run_id=args.run_id,
        )
        write_config(destination, config)
        print(describe(destination, config))
    except ColabConfigError as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    except StorageError as error:
        print(
            f"오류: S3에 접근하지 못했습니다({type(error).__name__}). "
            "AWS 자격 증명과 region을 확인해 주세요.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
