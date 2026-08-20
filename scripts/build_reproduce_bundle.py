"""발표용 재현 번들을 팀 S3에서 모아 저장소에 그대로 풀리는 한 덩어리로 만듭니다.

강사님과 멘토님은 팀 AWS 자격 증명이 없습니다. 그래서 최고 제출을 재현하는 데 실제로
필요한 것만 골라 공개 위치에 올릴 수 있는 크기로 묶습니다. 이 script는 **읽기만** 하고
S3에 아무것도 쓰지 않습니다.

    python -m scripts.build_reproduce_bundle --out artifacts/reproduce-bundle

번들 안의 경로는 저장소 root 기준입니다. 받는 쪽은 clone한 자리에서 풀기만 하면 됩니다.

    datasets/pill_detection/raw/v90/          데모 학습용 표본 원본 + 대회 test 이미지
    datasets/pill_detection/reproduce/        최고 제출(fusion-top3-ensemble) 재현 재료

`v90`은 발표 데모용 표본이라는 뜻으로 붙인 판 번호입니다. data pipeline이 산출물
directory 이름에 그대로 쓰므로(`v90-seed42-8020-group`), 팀이 쓰던 v1~v6과 절대 겹치지
않는 번호를 골랐습니다.

test 이미지는 **한 벌만** 둡니다. 데모 준비(`data.prepare`)는 원본 prefix 안에
`test_images/`가 있어야 하고, 재현 쪽 test manifest는 그 자리를 상대 경로로 가리킵니다.
1.5 GB짜리를 두 번 담지 않으려는 것입니다.

재학습은 하지 않습니다. 임베딩 checkpoint와 참조 crop 은행은 **최고 제출을 만든 그
파일 그대로**를 받아 담습니다. 참조 crop이 바뀌면 margin이 전부 바뀌어 다른 점수가
됩니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common import StorageError, create_storage  # noqa: E402


class BundleError(RuntimeError):
    """번들을 계속 만들 수 없을 때 냅니다."""


#: 번들 안에서 쓰는 저장소 기준 경로입니다.
DEMO_RAW_PREFIX = "datasets/pill_detection/raw/v90"
REPRODUCE_DIR = "datasets/pill_detection/reproduce"

#: 최고 제출이 실제로 읽은 S3 자리입니다. 하나라도 다른 파일을 담으면 재현이 아닙니다.
PROCESSED_V5 = "datasets/pill_detection/processed/v5-seed42-8020-group"
RAW_V5 = "datasets/pill_detection/raw/v5/original"
CROP_BANK_KEY = "datasets/pill_detection/crop-bank/20260817/crop-bank-contract.tar.gz"
EMBEDDING_PREFIX = "experiments/embeddings/20260817-contract"
FUSION_INPUT_PREFIX = "experiments/harvest-rescored"
#: 융합에 들어간 세 실행입니다. 순서가 결과를 바꾸지는 않지만 기록과 같게 둡니다.
FUSION_RUNS = ("dino-4675", "dino-b711", "dino-bfeb")
#: 검증 추론을 건너뛰려고 넘기는 예측입니다. 제출 CSV에는 영향을 주지 않습니다.
VALIDATION_PREDICTIONS_KEY = "experiments/ensemble/fusion-top3-rescored/predictions.json"
#: 대조용 원본입니다. 재현 결과를 이것과 견줍니다.
REFERENCE_SUBMISSION_KEY = "submissions/fusion-top3-ensemble/submission.csv"

#: 크기가 작아 그냥 통째로 받는 파일들입니다. (S3 key, 번들 안 경로)
FIXED_FILES: tuple[tuple[str, str], ...] = (
    (f"{PROCESSED_V5}/class_map.json", f"{REPRODUCE_DIR}/class_map.json"),
    (
        f"{PROCESSED_V5}/validation_manifest.json",
        f"{REPRODUCE_DIR}/validation_manifest.json",
    ),
    (VALIDATION_PREDICTIONS_KEY, f"{REPRODUCE_DIR}/validation_predictions.json"),
    (CROP_BANK_KEY, f"{REPRODUCE_DIR}/crop-bank-contract.tar.gz"),
    (REFERENCE_SUBMISSION_KEY, f"{REPRODUCE_DIR}/reference-submission.csv"),
    *(
        (f"{EMBEDDING_PREFIX}/{name}.pt", f"{REPRODUCE_DIR}/embeddings/{name}.pt")
        for name in ("resnet18", "resnet34", "resnet50")
    ),
    *(
        (f"{FUSION_INPUT_PREFIX}/{run}/test_predictions.json", f"{REPRODUCE_DIR}/fused/{run}.json")
        for run in FUSION_RUNS
    ),
)

#: AI Hub 조합 폴더 이름입니다. `K-000250-000573-002483-006192_json`처럼 생겼습니다.
_COMBINATION_DIRECTORY = re.compile(r"^(K(?:-\d{6})+)_json$")
#: 이름 안 알약 코드입니다. `K`는 맨 앞에 한 번만 붙으므로 숫자만 셉니다 — `K-`를
#: 요구하면 조합마다 첫 알약 하나만 세어 class를 덮은 줄 착각합니다.
_PILL_CODE = re.compile(r"\d{6}")

#: 한 번에 받는 개수입니다. S3 왕복이 대부분이라 thread로 겹칩니다.
DOWNLOAD_WORKERS = 16
DEFAULT_DEMO_GROUPS = 60
#: class 하나를 몇 개의 조합에서 모을지입니다. 나누는 단위가 조합 통째라 하나뿐인
#: class는 검증에 못 갑니다. 2로 두면 표본이 193 조합(511장)이 됩니다.
COVER_TIMES = 2


def _inside_repository(value: str, *, label: str) -> Path:
    """저장소 안으로 확정되는 절대 경로를 만듭니다.

    문자열 접두사로 견주면 안 됩니다. `<repo>-bundle`은 `<repo>`로 **시작하지만**
    저장소 밖입니다. 경로로 견줘야 그 둘이 갈립니다.
    """

    resolved = (REPOSITORY_ROOT / value).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise BundleError(f"{label}는 저장소 안이어야 합니다: {value}") from error
    return resolved


def write_derived(path: Path, text: str, *, rebuilding: bool) -> Path:
    """이 실행이 **만들어 내는** 파일을 씁니다. 내려받는 파일이 아닙니다.

    test manifest와 `SHA256SUMS`가 그렇습니다. 내려받는 쪽만 막아 두면 이 둘이 그 검사를
    지나지 않아, flag 없는 기본 실행도 그 자리에 있던 것을 지웁니다.

    ``rebuilding``은 `--resume`이나 `--overwrite`가 켜진 경우입니다. 둘 중 하나를 준
    사람은 이미 "이 자리에 파일이 있다"를 알고 있고, 그때 이 둘은 **다시 만들어야**
    합니다 — 담긴 파일 목록이 달라졌는데 옛 목록이 남아 있으면 그것이 더 나쁩니다.
    """

    if path.exists() and not rebuilding:
        raise BundleError(
            f"{path.name}이 이미 있습니다. 다시 만들려면 --resume 또는 --overwrite를 주세요."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def storage() -> Any:
    """팀 bucket을 읽는 storage입니다. bucket 이름은 환경 변수에서 옵니다."""

    return create_storage({"storage": {"backend": "s3", "s3": {"prefix": ""}}})


def _bucket_uri(store: Any, key: str) -> str:
    return f"s3://{store.bucket}/{key}"


def download_many(
    store: Any,
    pairs: Sequence[tuple[str, Path]],
    *,
    label: str,
    resume: bool,
    overwrite: bool,
) -> None:
    """(S3 key, 로컬 경로) 쌍을 한꺼번에 받습니다.

    **이미 있는 파일을 만나면 기본은 멈춥니다.** 두 갈래가 모두 위험해서 사람이 골라야
    합니다.

    - 건너뛰면 앞선 실행이나 다른 출처가 남긴 파일이 그대로 번들에 들어가고
      `SHA256SUMS`가 그것을 정품으로 서명해 줍니다. 이 번들의 존재 이유가 "점수를 낸 그
      바이트"인데 확인할 길이 사라집니다.
    - 덮어쓰면 그 자리에 있던 것이 사라집니다. 기본 목적지가 저장소 root라 남의 산출물
      위일 수 있습니다.

    ``resume``은 끊긴 다운로드를 이어받을 때(있는 것을 믿는다), ``overwrite``는 다시
    받아 채울 때(있는 것을 버린다) 켭니다.
    """

    existing = [path for _, path in pairs if path.exists()]
    if existing and not resume and not overwrite:
        shown = ", ".join(path.name for path in existing[:3])
        raise BundleError(
            f"{label}: 이미 있는 파일이 {len(existing)}개 있습니다 ({shown}…). "
            "이어받으려면 --resume, 다시 받으려면 --overwrite를 주세요."
        )

    todo = [(key, path) for key, path in pairs if overwrite or not path.exists()]
    print(f"{label}: {len(pairs)}개 중 {len(todo)}개 내려받습니다.")
    if not todo:
        return

    def fetch(item: tuple[str, Path]) -> None:
        key, path = item
        store.download_file(_bucket_uri(store, key), path, overwrite=overwrite)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for index, _ in enumerate(pool.map(fetch, todo), start=1):
            if index % 100 == 0 or index == len(todo):
                print(f"  {index}/{len(todo)}")


def write_test_manifest(store: Any, out: Path, *, rebuilding: bool) -> Path:
    """test manifest의 이미지 위치만 번들 안 상대 경로로 바꿔 씁니다.

    원본은 S3에서 만들어져 `file_name`이 `s3://...`입니다. 그대로 두면 자격 증명이
    없는 사람이 열 수 없습니다. **바꾸는 것은 위치뿐**이고 id, 크기, category는 그대로
    둡니다 — 그 값이 바뀌면 융합 입력과 이어지지 않습니다.
    """

    document = store.read_json(_bucket_uri(store, f"{PROCESSED_V5}/test_manifest.json"))
    images = document.get("images")
    if not isinstance(images, list) or not images:
        raise BundleError("test manifest에 images가 없습니다.")

    for image in images:
        name = str(image.get("file_name", ""))
        if not name:
            raise BundleError("test manifest의 image에 file_name이 없습니다.")
        # 재현 폴더에서 데모 원본 폴더의 test 이미지를 가리킵니다.
        image["file_name"] = f"../raw/v90/test_images/{name.rsplit('/', 1)[-1]}"

    target = write_derived(
        out / REPRODUCE_DIR / "test_manifest.json",
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        rebuilding=rebuilding,
    )
    print(f"test manifest: 이미지 {len(images)}장을 번들 안 경로로 바꿔 적었습니다.")
    return target


def retarget_fusion_inputs(out: Path) -> list[Path]:
    """합칠 예측 세 개가 가리키는 test manifest를 번들 안 자리로 바꿉니다.

    evaluate는 합칠 예측마다 "어느 시험지를 본 예측인가"를 확인하는데, 적힌 자리가 이
    실행의 manifest와 다르면 **그 자리를 실제로 읽어** 내용을 견줍니다. 원본에는
    `s3://`가 적혀 있어 자격 증명이 없는 사람은 그 확인을 통과할 수 없습니다.

    **바꾸는 것은 위치 문자열 하나**입니다. 그 s3 파일에서 그대로 내려받아 이름만 바꾼
    manifest를 가리키므로 내용은 같고, id·크기·category를 견주는 확인은 그대로 돕니다.
    """

    manifest = f"{REPRODUCE_DIR}/test_manifest.json"
    changed = []
    for run in FUSION_RUNS:
        path = out / REPRODUCE_DIR / "fused" / f"{run}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("test_manifest_uri") == manifest:
            continue
        document["test_manifest_uri"] = manifest
        path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
        changed.append(path)
    print(f"합칠 예측 {len(changed)}개의 test manifest 위치를 번들 안 자리로 바꿨습니다.")
    return changed


def _annotation_entries(store: Any) -> list[str]:
    try:
        return [str(entry) for entry in store.list(_bucket_uri(store, f"{RAW_V5}/train_annotations/"))]
    except StorageError as error:
        raise BundleError(f"원본 annotation 목록을 읽지 못했습니다({type(error).__name__}).") from error


def choose_demo_groups(entries: Iterable[str], *, wanted: int) -> dict[str, list[str]]:
    """데모 표본으로 쓸 조합을 고릅니다. 먼저 class를 덮고 그다음 수를 채웁니다.

    조합 폴더 이름이 그 조합에 든 알약 코드를 그대로 담고 있어서(`K-000250-...`),
    파일을 하나도 열지 않고 어떤 class가 들어 있는지 알 수 있습니다. 앞에서부터 자르면
    한쪽 class에 몰려 준비 단계가 "이 class는 검증에 못 간다"만 잔뜩 뱉습니다.

    class마다 조합을 **`COVER_TIMES`개씩** 모읍니다. 나누는 단위가 조합 통째라, 조합
    하나에만 있는 class는 그 조합이 어느 쪽으로 가든 한쪽에만 남아 검증에 못 갑니다.
    한 번만 덮었을 때 실제로 118종 중 93종이 그렇게 train 전용이 됐습니다.
    """

    groups: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        parts = entry.split("train_annotations/", 1)
        if len(parts) != 2:
            continue
        relative = parts[1]
        directory = relative.split("/", 1)[0]
        if _COMBINATION_DIRECTORY.fullmatch(directory):
            groups[directory].append(relative)

    if not groups:
        raise BundleError("조합 폴더를 하나도 찾지 못했습니다.")

    chosen: dict[str, list[str]] = {}
    covered: Counter[str] = Counter()
    leftovers: list[str] = []
    for name in sorted(groups):
        codes = set(_PILL_CODE.findall(name))
        if any(covered[code] < COVER_TIMES for code in codes):
            covered.update(codes)
            chosen[name] = groups[name]
        else:
            leftovers.append(name)

    for name in leftovers:
        if len(chosen) >= wanted:
            break
        chosen[name] = groups[name]

    print(f"데모 조합 {len(chosen)}개, class {len(covered)}종을 골랐습니다.")
    return chosen


def demo_pairs(chosen: dict[str, list[str]], out: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """고른 조합의 annotation과 그 이미지의 (S3 key, 로컬 경로) 쌍을 만듭니다."""

    annotations: list[tuple[str, Path]] = []
    images: dict[str, Path] = {}
    for relatives in chosen.values():
        for relative in relatives:
            annotations.append(
                (
                    f"{RAW_V5}/train_annotations/{relative}",
                    out / DEMO_RAW_PREFIX / "train_annotations" / relative,
                )
            )
            stem = relative.rsplit("/", 1)[-1].removesuffix(".json")
            images[f"{stem}.png"] = out / DEMO_RAW_PREFIX / "train_images" / f"{stem}.png"
    image_pairs = [(f"{RAW_V5}/train_images/{name}", path) for name, path in sorted(images.items())]
    return annotations, image_pairs


def test_image_pairs(store: Any, out: Path) -> list[tuple[str, Path]]:
    """대회 test 이미지 전부입니다. 데모 제출도 채점받을 수 있어야 하므로 다 담습니다."""

    entries = [str(entry) for entry in store.list(_bucket_uri(store, f"{RAW_V5}/test_images/"))]
    pairs = []
    for entry in entries:
        name = entry.rsplit("/", 1)[-1]
        if name:
            pairs.append((f"{RAW_V5}/test_images/{name}", out / DEMO_RAW_PREFIX / "test_images" / name))
    if not pairs:
        raise BundleError("test 이미지를 찾지 못했습니다.")
    return pairs


def write_checksums(out: Path, files: Sequence[Path], *, rebuilding: bool) -> Path:
    """번들에 담을 파일의 sha256을 적습니다. 받은 쪽이 깨진 다운로드를 알아챕니다.

    폴더를 훑지 않고 **이번에 담기로 한 목록**만 적습니다. 번들은 저장소 안 실제
    작업 폴더에 만들어지므로, 훑으면 남의 파일까지 목록에 들어갑니다.
    """

    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(out).as_posix()}")
    target = write_derived(
        out / REPRODUCE_DIR / "SHA256SUMS",
        "\n".join(lines) + "\n",
        rebuilding=rebuilding,
    )
    print(f"SHA256SUMS: {len(lines)}개 파일")
    return target


def pack(out: Path, files: Sequence[Path], archive: Path) -> None:
    """담기로 한 파일을 tar.gz 하나로 묶습니다. 받는 쪽은 저장소 root에서 풉니다."""

    if archive.exists():
        raise BundleError(f"이미 있는 파일은 덮지 않습니다: {archive.name}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(out).as_posix())
    print(f"묶었습니다: {archive.name} ({archive.stat().st_size / 1e9:.2f} GB)")


def build(
    out: Path, *, groups: int, skip_test_images: bool, resume: bool, overwrite: bool
) -> list[Path]:
    rebuilding = resume or overwrite
    """번들에 담을 파일을 모두 제자리에 놓고, 담을 목록을 돌려줍니다."""

    store = storage()
    files: list[Path] = []

    fixed = [(key, out / target) for key, target in FIXED_FILES]
    download_many(store, fixed, label="재현 재료", resume=resume, overwrite=overwrite)
    files.extend(path for _, path in fixed)

    files.append(write_test_manifest(store, out, rebuilding=rebuilding))
    retarget_fusion_inputs(out)

    if not skip_test_images:
        pairs = test_image_pairs(store, out)
        download_many(store, pairs, label="test 이미지", resume=resume, overwrite=overwrite)
        files.extend(path for _, path in pairs)

    if groups > 0:
        chosen = choose_demo_groups(_annotation_entries(store), wanted=groups)
        annotations, images = demo_pairs(chosen, out)
        download_many(store, annotations, label="데모 annotation", resume=resume, overwrite=overwrite)
        download_many(store, images, label="데모 train 이미지", resume=resume, overwrite=overwrite)
        files.extend(path for _, path in annotations)
        files.extend(path for _, path in images)

    files = sorted(set(files))
    total = sum(path.stat().st_size for path in files)
    print(f"번들 크기: {total / 1e9:.2f} GB, 파일 {len(files)}개")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="발표용 재현 번들 만들기")
    parser.add_argument(
        "--out",
        default=".",
        help=(
            "번들을 놓을 저장소 기준 폴더. 기본값은 저장소 root라, 만든 자리가 곧 "
            "받는 사람이 풀었을 때의 자리입니다"
        ),
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=DEFAULT_DEMO_GROUPS,
        help=(
            "데모 표본에 담을 조합 수의 목표치. class를 다 덮는 데 필요한 조합은 이 수와"
            f" 상관없이 먼저 들어갑니다. 0이면 데모 표본을 담지 않습니다 (기본 {DEFAULT_DEMO_GROUPS})"
        ),
    )
    parser.add_argument(
        "--skip-test-images",
        action="store_true",
        help="1.5 GB짜리 test 이미지를 건너뜁니다. 나머지만 먼저 확인할 때 씁니다",
    )
    parser.add_argument(
        "--pack",
        default=None,
        help="묶을 tar.gz 경로. 주지 않으면 폴더만 만듭니다",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "이미 있는 파일을 그대로 두고 없는 것만 받습니다. 끊긴 다운로드를 이어받을 "
            "때만 쓰세요 — 다른 출처가 남긴 파일도 번들에 들어가고 SHA256SUMS가 서명합니다"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 있는 파일을 버리고 다시 받습니다. 그 자리에 있던 것은 사라집니다",
    )
    args = parser.parse_args(argv)

    if args.resume and args.overwrite:
        # 하나는 있는 것을 믿고 하나는 버립니다. 조용히 한쪽을 고르면 파괴적인 쪽이
        # 이깁니다.
        print("실패: --resume과 --overwrite는 함께 쓸 수 없습니다.", file=sys.stderr)
        return 1

    out = _inside_repository(args.out, label="번들 폴더")
    out.mkdir(parents=True, exist_ok=True)

    try:
        files = build(
            out,
            groups=args.groups,
            skip_test_images=args.skip_test_images,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        files.append(write_checksums(out, files, rebuilding=args.resume or args.overwrite))
        if args.pack:
            pack(out, files, _inside_repository(args.pack, label="묶을 파일"))
    except (BundleError, StorageError) as error:
        print(f"실패: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
