"""`기타 알약`으로 뭉쳐 놓은 category_id를 알약별 코드로 되돌립니다.

`aihub_to_competition.py --label-space competition-plus-other`로 만든 raw prefix는
대회 56종만 자기 코드를 갖고 나머지 알약이 모두 `기타 알약`(999999) 하나에 들어가
있습니다. 대회가 밝힌 대로 **시험 데이터에는 학습 데이터에 없는 class가 있으므로**,
그 상태로는 그 알약들을 맞힐 방법이 아예 없습니다.

되돌릴 근거는 이미 파일 안에 있습니다. 뭉개진 것은 `category_id`뿐이고 제품코드
(`dl_mapping_code`)와 제품명(`dl_name`)은 그대로 남아 있어서, AI Hub 원본을 다시
받지 않고도 복원됩니다.

    python -m scripts.restore_full_classes --source <raw prefix> --output <새 prefix>

`--source`는 `train_annotations/`를 담은 디렉터리입니다. 이미지는 건드리지 않습니다.
image_id·annotation_id·bbox도 그대로 둡니다. 그 값이 바뀌면 같은 seed로도 data
pipeline의 split이 달라져 이전 실행과 비교할 수 없게 됩니다.

`--output`은 비어 있어야 합니다. 덮어쓰기 선택지를 두지 않은 이유는, 덧쓰기만 하면
이번에 안 나온 파일이 남아 옛 label이 섞이고 그것을 알아챌 방법이 없기 때문입니다.
이미 있는 결과를 지우는 것은 사람이 결정할 일입니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.aihub_to_competition import OTHER_CATEGORY_ID


ANNOTATION_DIRECTORY = "train_annotations"
PILL_CODE_PATTERN = re.compile(r"^K-(\d{6})$")


class RestoreError(RuntimeError):
    """복원을 계속할 수 없을 때 냅니다."""


def product_code(image: Mapping[str, Any], source: Path) -> tuple[int, str]:
    """제품코드에서 category_id와 이름을 만듭니다.

    `dl_idx`는 쓰지 않습니다. 그 값은 K코드 숫자보다 1 작거나(`K-001900` → `1899`)
    아예 다른 경우가 있어 대회 label 공간과 어긋납니다. `aihub_to_competition.py`가
    `dl_mapping_code`를 쓰는 것과 같은 이유입니다.

    **`drug_N`으로 대신하지 않습니다.** 두 필드는 대개 같지만 같다는 보장이 없고,
    권위값이 없을 때 다른 값으로 넘어가면 어긋나는 문서에서 엉뚱한 알약의 label이
    조용히 붙습니다. 그래서 없으면 멈추고, 둘이 다르면 그것도 멈춥니다.
    """

    code = image.get("dl_mapping_code")
    if not isinstance(code, str) or not PILL_CODE_PATTERN.match(code):
        raise RestoreError(
            f"dl_mapping_code가 없거나 K-000000 형식이 아닙니다: {source} ({code!r})"
        )
    alternate = image.get("drug_N")
    if alternate is not None and alternate != code:
        raise RestoreError(
            f"drug_N이 dl_mapping_code와 다릅니다: {source} ({alternate!r} != {code!r})"
        )
    name = image.get("dl_name")
    if not isinstance(name, str) or not name.strip():
        raise RestoreError(f"dl_name이 없습니다: {source}")
    return int(PILL_CODE_PATTERN.match(code).group(1)), name.strip()


def _restore_document(document: Mapping[str, Any], source: Path) -> tuple[dict[str, Any], int]:
    images = document.get("images")
    if not isinstance(images, Sequence) or not images:
        raise RestoreError(f"images가 비어 있습니다: {source}")
    category, name = product_code(images[0], source)

    restored = 0
    annotations: list[dict[str, Any]] = []
    for annotation in document.get("annotations") or []:
        existing = annotation.get("category_id")
        if existing == OTHER_CATEGORY_ID:
            restored += 1
        elif existing != category:
            # 뭉친 것도 아닌데 제품코드와 다르면 어느 쪽이 맞는지 알 수 없습니다.
            raise RestoreError(
                f"category_id가 제품코드와 다릅니다: {source} "
                f"({existing} != {category})"
            )
        annotations.append(dict(annotation, category_id=category))

    return (
        {
            **document,
            "annotations": annotations,
            "categories": [{"supercategory": "pill", "id": category, "name": name}],
        },
        restored,
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RestoreError(f"annotation을 읽지 못했습니다: {path} ({error})") from error


def restore(source: Path, output: Path) -> dict[str, Any]:
    """`source`의 annotation을 복원해 `output`에 같은 구조로 씁니다.

    **문서를 전부 확인한 뒤에 씁니다.** 뒤쪽 문서에서 멈췄는데 앞쪽 결과가 남으면 일부
    알약만 복원된 자리가 되고, class 수만 보고는 알아채지 못합니다.

    **출력 자리가 비어 있지 않으면 아무것도 하지 않습니다.** 덧쓰기만 하면 이번에 안
    나온 파일이 남아 옛 label이 섞이고, 지우는 것은 사람이 결정할 일입니다.
    """

    source, output = Path(source), Path(output)
    annotation_root = source / ANNOTATION_DIRECTORY
    if not annotation_root.is_dir():
        raise RestoreError(f"{ANNOTATION_DIRECTORY}/가 없습니다: {source}")
    if output.exists() and any(output.iterdir()):
        raise RestoreError(f"출력 자리에 이미 무언가 있습니다: {output}")

    documents = sorted(annotation_root.rglob("*.json"))
    if not documents:
        raise RestoreError(f"annotation json이 없습니다: {annotation_root}")

    # 먼저 전부 확인만 합니다. 문서가 4만 개를 넘어 결과를 들고 있으면 메모리가
    # 기가 단위로 늘어나므로, 값만 모으고 쓰기는 다시 훑습니다.
    categories: dict[int, str] = {}
    restored_annotations = 0
    for path in documents:
        fixed, restored = _restore_document(_read(path), path)
        restored_annotations += restored
        entry = fixed["categories"][0]
        categories[entry["id"]] = entry["name"]

    destination_root = output / ANNOTATION_DIRECTORY
    for path in documents:
        fixed, _ = _restore_document(_read(path), path)
        target = destination_root / path.relative_to(annotation_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(fixed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    report = {
        "source": str(source),
        "output": str(output),
        "documents": len(documents),
        "restored_annotations": restored_annotations,
        "category_count": len(categories),
        "categories": [
            {"id": key, "name": categories[key]} for key in sorted(categories)
        ],
    }
    (output / "restore_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="train_annotations/를 담은 디렉터리")
    parser.add_argument("--output", required=True, type=Path, help="복원한 annotation을 쓸 자리(비어 있어야 합니다)")
    arguments = parser.parse_args(argv)

    try:
        report = restore(arguments.source, arguments.output)
    except RestoreError as error:
        print(f"복원하지 못했습니다: {error}", file=sys.stderr)
        return 1

    print(
        f"문서 {report['documents']}개에서 상자 {report['restored_annotations']}개를 되돌렸습니다. "
        f"class {report['category_count']}종."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
