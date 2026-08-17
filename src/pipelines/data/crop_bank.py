"""정답 상자로 알약 하나짜리 crop을 잘라 참조 은행을 만듭니다.

검출된 상자가 어떤 알약인지 다시 판정하려면 **알약 하나만 담긴 그림**이 필요합니다.
검출 상자는 이미 정답에 가까우므로, 학습 정답 상자를 그대로 써서 그 자리를 채웁니다.

**train split에서만 자릅니다.** validation crop을 은행에 넣으면 그 dataset으로 잰
어떤 점수도 자기 답을 보고 매긴 것이 됩니다.

crop은 수천 개라 낱개로 올리면 저장소 왕복만으로 준비 시간이 몇 배가 됩니다. 그래서
**tar 하나**로 묶어 한 번에 올립니다. 안에는 `index.json`과 `crops/<category>/*.jpg`가
들어갑니다.
"""

from __future__ import annotations

import io
import json
import posixpath
import random
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.common import StorageError

from .errors import DatasetPreparationError


#: 은행에 담을 crop 한 변의 픽셀 수입니다. 알약 하나가 원본에서 200~500px이라
#: 이보다 키워도 정보가 늘지 않습니다.
CROP_SIZE = 224

#: 상자 바깥으로 이만큼 더 잘라 냅니다. 나중에 쓰는 쪽의 상자가 조금 어긋나도
#: 알약이 창 밖으로 나가지 않게 하는 여유입니다.
CROP_MARGIN = 0.08

#: class 하나에서 담을 crop 수의 기본값입니다. 늘리면 준비 시간과 용량이 함께 늡니다.
DEFAULT_PER_CLASS = 40

#: 은행 파일 이름입니다.
CROP_BANK_FILE_NAME = "crop_bank.tar"

#: tar 안에서 목록이 놓이는 자리입니다.
INDEX_MEMBER = "index.json"


def _sample(
    annotations_by_category: Mapping[int, list[Mapping[str, Any]]],
    group_of: Mapping[Any, str],
    per_class: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    """class마다 조합을 **돌아가며 한 장씩** 뽑습니다.

    같은 조합에서만 뽑으면 같은 알약을 같은 배치로 찍은 사진만 담겨, 은행이 그
    조합 하나를 외운 것과 다를 바 없어집니다.

    조합을 섞은 뒤 앞에서부터 채우면 그 일이 그대로 일어납니다. 조합 하나가
    상한보다 많은 사진을 갖고 있으면 **첫 조합 하나로 상한이 다 찹니다.** 원본은
    한 조합을 각도와 조명만 바꿔 수십 장씩 찍은 것이라 그것이 보통입니다. 그래서
    섞기만 하지 않고 조합을 번갈아 가며 한 장씩 가져옵니다.
    """

    generator = random.Random(seed)
    picked: list[Mapping[str, Any]] = []
    for category_id in sorted(annotations_by_category):
        by_group: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for annotation in annotations_by_category[category_id]:
            by_group[group_of.get(annotation["image_id"], "")].append(annotation)
        groups = sorted(by_group)
        generator.shuffle(groups)
        lanes = [by_group[group] for group in groups]
        chosen: list[Mapping[str, Any]] = []
        for position in range(max(len(lane) for lane in lanes)):
            for lane in lanes:
                if position < len(lane):
                    chosen.append(lane[position])
                    if len(chosen) >= per_class:
                        break
            if len(chosen) >= per_class:
                break
        picked.extend(chosen)
    return picked


def _crop(image: Image.Image, bbox: Sequence[float]) -> Image.Image | None:
    """상자 하나를 여유를 두고 잘라 정해진 크기로 맞춥니다."""

    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        return None
    pad_x, pad_y = width * CROP_MARGIN, height * CROP_MARGIN
    box = (
        max(0, int(x - pad_x)),
        max(0, int(y - pad_y)),
        min(image.width, int(x + width + pad_x)),
        min(image.height, int(y + height + pad_y)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return image.crop(box).convert("RGB").resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)


def build_crop_bank(
    storage: Any,
    images: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    train_image_ids: set[Any],
    group_of: Mapping[Any, str],
    per_class: int,
    seed: int,
    archive_path: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """train split의 정답 상자를 잘라 tar 하나로 묶습니다. **올리지는 않습니다.**

    이미지를 전부 여는 비싼 단계라, 산출물을 하나라도 내보내기 **전에** 여기서
    끝냅니다. 도중에 실패했는데 manifest가 이미 나가 있으면 반쪽짜리 dataset이
    남기 때문입니다. 올리는 것은 부르는 쪽이 publish 단계에서 합니다.

    담을 crop이 하나도 없으면 :class:`DatasetPreparationError`입니다 — 빈 은행을
    남기면 읽는 쪽이 "class가 없다"와 "아직 안 만들었다"를 구분하지 못합니다.
    """

    if per_class < 1:
        raise DatasetPreparationError(
            "config['data']['crop_bank_per_class']는 1 이상이어야 합니다."
        )

    by_image: defaultdict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    by_category: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        if annotation["image_id"] not in train_image_ids:
            continue
        by_image[annotation["image_id"]].append(annotation)
        by_category[int(annotation["category_id"])].append(annotation)
    if not by_category:
        raise DatasetPreparationError("train split에 crop을 만들 annotation이 없습니다.")

    wanted = _sample(by_category, group_of, per_class, seed)
    wanted_by_image: defaultdict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in wanted:
        wanted_by_image[annotation["image_id"]].append(annotation)

    location_of = {image["id"]: str(image["file_name"]) for image in images}
    records: list[dict[str, Any]] = []
    per_category: defaultdict[int, int] = defaultdict(int)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crop-bank-") as scratch:
        with tarfile.open(archive_path, "w") as archive:
            chosen_images = sorted(wanted_by_image, key=lambda value: str(value))
            for done, image_id in enumerate(chosen_images, start=1):
                location = location_of.get(image_id)
                if location is None:
                    continue
                local = Path(scratch) / posixpath.basename(location.replace("\\", "/"))
                try:
                    storage.download_file(location, local)
                    with Image.open(local) as opened:
                        opened.load()
                        for index, annotation in enumerate(wanted_by_image[image_id]):
                            patch = _crop(opened, annotation["bbox"])
                            if patch is None:
                                continue
                            category_id = int(annotation["category_id"])
                            name = f"crops/{category_id}/{image_id}_{index}.jpg"
                            buffer = io.BytesIO()
                            patch.save(buffer, format="JPEG", quality=92)
                            payload = buffer.getvalue()
                            info = tarfile.TarInfo(name)
                            info.size = len(payload)
                            archive.addfile(info, io.BytesIO(payload))
                            per_category[category_id] += 1
                            records.append(
                                {
                                    "path": name,
                                    "category_id": category_id,
                                    "image_id": image_id,
                                    "group": group_of.get(image_id, ""),
                                }
                            )
                except (StorageError, OSError, UnidentifiedImageError) as error:
                    raise DatasetPreparationError(
                        f"crop 은행을 만들려고 이미지를 여는 데 실패했습니다: {location}"
                    ) from error
                finally:
                    local.unlink(missing_ok=True)
                if on_progress is not None:
                    on_progress(done, len(chosen_images))

            if not records:
                raise DatasetPreparationError("잘라 낸 crop이 하나도 없습니다.")
            payload = json.dumps(
                {
                    "version": 1,
                    "crop_size": CROP_SIZE,
                    "crop_margin": CROP_MARGIN,
                    "per_class": per_class,
                    "seed": seed,
                    "split": "train",
                    "records": records,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            info = tarfile.TarInfo(INDEX_MEMBER)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    counts = sorted(per_category.values())
    return {
        "crop_count": len(records),
        "category_count": len(per_category),
        "per_class": per_class,
        "crop_size": CROP_SIZE,
        "smallest_class": counts[0],
        "largest_class": counts[-1],
    }


# 은행을 **읽는** 함수는 여기 두지 않습니다. 읽는 쪽(train, evaluate)은 이 module을
# import할 수 없어 각자 자기 reader를 갖고 있고, 여기 하나를 더 두면 아무도 부르지
# 않는 tar 해제 코드가 남습니다. 푸는 코드는 그 자체로 위험면이라, 쓰지 않는 것을
# "언젠가 쓸지도 모르니" 남겨 둘 이유가 없습니다.
