"""화면이 한 번에 채워 주는 설정이 정말 시작될 수 있는 값인지 확인합니다.

값을 손으로 옮겨 적은 표라 여기가 유일한 안전장치입니다. 이름이나 규칙이 train에서
바뀌면 화면은 조용히 못 쓰는 값을 채워 주고, 누른 사람은 시작 단추에서야 알게 됩니다.
"""

from __future__ import annotations

import pytest

from src.pipelines.web import recipes
from src.pipelines.web.train_config import DATA_ARTIFACT_KEYS, validate_request


DATA_INPUTS = {
    key: f"datasets/pill_detection/reproduce/{key}.json" for key in DATA_ARTIFACT_KEYS
}


@pytest.mark.parametrize("recipe", recipes.RECIPES, ids=lambda item: item["name"])
def test_every_recipe_passes_the_check_that_guards_the_start(recipe):
    result = validate_request(
        {
            "train": {**recipe["settings"], "run_id": recipe["name"]},
            "data": dict(DATA_INPUTS),
        }
    )

    assert result["errors"] == []
    assert result["valid"] is True


def test_recipes_are_copies_so_a_caller_cannot_change_the_record():
    first = recipes.recipe_specs()
    first[0]["settings"]["epochs"] = 1

    assert recipes.recipe_specs()[0]["settings"]["epochs"] != 1


def test_the_new_experiment_form_is_offered_the_recipes(client):
    """화면은 서버가 주는 것만 그립니다. 여기 없으면 버튼도 없습니다."""

    body = client.get("/api/train/defaults").json()

    names = [item["name"] for item in body["recipes"]]
    assert "best-detector" in names
    assert body["recipes"][0]["settings"]["architecture"] == "dino_r50_4scale"
