import json
from pathlib import Path


def load_config(path: str | Path | None = None) -> dict:
    """JSON config를 읽습니다. 경로가 없으면 빈 config를 반환합니다."""
    if path is None:
        return {}

    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config의 최상위 값은 object여야 합니다.")
    return config
