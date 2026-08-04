import argparse
import json
from collections.abc import Sequence

from src.common import load_config, validate_pipeline_result
from src.pipelines import data, evaluate, registry, train, web


_STAGES = (
    ("data", data),
    ("train", train),
    ("evaluate", evaluate),
    ("registry", registry),
    ("web", web),
)


def run(config: dict | None = None, only: str | None = None) -> dict:
    """전체 dummy pipeline 또는 선택한 pipeline 하나를 실행합니다."""
    resolved_config = dict(config or {})
    stage_names = {name for name, _ in _STAGES}
    if only is not None and only not in stage_names:
        raise ValueError(f"지원하지 않는 pipeline입니다: {only}")

    artifacts = {}
    summaries = {}

    for name, pipeline in _STAGES:
        if only is not None and name != only:
            continue

        stage_config = dict(resolved_config)
        stage_config["inputs"] = dict(artifacts)
        result = pipeline.run(stage_config)
        validate_pipeline_result(result, pipeline_name=name)

        artifacts[name] = result["artifacts"]
        summaries[name] = result["summary"]

        if result["status"] != "ok":
            return {
                "status": "error",
                "artifacts": artifacts,
                "summary": summaries,
                "message": f"{name}: {result['message']}",
            }

    return {
        "status": "ok",
        "artifacts": artifacts,
        "summary": summaries,
        "message": "dummy pipeline 실행 완료",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pill object-detection dummy pipeline")
    parser.add_argument("--config", help="읽을 JSON config 경로")
    parser.add_argument("--only", choices=[name for name, _ in _STAGES])
    args = parser.parse_args(argv)

    result = run(load_config(args.config), only=args.only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
