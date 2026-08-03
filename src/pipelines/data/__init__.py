__all__ = ["run"]


def run(config: dict) -> dict:
    """Data pipeline의 연결 상태를 확인하는 dummy 실행입니다."""
    return {
        "status": "ok",
        "artifacts": {},
        "summary": {"pipeline": "data", "mode": "dummy"},
        "message": "data pipeline dummy 실행 완료",
    }
