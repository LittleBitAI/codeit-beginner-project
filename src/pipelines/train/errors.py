"""Train pipeline 내부에서 사용하는 typed error입니다."""


class TrainError(RuntimeError):
    """학습을 안전하게 중단하고 공개 error 결과로 바꿔야 하는 오류입니다."""
