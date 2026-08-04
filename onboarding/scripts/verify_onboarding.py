import argparse
import importlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PACKAGES = {
    "boto3": ("boto3", "1.43.63"),
    "numpy": ("numpy", "2.0.2"),
    "Pillow": ("PIL", "11.3.0"),
    "pytest": ("pytest", "8.3.5"),
    "torch": ("torch", "2.12.1+cu126"),
    "torchvision": ("torchvision", "0.27.1+cu126"),
}
REQUIRED_PACKAGES = tuple(
    import_name for import_name, _ in EXPECTED_PACKAGES.values()
)
EXPECTED_CUDA_VERSION = "12.6"
LOCAL_PYTHON_VERSION = (3, 11)
COLAB_PYTHON_VERSIONS = ((3, 11), (3, 12))
REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "pytest.ini",
    "onboarding/docs/onboarding.md",
    "onboarding/docs/onboarding-status.md",
    "onboarding/scripts/verify_onboarding.py",
    "onboarding/tests/test_onboarding.py",
)


class VerificationError(RuntimeError):
    """온보딩 검증 실패를 나타냅니다."""


def supported_python_versions(profile: str) -> tuple[tuple[int, int], ...]:
    if profile == "colab":
        return COLAB_PYTHON_VERSIONS
    return (LOCAL_PYTHON_VERSION,)


def check_python_version(profile: str = "cpu") -> None:
    supported = supported_python_versions(profile)
    if sys.version_info[:2] not in supported:
        version = ".".join(str(part) for part in sys.version_info[:3])
        expected = " 또는 ".join(
            f"{major}.{minor}" for major, minor in supported
        )
        raise VerificationError(
            f"{profile} profile은 Python {expected}이 필요합니다. "
            f"현재 버전: {version}"
        )


def check_required_packages() -> None:
    missing = []
    mismatched = []
    for distribution, (import_name, expected_version) in EXPECTED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(distribution)
            continue

        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
            continue

        if installed_version != expected_version:
            mismatched.append(
                f"{distribution}=={installed_version} "
                f"(필요: {expected_version})"
            )

    if missing:
        raise VerificationError(
            "import할 수 없는 package: " + ", ".join(sorted(missing))
        )
    if mismatched:
        raise VerificationError(
            "설치 버전 불일치: " + "; ".join(sorted(mismatched))
        )


def check_dependency_consistency() -> None:
    completed = subprocess.run(
        (sys.executable, "-m", "pip", "check"),
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = completed.stdout.strip() or completed.stderr.strip()
        raise VerificationError(f"pip dependency 충돌: {details}")


def check_required_files() -> None:
    missing = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not (REPOSITORY_ROOT / relative_path).is_file()
    ]
    if missing:
        raise VerificationError("누락된 repository file: " + ", ".join(missing))


def check_utf8_compatibility() -> None:
    issues = []
    for relative_path in REQUIRED_FILES:
        path = REPOSITORY_ROOT / relative_path
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf"):
            issues.append(f"{relative_path}: UTF-8 BOM")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"{relative_path}: invalid UTF-8")
        if b"\r" in data:
            issues.append(f"{relative_path}: CR/CRLF line ending")

    if issues:
        raise VerificationError("text compatibility failure: " + "; ".join(issues))


def parse_driver_version(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.strip().split("."))
    except ValueError as error:
        raise VerificationError(
            f"NVIDIA driver version을 해석할 수 없습니다: {version}"
        ) from error


def check_nvidia_driver() -> None:
    command = (
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError as error:
        raise VerificationError("nvidia-smi를 실행할 수 없습니다.") from error

    if completed.returncode != 0:
        details = completed.stderr.strip() or "unknown nvidia-smi error"
        raise VerificationError(f"NVIDIA driver 확인 실패: {details}")

    minimum = (528, 33) if platform.system() == "Windows" else (525, 60, 13)
    gpu_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not gpu_lines:
        raise VerificationError("nvidia-smi가 GPU 정보를 반환하지 않았습니다.")

    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise VerificationError(f"예상하지 못한 nvidia-smi 출력: {line}")
        name, driver, memory = fields
        if parse_driver_version(driver) < minimum:
            expected = ".".join(str(part) for part in minimum)
            raise VerificationError(
                f"CUDA 12.x에는 NVIDIA driver {expected} 이상이 필요합니다. "
                f"현재 버전: {driver}"
            )
        print(f"[INFO] GPU: {name} | driver {driver} | {memory}")


def check_deep_learning_compatibility(profile: str = "cpu") -> None:
    try:
        import numpy as np
        import torch
        import torchvision
        from PIL import Image
        from torchvision.transforms import functional as transform_functional

        if torch.version.cuda != EXPECTED_CUDA_VERSION:
            raise VerificationError(
                f"PyTorch CUDA build가 {EXPECTED_CUDA_VERSION}이 아닙니다. "
                f"현재 build: {torch.version.cuda or 'CPU only'}"
            )

        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]])
        loss = (left @ right).sum()
        loss.backward()
        if loss.item() != 11.0 or left.grad is None:
            raise VerificationError("PyTorch CPU tensor/autograd 결과가 올바르지 않습니다.")

        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor_from_numpy = torch.from_numpy(array)
        if not np.array_equal(tensor_from_numpy.numpy(), array):
            raise VerificationError("NumPy와 PyTorch tensor 변환이 실패했습니다.")

        image = Image.new("RGB", (2, 2), color=(10, 20, 30))
        image_tensor = transform_functional.pil_to_tensor(image)
        if tuple(image_tensor.shape) != (3, 2, 2):
            raise VerificationError("Pillow와 TorchVision 이미지 변환이 실패했습니다.")

        boxes = torch.tensor(
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        kept = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        if kept.tolist() != [0, 2]:
            raise VerificationError("TorchVision CPU NMS 결과가 올바르지 않습니다.")

        if profile in {"cuda", "colab"}:
            check_nvidia_driver()
            if not torch.cuda.is_available():
                raise VerificationError(
                    "CUDA profile이지만 torch.cuda.is_available()이 False입니다."
                )

            device = torch.device("cuda")
            torch.cuda.set_device(0)
            warmup = torch.ones((1, 1), device=device)
            warmup @ warmup
            torch.cuda.synchronize()
            cuda_left = left.detach().clone().to(device).requires_grad_(True)
            cuda_right = right.to(device)
            cuda_result = cuda_left @ cuda_right
            cuda_loss = cuda_left.square().sum()
            cuda_loss.backward()
            cuda_boxes = boxes.to(device)
            cuda_scores = scores.to(device)
            cuda_kept = torchvision.ops.nms(
                cuda_boxes, cuda_scores, iou_threshold=0.5
            )
            torch.cuda.synchronize()
            if cuda_result.item() != 11.0 or cuda_left.grad is None:
                raise VerificationError("CUDA tensor/autograd 결과가 올바르지 않습니다.")
            if cuda_kept.cpu().tolist() != [0, 2]:
                raise VerificationError("CUDA tensor/NMS 결과가 올바르지 않습니다.")

            properties = torch.cuda.get_device_properties(0)
            memory_mib = properties.total_memory // (1024 * 1024)
            print(
                f"[INFO] PyTorch CUDA device: {properties.name} | "
                f"{memory_mib} MiB"
            )
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(
            f"딥러닝 library 상호 작동 실패: {type(error).__name__}: {error}"
        ) from error


def run_onboarding_tests() -> None:
    command = (
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "onboarding",
        "onboarding/tests/test_onboarding.py",
        "-q",
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode != 0:
        raise VerificationError("onboarding test가 실패했습니다.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="팀 공통 dependency 온보딩 검증")
    parser.add_argument(
        "--profile",
        choices=("cpu", "cuda", "colab"),
        default="cpu",
        help="cpu는 공통 기준, cuda/colab은 NVIDIA GPU까지 필수 검사합니다.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = (
        ("supported Python", lambda: check_python_version(args.profile)),
        ("required package imports and versions", check_required_packages),
        ("pip dependency consistency", check_dependency_consistency),
        ("required repository files", check_required_files),
        ("UTF-8 compatibility", check_utf8_compatibility),
        (
            "PyTorch library compatibility",
            lambda: check_deep_learning_compatibility(args.profile),
        ),
        ("onboarding tests", run_onboarding_tests),
    )

    print(f"ONBOARDING PROFILE: {args.profile}")
    for label, check in checks:
        try:
            check()
        except VerificationError as error:
            print(f"[FAIL] {label}: {error}")
            print("ONBOARDING VERIFICATION FAILED")
            return 1
        print(f"[PASS] {label}")

    print("ONBOARDING VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
