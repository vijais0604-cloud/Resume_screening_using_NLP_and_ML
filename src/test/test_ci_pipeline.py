import pytest
from pathlib import Path


def test_ml_dependencies():
    """Ensure our primary machine learning packages import correctly."""
    try:
        import torch
        import sklearn
        import xgboost as xgb
        import mlflow
    except ImportError as e:
        pytest.fail(f"Critical dependency failed to import: {e}")


def test_torch_cpu_mode():
    """Verify that PyTorch is working correctly in CPU mode as configured."""
    import torch

    tensor = torch.rand(2, 3)
    assert tensor.shape == (2, 3), "Tensor shape mismatch"
    assert not tensor.is_cuda, "Expected CPU tensor, but CUDA was detected"


def test_required_files():
    """Ensure the codebase has the necessary files for a clean build."""
    repo_root = Path(__file__).resolve().parents[2]
    required_files = [repo_root / "requirements.txt"]

    for filepath in required_files:
        assert filepath.exists(), f"Missing essential file: {filepath.name}"

    dockerfile_path = repo_root / "Dockerfile"
    dockerfile_lower = repo_root / "dockerfile"
    assert dockerfile_path.exists() or dockerfile_lower.exists(), (
        "Missing essential Docker file: expected 'Dockerfile' or 'dockerfile'"
    )
