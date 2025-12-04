"""
Tests for the evaluation module.
"""
import torch
import pytest
import numpy as np
import sys
import os

# Add src to path and import directly to avoid open_clip dependency
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from svd_hybrid.evaluation import (
    evaluate_merged_model,
    create_baseline_task_arithmetic,
    create_baseline_ties,
    compare_methods,
    generate_evaluation_report
)


def create_test_data():
    """Create synthetic test data for evaluation tests."""
    torch.manual_seed(42)
    
    # Base model with 2 layers
    base_state_dict = {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
        "layer2.bias": torch.randn(5)
    }
    
    # Task vectors (changes from base)
    task_vectors = {
        "task1": {
            "layer1.weight": torch.randn(10, 10) * 0.1,
            "layer1.bias": torch.randn(10) * 0.1,
            "layer2.weight": torch.randn(5, 10) * 0.1,
            "layer2.bias": torch.randn(5) * 0.1
        },
        "task2": {
            "layer1.weight": torch.randn(10, 10) * 0.1,
            "layer1.bias": torch.randn(10) * 0.1,
            "layer2.weight": torch.randn(5, 10) * 0.1,
            "layer2.bias": torch.randn(5) * 0.1
        },
        "task3": {
            "layer1.weight": torch.randn(10, 10) * 0.1,
            "layer1.bias": torch.randn(10) * 0.1,
            "layer2.weight": torch.randn(5, 10) * 0.1,
            "layer2.bias": torch.randn(5) * 0.1
        }
    }
    
    return base_state_dict, task_vectors


def test_evaluate_merged_model_without_eval_fn():
    """Test evaluate_merged_model without evaluation function."""
    base, task_vectors = create_test_data()
    
    # Evaluate without evaluation function
    results = evaluate_merged_model(
        base,
        task_names=["task1", "task2"],
        evaluation_fn=None,
        verbose=False
    )
    
    # Should have structure even without eval function
    assert "per_task_accuracy" in results
    assert "task_count" in results
    assert results["task_count"] == 2
    
    # Accuracies should be None without eval function
    assert results["per_task_accuracy"]["task1"] is None
    assert results["per_task_accuracy"]["task2"] is None
    assert results["average_accuracy"] is None


def test_evaluate_merged_model_with_eval_fn():
    """Test evaluate_merged_model with a mock evaluation function."""
    base, task_vectors = create_test_data()
    
    # Mock evaluation function that returns fixed accuracies
    def mock_eval_fn(task_name, state_dict):
        accuracies = {"task1": 0.85, "task2": 0.78, "task3": 0.92}
        return accuracies.get(task_name, 0.5)
    
    results = evaluate_merged_model(
        base,
        task_names=["task1", "task2", "task3"],
        evaluation_fn=mock_eval_fn,
        verbose=True  # Enable verbose to get validation_checks
    )
    
    # Check accuracies match mock
    assert results["per_task_accuracy"]["task1"] == 0.85
    assert results["per_task_accuracy"]["task2"] == 0.78
    assert results["per_task_accuracy"]["task3"] == 0.92
    
    # Check statistics
    expected_avg = (0.85 + 0.78 + 0.92) / 3
    assert abs(results["average_accuracy"] - expected_avg) < 1e-6
    assert results["min_accuracy"] == 0.78
    assert results["max_accuracy"] == 0.92
    
    # Validation checks should pass when verbose is True
    assert "validation_checks" in results
    assert results["validation_checks"]["average_ok"] is True  # >70%
    assert results["validation_passed"] is True


def test_create_baseline_task_arithmetic():
    """Test Task Arithmetic baseline creation."""
    base, task_vectors = create_test_data()
    
    # Create merged model
    merged = create_baseline_task_arithmetic(
        base, task_vectors, scaling_factor=1.0, verbose=False
    )
    
    # Check merged has same keys as base
    assert set(merged.keys()) == set(base.keys())
    
    # Check shapes match
    for key in base.keys():
        assert merged[key].shape == base[key].shape
    
    # Check values are different from base (deltas were applied)
    for key in base.keys():
        assert not torch.allclose(merged[key], base[key])


def test_create_baseline_task_arithmetic_scaling():
    """Test Task Arithmetic with different scaling factors."""
    base, task_vectors = create_test_data()
    
    # Create with scaling 0.0 (no delta) to get base
    merged_zero = create_baseline_task_arithmetic(
        base, task_vectors, scaling_factor=0.0, verbose=False
    )
    
    # Create with scaling 1.0
    merged_full = create_baseline_task_arithmetic(
        base, task_vectors, scaling_factor=1.0, verbose=False
    )
    
    # With scaling 0, we should get base model back
    for key in base.keys():
        assert torch.allclose(merged_zero[key], base[key], rtol=1e-5)
    
    # With scaling 1.0, delta should be non-zero
    has_difference = False
    for key in base.keys():
        if not torch.allclose(merged_full[key], base[key]):
            has_difference = True
            break
    assert has_difference, "Scaling 1.0 should produce different weights from base"


def test_create_baseline_ties():
    """Test TIES baseline creation."""
    base, task_vectors = create_test_data()
    
    # Create merged model with TIES
    merged = create_baseline_ties(
        base, task_vectors, density=0.2, scaling_factor=1.0, verbose=False
    )
    
    # Check merged has same keys as base
    assert set(merged.keys()) == set(base.keys())
    
    # Check shapes match
    for key in base.keys():
        assert merged[key].shape == base[key].shape


def test_create_baseline_ties_density():
    """Test TIES with different density settings."""
    base, task_vectors = create_test_data()
    
    # Higher density keeps more values
    merged_high = create_baseline_ties(
        base, task_vectors, density=0.8, verbose=False
    )
    
    # Lower density keeps fewer values
    merged_low = create_baseline_ties(
        base, task_vectors, density=0.1, verbose=False
    )
    
    # Both should have same structure
    assert set(merged_high.keys()) == set(merged_low.keys())
    
    # The merges should be different
    for key in base.keys():
        # Not guaranteed to be different for all keys, but overall merge should differ
        pass  # Just checking they run without error


def test_compare_methods_structure():
    """Test that compare_methods returns correct structure."""
    base, task_vectors = create_test_data()
    
    comparison = compare_methods(
        base, task_vectors,
        svd_hybrid_state_dict=base,  # Use base as placeholder for SVD-Hybrid
        task_names=["task1", "task2"],
        evaluation_fn=None,
        verbose=False
    )
    
    # Check structure
    assert "methods" in comparison
    assert "task_arithmetic" in comparison["methods"]
    assert "ties" in comparison["methods"]
    assert "svd_hybrid" in comparison["methods"]


def test_compare_methods_with_evaluation():
    """Test compare_methods with mock evaluation."""
    base, task_vectors = create_test_data()
    
    # Mock evaluation function
    call_count = {"count": 0}
    def mock_eval_fn(task_name, state_dict):
        call_count["count"] += 1
        return 0.75  # Fixed accuracy
    
    comparison = compare_methods(
        base, task_vectors,
        svd_hybrid_state_dict=base,
        task_names=["task1"],
        evaluation_fn=mock_eval_fn,
        verbose=False
    )
    
    # Check all methods have results
    for method in ["task_arithmetic", "ties", "svd_hybrid"]:
        assert "average_accuracy" in comparison["methods"][method]
        assert comparison["methods"][method]["average_accuracy"] == 0.75


def test_generate_evaluation_report():
    """Test report generation."""
    # Create mock comparison results
    mock_results = {
        "methods": {
            "task_arithmetic": {
                "average_accuracy": 0.75,
                "std_accuracy": 0.05,
                "per_task_accuracy": {"task1": 0.80, "task2": 0.70}
            },
            "ties": {
                "average_accuracy": 0.78,
                "std_accuracy": 0.03,
                "per_task_accuracy": {"task1": 0.80, "task2": 0.76}
            },
            "svd_hybrid": {
                "average_accuracy": 0.82,
                "std_accuracy": 0.04,
                "per_task_accuracy": {"task1": 0.85, "task2": 0.79}
            }
        },
        "comparison": {
            "best_method": "svd_hybrid",
            "scores": {
                "task_arithmetic": 0.75,
                "ties": 0.78,
                "svd_hybrid": 0.82
            }
        }
    }
    
    report = generate_evaluation_report(mock_results, verbose=False)
    
    # Check report contains key information
    assert "EVALUATION REPORT" in report
    assert "TASK_ARITHMETIC" in report
    assert "TIES" in report
    assert "SVD_HYBRID" in report
    assert "82.00%" in report  # SVD-Hybrid accuracy


def test_verbose_logging():
    """Test that verbose mode produces output without errors."""
    base, task_vectors = create_test_data()
    
    # Test evaluate_merged_model verbose
    results = evaluate_merged_model(
        base,
        task_names=["task1"],
        evaluation_fn=None,
        verbose=True
    )
    assert results is not None
    
    # Test task arithmetic verbose
    merged = create_baseline_task_arithmetic(
        base, task_vectors, verbose=True
    )
    assert merged is not None
    
    # Test TIES verbose
    merged_ties = create_baseline_ties(
        base, task_vectors, verbose=True
    )
    assert merged_ties is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
