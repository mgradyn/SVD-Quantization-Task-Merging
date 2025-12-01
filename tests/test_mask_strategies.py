"""
Tests for mask combination strategies (union, intersection, majority).
"""
import torch
import numpy as np
import pytest
import os
import tempfile
from src.svd_hybrid.mask_loader import (
    combine_masks, 
    state_dict_to_vector, 
    vector_to_state_dict,
    load_tall_mask_file,
    load_task_masks
)


def test_union_strategy():
    """Test union mask strategy produces OR of all masks."""
    # Create 3 task masks for 2 parameters
    task_masks = {
        "task1": {
            "layer1.weight": torch.tensor([[True, False, False],
                                          [False, True, False]]),
            "layer2.weight": torch.tensor([True, False, True])
        },
        "task2": {
            "layer1.weight": torch.tensor([[False, True, False],
                                          [False, False, True]]),
            "layer2.weight": torch.tensor([False, True, False])
        },
        "task3": {
            "layer1.weight": torch.tensor([[False, False, True],
                                          [True, False, False]]),
            "layer2.weight": torch.tensor([False, False, False])
        }
    }
    
    combined = combine_masks(task_masks, strategy="union")
    
    # Union should have True where ANY task has True
    expected_layer1 = torch.tensor([[True, True, True],
                                    [True, True, True]])
    expected_layer2 = torch.tensor([True, True, True])
    
    assert torch.equal(combined["layer1.weight"], expected_layer1)
    assert torch.equal(combined["layer2.weight"], expected_layer2)


def test_intersection_strategy():
    """Test intersection mask strategy produces AND of all masks."""
    task_masks = {
        "task1": {
            "layer1.weight": torch.tensor([[True, True, False],
                                          [True, True, False]]),
            "layer2.weight": torch.tensor([True, True, True])
        },
        "task2": {
            "layer1.weight": torch.tensor([[True, False, True],
                                          [True, True, True]]),
            "layer2.weight": torch.tensor([True, False, True])
        },
        "task3": {
            "layer1.weight": torch.tensor([[True, True, True],
                                          [True, False, False]]),
            "layer2.weight": torch.tensor([True, True, False])
        }
    }
    
    combined = combine_masks(task_masks, strategy="intersection")
    
    # Intersection should have True only where ALL tasks have True
    expected_layer1 = torch.tensor([[True, False, False],
                                    [True, False, False]])
    expected_layer2 = torch.tensor([True, False, False])
    
    assert torch.equal(combined["layer1.weight"], expected_layer1)
    assert torch.equal(combined["layer2.weight"], expected_layer2)


def test_majority_strategy():
    """Test majority mask strategy keeps coordinates where >= half tasks have True."""
    # 4 tasks
    task_masks = {
        "task1": {
            "layer.weight": torch.tensor([True, True, False, False, True])
        },
        "task2": {
            "layer.weight": torch.tensor([True, False, True, False, False])
        },
        "task3": {
            "layer.weight": torch.tensor([True, False, False, True, True])
        },
        "task4": {
            "layer.weight": torch.tensor([False, False, False, False, True])
        }
    }
    
    combined = combine_masks(task_masks, strategy="majority")
    
    # Majority voting with 4 tasks: need >= 2 tasks
    # Position 0: 3 True -> True
    # Position 1: 1 True -> False
    # Position 2: 1 True -> False
    # Position 3: 1 True -> False
    # Position 4: 3 True -> True
    expected = torch.tensor([True, False, False, False, True])
    
    assert torch.equal(combined["layer.weight"], expected)


def test_majority_strategy_odd_tasks():
    """Test majority with odd number of tasks."""
    # 3 tasks - need >= 2 for majority
    task_masks = {
        "task1": {
            "param": torch.tensor([True, True, False, False])
        },
        "task2": {
            "param": torch.tensor([True, False, True, False])
        },
        "task3": {
            "param": torch.tensor([False, True, True, False])
        }
    }
    
    combined = combine_masks(task_masks, strategy="majority")
    
    # Position 0: 2 True -> True
    # Position 1: 2 True -> True
    # Position 2: 2 True -> True
    # Position 3: 0 True -> False
    expected = torch.tensor([True, True, True, False])
    
    assert torch.equal(combined["param"], expected)


def test_empty_masks():
    """Test handling of empty mask dictionary."""
    task_masks = {}
    
    combined = combine_masks(task_masks, strategy="union")
    
    assert combined == {}


def test_single_task_all_strategies():
    """Test that all strategies produce same result for single task."""
    task_masks = {
        "task1": {
            "weight": torch.tensor([[True, False], [False, True]])
        }
    }
    
    union_result = combine_masks(task_masks, strategy="union")
    intersection_result = combine_masks(task_masks, strategy="intersection")
    majority_result = combine_masks(task_masks, strategy="majority")
    
    # All should be identical for single task
    assert torch.equal(union_result["weight"], task_masks["task1"]["weight"])
    assert torch.equal(intersection_result["weight"], task_masks["task1"]["weight"])
    assert torch.equal(majority_result["weight"], task_masks["task1"]["weight"])


def test_mask_size_consistency():
    """Test that combined masks have expected size."""
    task_masks = {
        "task1": {
            "param1": torch.ones(100, dtype=torch.bool),
            "param2": torch.zeros(50, dtype=torch.bool)
        },
        "task2": {
            "param1": torch.zeros(100, dtype=torch.bool),
            "param2": torch.ones(50, dtype=torch.bool)
        }
    }
    
    for strategy in ["union", "intersection", "majority"]:
        combined = combine_masks(task_masks, strategy=strategy)
        
        assert combined["param1"].shape == torch.Size([100])
        assert combined["param2"].shape == torch.Size([50])
        assert combined["param1"].dtype == torch.bool
        assert combined["param2"].dtype == torch.bool


def test_union_produces_largest_mask():
    """Test that union strategy produces the largest combined mask."""
    task_masks = {
        "task1": {
            "weight": torch.tensor([True, False, True, False])
        },
        "task2": {
            "weight": torch.tensor([False, True, True, False])
        }
    }
    
    union_result = combine_masks(task_masks, strategy="union")
    intersection_result = combine_masks(task_masks, strategy="intersection")
    
    # Union should have more or equal True values than intersection
    union_count = union_result["weight"].sum().item()
    intersection_count = intersection_result["weight"].sum().item()
    
    assert union_count >= intersection_count


def test_intersection_produces_smallest_mask():
    """Test that intersection strategy produces the smallest combined mask."""
    task_masks = {
        "task1": {
            "weight": torch.tensor([True, True, True, False])
        },
        "task2": {
            "weight": torch.tensor([True, True, False, True])
        },
        "task3": {
            "weight": torch.tensor([True, False, True, True])
        }
    }
    
    union_result = combine_masks(task_masks, strategy="union")
    intersection_result = combine_masks(task_masks, strategy="intersection")
    majority_result = combine_masks(task_masks, strategy="majority")
    
    # Intersection should have fewest True values
    union_count = union_result["weight"].sum().item()
    intersection_count = intersection_result["weight"].sum().item()
    majority_count = majority_result["weight"].sum().item()
    
    assert intersection_count <= majority_count <= union_count


def test_all_false_masks():
    """Test handling when all masks are False."""
    task_masks = {
        "task1": {"param": torch.zeros(10, dtype=torch.bool)},
        "task2": {"param": torch.zeros(10, dtype=torch.bool)},
    }
    
    # All strategies should produce all False
    for strategy in ["union", "intersection", "majority"]:
        combined = combine_masks(task_masks, strategy=strategy)
        assert combined["param"].sum().item() == 0


def test_all_true_masks():
    """Test handling when all masks are True."""
    task_masks = {
        "task1": {"param": torch.ones(10, dtype=torch.bool)},
        "task2": {"param": torch.ones(10, dtype=torch.bool)},
    }
    
    # All strategies should produce all True
    for strategy in ["union", "intersection", "majority"]:
        combined = combine_masks(task_masks, strategy=strategy)
        assert combined["param"].sum().item() == 10


# ==================== Tests for state_dict/vector conversion ====================

def test_state_dict_to_vector_basic():
    """Test basic state dict to vector conversion."""
    state_dict = {
        "layer1.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer1.bias": torch.tensor([5.0, 6.0])
    }
    
    vector = state_dict_to_vector(state_dict)
    
    # Should have 4 + 2 = 6 elements
    assert vector.shape == torch.Size([6])


def test_state_dict_to_vector_with_remove_keys():
    """Test state dict to vector with excluded keys."""
    state_dict = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([3.0, 4.0]),
        "running_mean": torch.tensor([5.0, 6.0])  # should be skipped
    }
    
    vector = state_dict_to_vector(state_dict, remove_keys=["running_mean"])
    
    # Should have 2 + 2 = 4 elements (excluding running_mean)
    assert vector.shape == torch.Size([4])


def test_vector_to_state_dict_basic():
    """Test basic vector to state dict reconstruction."""
    reference = {
        "layer1.weight": torch.randn(2, 3),
        "layer1.bias": torch.randn(3)
    }
    
    # Create a vector with correct total elements (6 + 3 = 9)
    vector = torch.arange(9, dtype=torch.float)
    
    reconstructed = vector_to_state_dict(vector, reference)
    
    assert "layer1.weight" in reconstructed
    assert "layer1.bias" in reconstructed
    assert reconstructed["layer1.weight"].shape == torch.Size([2, 3])
    assert reconstructed["layer1.bias"].shape == torch.Size([3])


def test_state_dict_vector_roundtrip():
    """Test that state_dict -> vector -> state_dict preserves structure."""
    original = {
        "encoder.layer1.weight": torch.randn(10, 5),
        "encoder.layer1.bias": torch.randn(10),
        "decoder.weight": torch.randn(5, 10)
    }
    
    # Convert to vector
    vector = state_dict_to_vector(original)
    
    # Convert back to state dict
    reconstructed = vector_to_state_dict(vector, original)
    
    # Verify shapes match
    for key in original:
        assert key in reconstructed
        assert reconstructed[key].shape == original[key].shape


def test_state_dict_vector_values_preserved():
    """Test that values are preserved during roundtrip."""
    original = {
        "a": torch.tensor([1.0, 2.0, 3.0]),
        "b": torch.tensor([[4.0, 5.0], [6.0, 7.0]])
    }
    
    vector = state_dict_to_vector(original)
    reconstructed = vector_to_state_dict(vector, original)
    
    for key in original:
        assert torch.allclose(reconstructed[key], original[key])


# ==================== Tests for TALL mask file loading ====================

def test_load_tall_mask_file():
    """Test loading TALL mask from packed format."""
    # Create a reference state dict
    reference = {
        "layer1.weight": torch.randn(4, 4),  # 16 elements
        "layer1.bias": torch.randn(4)         # 4 elements
    }
    # Total: 20 elements
    
    # Create packed mask data for two tasks
    # Task 1: first 10 True, rest False
    mask1_flat = np.array([1]*10 + [0]*10, dtype=np.uint8)
    mask1_packed = np.packbits(mask1_flat)
    
    # Task 2: alternating True/False
    mask2_flat = np.array([1, 0]*10, dtype=np.uint8)
    mask2_packed = np.packbits(mask2_flat)
    
    packed_masks = {
        "Cars": mask1_packed,
        "DTD": mask2_packed
    }
    
    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        mask_path = os.path.join(tmpdir, "TALL_mask_2task.npy")
        torch.save(packed_masks, mask_path)
        
        # Load the masks
        task_masks = load_tall_mask_file(mask_path, reference)
        
        # Verify structure
        assert "Cars" in task_masks
        assert "DTD" in task_masks
        assert "layer1.weight" in task_masks["Cars"]
        assert "layer1.bias" in task_masks["Cars"]
        
        # Verify dtypes are bool
        assert task_masks["Cars"]["layer1.weight"].dtype == torch.bool
        assert task_masks["DTD"]["layer1.bias"].dtype == torch.bool


def test_load_task_masks_with_tall_format():
    """Test load_task_masks finds and loads TALL mask format."""
    # Create reference state dict
    reference = {
        "param": torch.randn(8)  # 8 elements
    }
    
    # Create packed masks
    mask1 = np.packbits(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8))
    mask2 = np.packbits(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8))
    
    packed_masks = {
        "Task1": mask1,
        "Task2": mask2
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as TALL mask format
        mask_path = os.path.join(tmpdir, "TALL_mask_2task.npy")
        torch.save(packed_masks, mask_path)
        
        # Load using load_task_masks
        task_masks = load_task_masks(
            tmpdir,
            ["Task1", "Task2"],
            reference_state_dict=reference
        )
        
        assert "Task1" in task_masks
        assert "Task2" in task_masks
        assert task_masks["Task1"]["param"].shape == torch.Size([8])


def test_load_task_masks_fallback_to_individual():
    """Test load_task_masks falls back to individual files when TALL not found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create individual mask files
        mask1 = {"param": torch.tensor([True, False, True])}
        mask2 = {"param": torch.tensor([False, True, False])}
        
        torch.save(mask1, os.path.join(tmpdir, "Task1_mask.pt"))
        torch.save(mask2, os.path.join(tmpdir, "Task2_mask.pt"))
        
        # Load without reference_state_dict (should use individual files)
        task_masks = load_task_masks(tmpdir, ["Task1", "Task2"])
        
        assert "Task1" in task_masks
        assert "Task2" in task_masks
        assert torch.equal(task_masks["Task1"]["param"], mask1["param"])
        assert torch.equal(task_masks["Task2"]["param"], mask2["param"])


def test_load_tall_mask_8_tasks():
    """Test loading TALL mask with 8 standard tasks."""
    # Standard 8 tasks
    tasks = ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
    
    # Create reference state dict
    reference = {
        "encoder.weight": torch.randn(8, 8),  # 64 elements
        "encoder.bias": torch.randn(8)         # 8 elements
    }
    # Total: 72 elements
    
    # Create packed masks for all 8 tasks
    packed_masks = {}
    for i, task in enumerate(tasks):
        # Each task has a different mask pattern
        mask = np.array([(j % (i + 1) == 0) for j in range(72)], dtype=np.uint8)
        packed_masks[task] = np.packbits(mask)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mask_path = os.path.join(tmpdir, "TALL_mask_8task.npy")
        torch.save(packed_masks, mask_path)
        
        # Load the masks
        task_masks = load_tall_mask_file(mask_path, reference)
        
        # Verify all 8 tasks are loaded
        assert len(task_masks) == 8
        for task in tasks:
            assert task in task_masks
            assert "encoder.weight" in task_masks[task]
            assert "encoder.bias" in task_masks[task]
            assert task_masks[task]["encoder.weight"].shape == torch.Size([8, 8])
            assert task_masks[task]["encoder.bias"].shape == torch.Size([8])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
