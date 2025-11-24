"""
Tests for mask combination strategies (union, intersection, majority).
"""
import torch
import pytest
from src.svd_hybrid.mask_loader import combine_masks


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
