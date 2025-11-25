"""
Tests for src.modeling module.

This module tests that the modeling classes are properly defined and can be
used for saving/loading model checkpoints.
"""
import torch
import pytest
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_modeling_module_imports():
    """Test that src.modeling can be imported."""
    from src.modeling import ImageClassifier, ImageEncoder, ClassificationHead
    
    assert ImageClassifier is not None
    assert ImageEncoder is not None
    assert ClassificationHead is not None


def test_classification_head():
    """Test ClassificationHead basic functionality."""
    from src.modeling import ClassificationHead
    
    head = ClassificationHead(in_features=512, num_classes=10)
    
    # Test forward pass
    x = torch.randn(4, 512)
    output = head(x)
    
    assert output.shape == (4, 10)


def test_image_classifier():
    """Test ImageClassifier basic functionality."""
    from src.modeling import ImageClassifier
    
    classifier = ImageClassifier(
        image_encoder=None,
        embed_dim=512,
        num_classes=100
    )
    
    assert classifier.num_classes == 100
    assert classifier.embed_dim == 512


def test_model_save_load_full():
    """Test that full model can be saved and loaded via pickle."""
    from src.modeling import ImageClassifier
    
    # Create model
    model = ImageClassifier(
        image_encoder=None,
        embed_dim=768,
        num_classes=47  # DTD has 47 classes
    )
    
    # Set specific weights to verify reconstruction
    model.classification_head.head.weight.data.fill_(0.5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        
        # Save full model (not just state_dict)
        torch.save(model, path)
        
        # Load it back
        loaded = torch.load(path, weights_only=False)
        
        # Verify
        assert isinstance(loaded, ImageClassifier)
        assert loaded.num_classes == 47
        expected_weight = torch.ones_like(loaded.classification_head.head.weight) * 0.5
        assert torch.allclose(loaded.classification_head.head.weight, expected_weight)


def test_model_state_dict():
    """Test that state_dict can be saved and loaded."""
    from src.modeling import ImageClassifier
    
    model = ImageClassifier(
        image_encoder=None,
        embed_dim=512,
        num_classes=10
    )
    
    state_dict = model.state_dict()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "state_dict.pt")
        
        # Save state dict
        torch.save(state_dict, path)
        
        # Load it back - weights_only=False needed because we're testing
        # full pickle loading as used by checkpoint loading
        loaded = torch.load(path, weights_only=False)
        
        # Create new model and load state dict
        new_model = ImageClassifier(
            image_encoder=None,
            embed_dim=512,
            num_classes=10
        )
        new_model.load_state_dict(loaded)


def test_aliases_exist():
    """Test that class aliases are available for compatibility."""
    from src import modeling
    
    # These aliases should exist for checkpoint compatibility
    assert hasattr(modeling, 'Classifier')
    assert hasattr(modeling, 'CLIPClassifier')
    assert hasattr(modeling, 'VisionClassifier')
    assert hasattr(modeling, 'CLIPEncoder')
    
    # They should be aliases to the main classes
    assert modeling.Classifier is modeling.ImageClassifier
    assert modeling.CLIPClassifier is modeling.ImageClassifier


def test_import_via_full_path():
    """Test that module is importable via the full path pickle would use."""
    import importlib
    
    # This is what torch.load() does internally when unpickling
    module = importlib.import_module('src.modeling')
    
    assert hasattr(module, 'ImageClassifier')
    assert hasattr(module, 'ImageEncoder')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
