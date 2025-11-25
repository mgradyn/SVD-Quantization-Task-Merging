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
    from src.modeling import ImageClassifier, ClassificationHead
    
    # Create classification head with pre-computed weights
    weights = torch.randn(47, 768)  # DTD has 47 classes
    head = ClassificationHead(normalize=True, weights=weights)
    
    # Create model with the classification head
    model = ImageClassifier(
        image_encoder=None,
        classification_head=head,
        embed_dim=768,
        num_classes=47
    )
    
    # Set specific weights to verify reconstruction
    model.classification_head.weight.data.fill_(0.5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        
        # Save full model (not just state_dict)
        torch.save(model, path)
        
        # Load it back
        loaded = torch.load(path, weights_only=False)
        
        # Verify
        assert isinstance(loaded, ImageClassifier)
        assert loaded.num_classes == 47
        expected_weight = torch.ones_like(loaded.classification_head.weight) * 0.5
        assert torch.allclose(loaded.classification_head.weight, expected_weight)


def test_model_state_dict():
    """Test that state_dict can be saved and loaded."""
    from src.modeling import ImageClassifier, ClassificationHead
    
    # Create classification head with dimensions
    head = ClassificationHead(normalize=True, in_features=512, num_classes=10)
    
    model = ImageClassifier(
        image_encoder=None,
        classification_head=head,
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
        new_head = ClassificationHead(normalize=True, in_features=512, num_classes=10)
        new_model = ImageClassifier(
            image_encoder=None,
            classification_head=new_head,
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
    
    # VisualTransformer alias should exist for checkpoint loading compatibility
    assert hasattr(modeling, 'VisualTransformer')
    
    # VisionTransformer should also be available
    assert hasattr(modeling, 'VisionTransformer')
    
    # VisualTransformer should be an alias for VisionTransformer
    assert modeling.VisualTransformer is modeling.VisionTransformer
    
    # New classes should exist
    assert hasattr(modeling, 'ImageClassifier_debug')
    assert hasattr(modeling, 'MultiHeadImageClassifier')


def test_import_via_full_path():
    """Test that module is importable via the full path pickle would use."""
    import importlib
    
    # This is what torch.load() does internally when unpickling
    module = importlib.import_module('src.modeling')
    
    assert hasattr(module, 'ImageClassifier')
    assert hasattr(module, 'ImageEncoder')


def test_multi_head_image_classifier():
    """Test MultiHeadImageClassifier basic functionality."""
    from src.modeling import MultiHeadImageClassifier, ClassificationHead
    
    # Create a simple mock encoder
    class MockEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 512)
            self.train_preprocess = None
            self.val_preprocess = None
        
        def forward(self, x):
            return self.linear(x.view(x.size(0), -1)[:, :3])
    
    encoder = MockEncoder()
    heads = [
        ClassificationHead(normalize=True, in_features=512, num_classes=10),
        ClassificationHead(normalize=True, in_features=512, num_classes=20)
    ]
    
    classifier = MultiHeadImageClassifier(encoder, heads)
    
    # Test forward pass with different heads
    x = torch.randn(4, 3, 224, 224)
    output0 = classifier(x, head_idx=0)
    output1 = classifier(x, head_idx=1)
    
    assert output0.shape == (4, 10)
    assert output1.shape == (4, 20)


def test_classification_head_with_weights():
    """Test ClassificationHead initialized with pre-computed weights."""
    from src.modeling import ClassificationHead
    
    # Create pre-computed weights (e.g., from text embeddings)
    weights = torch.randn(100, 512)  # 100 classes, 512 features
    biases = torch.randn(100)
    
    head = ClassificationHead(normalize=True, weights=weights, biases=biases)
    
    # Test forward pass
    x = torch.randn(4, 512)
    output = head(x)
    
    assert output.shape == (4, 100)
    assert head.normalize is True


def test_classification_head_normalize():
    """Test that ClassificationHead normalizes inputs when normalize=True."""
    from src.modeling import ClassificationHead
    
    head = ClassificationHead(normalize=True, in_features=512, num_classes=10)
    
    # Create input that is not normalized
    x = torch.randn(4, 512) * 10  # Scale by 10 to make it not unit norm
    
    # The head should normalize internally, so we can verify the output is consistent
    output1 = head(x)
    output2 = head(x * 2)  # Scaling input shouldn't matter if normalized
    
    # Outputs should be the same if normalization is applied
    assert torch.allclose(output1, output2, atol=1e-5)


def test_clip_library_availability():
    """Test that CLIP library availability flags are correctly set."""
    from src import modeling
    
    # At least one of the CLIP libraries should be available
    assert modeling.CLIP_AVAILABLE or modeling.OPEN_CLIP_AVAILABLE, \
        "At least one CLIP library should be available"


def test_vision_transformer_from_clip():
    """Test that VisionTransformer comes from original CLIP when available."""
    from src import modeling
    import inspect
    
    # If original CLIP is available, VisionTransformer should come from it
    if modeling.CLIP_AVAILABLE:
        source_file = inspect.getfile(modeling.VisionTransformer)
        # Check that it's from clip, not open_clip
        assert 'clip/model.py' in source_file or 'clip\\model.py' in source_file, \
            f"VisionTransformer should come from clip library when available, got: {source_file}"
        # Make sure it's not from open_clip
        assert 'open_clip' not in source_file, \
            f"VisionTransformer should NOT come from open_clip when original clip is available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
