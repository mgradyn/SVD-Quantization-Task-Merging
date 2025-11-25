"""
Model definitions for checkpoint loading compatibility.

This module provides model class definitions needed to load checkpoints that were
saved from codebases using `src.modeling`. It enables deserialization of pickled
model objects.

=== WHY THIS MODULE EXISTS ===

PyTorch's torch.save() uses pickle, which serializes not just the model weights
but also the class definition path. When loading a checkpoint that contains a full
model object (not just state_dict), Python needs to import the original module.

If you see: `No module named 'src.modeling'`
It means the checkpoint was saved with a model class from `src.modeling`, and this
module provides the necessary class definitions to load those checkpoints.

=== SUPPORTED MODEL TYPES ===

1. **ImageClassifier**: CLIP-based image classifier with classification head
2. **ImageEncoder**: CLIP image encoder wrapper
3. **ClassificationHead**: Linear classification head

=== USAGE ===

This module is automatically imported when loading checkpoints. You typically
don't need to use it directly:

    # This will work after importing this module
    checkpoint = torch.load("finetuned_model.pt")

If you need to explicitly import for custom unpickling:

    from src.modeling import ImageClassifier, ImageEncoder
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Set


# Whitelist of allowed kwargs that can be set as attributes on model classes
# This is needed for checkpoint compatibility where models may have extra attributes
ALLOWED_KWARGS: Set[str] = {
    # Common model configuration attributes
    'model_name', 'model_type', 'pretrained',
    'input_resolution', 'patch_size', 'width', 'layers', 'heads',
    'output_dim', 'vision_width', 'vision_layers', 'vision_heads',
    'image_size', 'vision_patch_size', 'context_length',
    # Training-related attributes that may be saved
    'training_dataset', 'eval_datasets', 'task_name',
    # CLIP-specific attributes
    'transformer_width', 'transformer_layers', 'transformer_heads',
    'vocab_size', 'token_embedding_dim',
}


def _set_allowed_kwargs(obj: nn.Module, kwargs: Dict[str, Any]) -> None:
    """
    Set only allowed kwargs as attributes on the object.
    
    Args:
        obj: Module to set attributes on
        kwargs: Dictionary of attributes to potentially set
    """
    for key, value in kwargs.items():
        if key in ALLOWED_KWARGS:
            setattr(obj, key, value)
        # Silently ignore unknown kwargs for checkpoint compatibility


class ClassificationHead(nn.Module):
    """
    Linear classification head for image classification.
    
    Takes features from an encoder and produces class logits.
    
    Attributes:
        head: Linear layer mapping features to num_classes
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bias: bool = True
    ):
        """
        Initialize classification head.
        
        Args:
            in_features: Input feature dimension (from encoder)
            num_classes: Number of output classes
            bias: Whether to include bias in linear layer
        """
        super().__init__()
        self.head = nn.Linear(in_features, num_classes, bias=bias)
        self.in_features = in_features
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head."""
        return self.head(x)


class ImageEncoder(nn.Module):
    """
    Wrapper for CLIP image encoder.
    
    Provides a consistent interface for different CLIP model variants.
    Can wrap encoders from open_clip, transformers, or other sources.
    
    Attributes:
        model: The underlying image encoder model
        embed_dim: Output embedding dimension
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        embed_dim: int = 512,
        **kwargs
    ):
        """
        Initialize image encoder wrapper.
        
        Args:
            model: Underlying encoder model (optional, can be set later)
            embed_dim: Output embedding dimension
            **kwargs: Additional arguments for compatibility (filtered for security)
        """
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        # Store allowed kwargs for compatibility with various checkpoint formats
        _set_allowed_kwargs(self, kwargs)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors.
        
        Args:
            images: Input images tensor [B, C, H, W]
            
        Returns:
            Image features [B, embed_dim]
        """
        if self.model is not None:
            return self.model(images)
        raise NotImplementedError("No underlying model set")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for forward, matching CLIP API."""
        return self.forward(images)


class ImageClassifier(nn.Module):
    """
    CLIP-based image classifier.
    
    Combines a CLIP image encoder with a classification head for
    fine-tuned image classification tasks.
    
    This is the main model class used for fine-tuning CLIP on downstream
    tasks like Cars, DTD, EuroSAT, etc.
    
    Attributes:
        image_encoder: CLIP image encoder (or wrapper)
        classification_head: Linear head for classification
        process_images: Whether to preprocess images
    """
    
    def __init__(
        self,
        image_encoder: Optional[nn.Module] = None,
        classification_head: Optional[nn.Module] = None,
        embed_dim: int = 512,
        num_classes: int = 1000,
        process_images: bool = True,
        **kwargs
    ):
        """
        Initialize image classifier.
        
        Args:
            image_encoder: Pre-trained image encoder
            classification_head: Classification head (created if None)
            embed_dim: Embedding dimension from encoder
            num_classes: Number of output classes
            process_images: Whether to apply preprocessing
            **kwargs: Additional arguments for compatibility (filtered for security)
        """
        super().__init__()
        
        self.image_encoder = image_encoder
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.process_images = process_images
        
        # Create classification head if not provided
        if classification_head is not None:
            self.classification_head = classification_head
        else:
            self.classification_head = ClassificationHead(embed_dim, num_classes)
        
        # Store allowed kwargs for compatibility
        _set_allowed_kwargs(self, kwargs)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode images and classify.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Class logits [B, num_classes]
        """
        # Get image features
        if self.image_encoder is not None:
            if hasattr(self.image_encoder, 'encode_image'):
                features = self.image_encoder.encode_image(images)
            else:
                features = self.image_encoder(images)
        else:
            # If no encoder, assume images are already features
            features = images
        
        # Classify
        logits = self.classification_head(features)
        return logits
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get image features without classification."""
        if self.image_encoder is not None:
            if hasattr(self.image_encoder, 'encode_image'):
                return self.image_encoder.encode_image(images)
            return self.image_encoder(images)
        return images


class CLIPEncoder(nn.Module):
    """
    Alternative CLIP encoder wrapper for compatibility.
    
    Some codebases use different naming conventions. This class
    provides compatibility with checkpoints saved as CLIPEncoder.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        embed_dim: int = 512,
        **kwargs
    ):
        """Initialize CLIP encoder wrapper."""
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim
        # Store allowed kwargs for compatibility
        _set_allowed_kwargs(self, kwargs)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images."""
        if self.model is not None:
            return self.model(images)
        raise NotImplementedError("No underlying model set")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for forward."""
        return self.forward(images)


# Alias for different naming conventions that may appear in checkpoints
Classifier = ImageClassifier
CLIPClassifier = ImageClassifier
VisionClassifier = ImageClassifier
