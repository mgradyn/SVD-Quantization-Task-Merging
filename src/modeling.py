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
2. **ImageEncoder**: CLIP image encoder wrapper (supports open_clip)
3. **ClassificationHead**: Linear classification head with optional normalization
4. **MultiHeadImageClassifier**: Classifier with multiple heads for multi-task learning
5. **ImageClassifier_debug**: Debug classifier with two encoders

=== USAGE ===

This module is automatically imported when loading checkpoints. You typically
don't need to use it directly:

    # This will work after importing this module
    checkpoint = torch.load("finetuned_model.pt")

If you need to explicitly import for custom unpickling:

    from src.modeling import ImageClassifier, ImageEncoder

=== COMPATIBILITY ===

This module includes aliases for older checkpoint formats:
- VisualTransformer: Alias for open_clip's VisionTransformer (name changed in newer versions)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Set, List

# Try to import CLIP libraries for checkpoint compatibility
# Priority: Original OpenAI CLIP first, then open_clip as fallback
# This is important because checkpoints created with OpenAI CLIP cannot be
# loaded with open_clip due to different internal class names and structures.

try:
    import clip
    import clip.model
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


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
    # ImageEncoder specific attributes
    'cache_dir', 'openclip_cachedir', 'preweight',
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


def torch_save(obj: Any, filename: str) -> None:
    """Save object using torch.save."""
    torch.save(obj, filename)


def torch_load(filename: str, map_location: str = 'cpu') -> Any:
    """Load object using torch.load."""
    return torch.load(filename, map_location=map_location, weights_only=False)


class ClassificationHead(nn.Linear):
    """
    Linear classification head for image classification.
    
    Takes features from an encoder and produces class logits.
    Optionally normalizes input features before classification.
    
    This class extends nn.Linear to provide direct weight/bias initialization
    from pre-computed embeddings (e.g., text embeddings for zero-shot).
    
    Attributes:
        normalize: Whether to L2-normalize input features
        weight: Classification weights [num_classes, embed_dim]
        bias: Classification biases [num_classes]
    """
    
    def __init__(
        self,
        normalize: bool = True,
        weights: Optional[torch.Tensor] = None,
        biases: Optional[torch.Tensor] = None,
        in_features: Optional[int] = None,
        num_classes: Optional[int] = None
    ):
        """
        Initialize classification head.
        
        Can be initialized either with pre-computed weights or with dimensions.
        
        Args:
            normalize: Whether to L2-normalize input features before classification
            weights: Pre-computed weight matrix [num_classes, embed_dim]
            biases: Pre-computed bias vector [num_classes] (optional)
            in_features: Input feature dimension (used if weights is None)
            num_classes: Number of output classes (used if weights is None)
        """
        if weights is not None:
            output_size, input_size = weights.shape
        else:
            if in_features is None or num_classes is None:
                raise ValueError("Either weights or (in_features, num_classes) must be provided")
            input_size = in_features
            output_size = num_classes
        
        super().__init__(input_size, output_size)
        self.normalize = normalize
        self.in_features = input_size
        self.num_classes = output_size
        
        # Overwrite the weight/bias data (not the Parameter itself) to preserve registration
        if weights is not None:
            self.weight.data.copy_(weights)
        if biases is not None:
            self.bias.data.copy_(biases)
        elif self.bias is not None:
            # Initialize bias to zeros using the existing bias parameter from nn.Linear
            self.bias.data.zero_()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            inputs: Feature tensor [B, embed_dim]
            
        Returns:
            Class logits [B, num_classes]
        """
        if self.normalize:
            # Add small epsilon to avoid division by zero
            inputs = inputs / (inputs.norm(dim=-1, keepdim=True) + 1e-8)
        return super().forward(inputs)
    
    def save(self, filename: str) -> None:
        """Save classification head to file."""
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)
    
    @classmethod
    def load(cls, filename: str) -> 'ClassificationHead':
        """Load classification head from file."""
        print(f'Loading classification head from {filename}')
        return torch_load(filename)


class ImageEncoder(nn.Module):
    """
    Wrapper for CLIP image encoder using open_clip.
    
    Provides a consistent interface for different CLIP model variants.
    Supports loading models from open_clip with custom pretrained weights.
    
    Attributes:
        model: The underlying CLIP model (full model, not just visual)
        train_preprocess: Preprocessing transform for training
        val_preprocess: Preprocessing transform for validation/inference
        cache_dir: Directory for caching checkpoints
    """
    
    def __init__(
        self,
        args: Optional[Any] = None,
        keep_lang: bool = False,
        model: Optional[nn.Module] = None,
        embed_dim: int = 512,
        **kwargs
    ):
        """
        Initialize image encoder wrapper.
        
        Can be initialized either with args (for open_clip loading) or with a model directly.
        
        Args:
            args: Arguments object with model, preweight, openclip_cachedir, cache_dir attributes
            keep_lang: Whether to keep the language transformer (default: False)
            model: Underlying encoder model (optional, can be set later for compatibility)
            embed_dim: Output embedding dimension (used if model is None)
            **kwargs: Additional arguments for compatibility (filtered for security)
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cache_dir = None
        self.train_preprocess = None
        self.val_preprocess = None
        
        if args is not None and OPEN_CLIP_AVAILABLE:
            print(f'Loading {args.model} pre-trained weights.')
            if '__pretrained__' in args.model:
                name, pretrained = args.model.split('__pretrained__')
            else:
                name = args.model
                pretrained = getattr(args, 'preweight', 'openai')
            
            openclip_cachedir = getattr(args, 'openclip_cachedir', None)
            self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
                name, pretrained=pretrained, cache_dir=openclip_cachedir)
            
            self.cache_dir = getattr(args, 'cache_dir', None)
            
            if not keep_lang and hasattr(self.model, 'transformer'):
                delattr(self.model, 'transformer')
        else:
            self.model = model
        
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
            return self.model.encode_image(images)
        raise NotImplementedError("No underlying model set")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for forward, matching CLIP API."""
        return self.forward(images)
    
    def save(self, filename: str) -> None:
        """Save image encoder to file."""
        print(f'Saving image encoder to {filename}')
        torch_save(self, filename)
    
    @classmethod
    def load(cls, filename: str) -> 'ImageEncoder':
        """Load image encoder from file."""
        print(f'Loading image encoder from {filename}')
        return torch_load(filename)


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
        train_preprocess: Preprocessing transform for training (from encoder)
        val_preprocess: Preprocessing transform for validation (from encoder)
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
        self.classification_head = classification_head
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.process_images = process_images
        
        # Inherit preprocessing transforms from encoder if available
        if self.image_encoder is not None:
            if hasattr(self.image_encoder, 'train_preprocess'):
                self.train_preprocess = self.image_encoder.train_preprocess
                self.val_preprocess = self.image_encoder.val_preprocess
            elif hasattr(self.image_encoder, 'model') and hasattr(self.image_encoder.model, 'train_preprocess'):
                self.train_preprocess = self.image_encoder.model.train_preprocess
                self.val_preprocess = self.image_encoder.model.val_preprocess
        
        # Store allowed kwargs for compatibility
        _set_allowed_kwargs(self, kwargs)
    
    def freeze_head(self) -> None:
        """Freeze classification head weights and biases."""
        if self.classification_head is not None:
            self.classification_head.weight.requires_grad_(False)
            if self.classification_head.bias is not None:
                self.classification_head.bias.requires_grad_(False)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode images and classify.
        
        Args:
            inputs: Input images [B, C, H, W]
            
        Returns:
            Class logits [B, num_classes]
        """
        if self.image_encoder is None:
            raise ValueError("image_encoder is None. Cannot perform forward pass.")
        if self.classification_head is None:
            raise ValueError("classification_head is None. Cannot perform forward pass.")
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get image features without classification."""
        if self.image_encoder is not None:
            if hasattr(self.image_encoder, 'encode_image'):
                return self.image_encoder.encode_image(images)
            return self.image_encoder(images)
        return images
    
    def save(self, filename: str) -> None:
        """Save image classifier to file."""
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)
    
    @classmethod
    def load(cls, filename: str) -> 'ImageClassifier':
        """Load image classifier from file."""
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)


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


class ImageClassifier_debug(nn.Module):
    """
    Debug classifier with two image encoders.
    
    This class is useful for debugging and comparing encoder outputs.
    It combines features from two encoders before classification.
    
    Attributes:
        image_encoder: First CLIP image encoder
        image_encoder2: Second CLIP image encoder
        classification_head: Linear head for classification
    """
    
    def __init__(
        self,
        image_encoder: nn.Module,
        image_encoder2: nn.Module,
        classification_head: nn.Module
    ):
        """
        Initialize debug classifier.
        
        Args:
            image_encoder: First image encoder
            image_encoder2: Second image encoder
            classification_head: Classification head
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.image_encoder2 = image_encoder2
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = getattr(self.image_encoder, 'train_preprocess', None)
            self.val_preprocess = getattr(self.image_encoder, 'val_preprocess', None)
    
    def freeze_head(self) -> None:
        """Freeze classification head weights and biases."""
        if self.classification_head is not None:
            self.classification_head.weight.requires_grad_(False)
            if self.classification_head.bias is not None:
                self.classification_head.bias.requires_grad_(False)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with combined features from both encoders.
        
        Args:
            inputs: Input images [B, C, H, W]
            
        Returns:
            Class logits [B, num_classes]
        """
        if self.image_encoder is None or self.image_encoder2 is None:
            raise ValueError("Both image encoders must be set. Cannot perform forward pass.")
        if self.classification_head is None:
            raise ValueError("classification_head is None. Cannot perform forward pass.")
        features = self.image_encoder(inputs)
        features2 = self.image_encoder2(inputs)
        outputs = self.classification_head(features + features2)
        return outputs
    
    def save(self, filename: str) -> None:
        """Save debug classifier to file."""
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)
    
    @classmethod
    def load(cls, filename: str) -> 'ImageClassifier_debug':
        """Load debug classifier from file."""
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)


class MultiHeadImageClassifier(nn.Module):
    """
    Image classifier with multiple classification heads for multi-task learning.
    
    This class allows a single encoder to be shared across multiple tasks,
    each with its own classification head.
    
    Attributes:
        image_encoder: Shared CLIP image encoder
        classification_heads: List of classification heads (one per task)
    """
    
    def __init__(
        self,
        image_encoder: nn.Module,
        classification_heads: List[nn.Module]
    ):
        """
        Initialize multi-head classifier.
        
        Args:
            image_encoder: Shared image encoder
            classification_heads: List of classification heads for different tasks
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = getattr(self.image_encoder, 'train_preprocess', None)
            self.val_preprocess = getattr(self.image_encoder, 'val_preprocess', None)
    
    def freeze_head(self) -> None:
        """Freeze all classification head weights and biases."""
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            if self.classification_heads[idx].bias is not None:
                self.classification_heads[idx].bias.requires_grad_(False)
    
    def forward(self, inputs: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        Forward pass through specified classification head.
        
        Args:
            inputs: Input images [B, C, H, W]
            head_idx: Index of classification head to use
            
        Returns:
            Class logits [B, num_classes] for the specified head
        """
        if self.image_encoder is None:
            raise ValueError("image_encoder is None. Cannot perform forward pass.")
        if head_idx < 0 or head_idx >= len(self.classification_heads):
            raise IndexError(f"head_idx {head_idx} is out of range. Valid range: 0 to {len(self.classification_heads) - 1}")
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs
    
    def save(self, filename: str) -> None:
        """Save multi-head classifier to file."""
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)
    
    @classmethod
    def load(cls, filename: str) -> 'MultiHeadImageClassifier':
        """Load multi-head classifier from file."""
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)


# Alias for different naming conventions that may appear in checkpoints
Classifier = ImageClassifier
CLIPClassifier = ImageClassifier
VisionClassifier = ImageClassifier


# === COMPATIBILITY ALIASES FOR CHECKPOINT LOADING ===
# 
# Some older checkpoints were saved with different class names.
# These aliases allow loading such checkpoints without modification.
#
# VisualTransformer: Older checkpoints used this name for the vision encoder.
#                    Newer versions use VisionTransformer.
#
# IMPORTANT: Checkpoints created with OpenAI CLIP must be loaded using the
# original clip library, not open_clip. The two libraries have different
# internal class structures and cannot deserialize each other's objects.

if CLIP_AVAILABLE:
    # Prefer original OpenAI CLIP library for loading checkpoints
    # This ensures checkpoints created with OpenAI CLIP can be loaded correctly
    from clip.model import VisionTransformer
    VisualTransformer = VisionTransformer
elif OPEN_CLIP_AVAILABLE:
    # Fall back to open_clip if original clip is not available
    from open_clip.model import VisionTransformer
    VisualTransformer = VisionTransformer
else:
    # If neither library is available, create a placeholder class
    class VisualTransformer(nn.Module):
        """Placeholder for VisualTransformer when CLIP libraries are not available."""
        pass
    
    class VisionTransformer(nn.Module):
        """Placeholder for VisionTransformer when CLIP libraries are not available."""
        pass
