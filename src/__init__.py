# Source package

# Import modeling module to ensure it's available for checkpoint loading
# This allows torch.load() to find the model classes when loading checkpoints
# that were saved with models from src.modeling
try:
    from . import modeling
except ImportError:
    # modeling may not be available in test environments without open_clip
    pass
