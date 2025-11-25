# Source package

# Import modeling module to ensure it's available for checkpoint loading
# This allows torch.load() to find the model classes when loading checkpoints
# that were saved with models from src.modeling
from . import modeling
