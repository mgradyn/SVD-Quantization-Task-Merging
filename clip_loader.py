import torch
import re
from typing import Dict
try:
    import clip  # OpenAI CLIP
except ImportError:
    clip = None

def load_clip_vit_b32(device: str = "cpu"):
    """
    Loads base CLIP ViT-B/32 (text + visual). Requires 'clip' package.
    """
    if clip is None:
        raise ImportError("Please install openai clip: pip install git+https://github.com/openai/CLIP.git")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

def extract_visual_state_dict(model) -> Dict[str, torch.Tensor]:
    """
    Extract only visual backbone weights for merging.
    """
    sd = model.state_dict()
    visual_sd = {k: v.clone().detach() for k, v in sd.items() if k.startswith("visual.")}
    return visual_sd

def filter_layers_for_light_run(full_sd: Dict[str, torch.Tensor],
                                blocks=(0, 6, 11),
                                layer_types=("attn.in_proj_weight", "mlp.fc1.weight", "mlp.fc2.weight")):
    keep = {}
    pattern_blocks = [f"resblocks.{b}." for b in blocks]
    for name, w in full_sd.items():
        if any(pb in name for pb in pattern_blocks) and any(lt in name for lt in layer_types):
            keep[name] = w
    return keep