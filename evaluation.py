import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from typing import Dict, Callable

# Minimal dataset adapters; expand as needed.
from torchvision.datasets import StanfordCars, EuroSAT
from torchvision.datasets import ImageFolder
# DTD & SUN397 might require custom loaders or existing torchvision wrappers.
# For simplicity, define placeholders:
# - DTD: torchvision.datasets.DTD (available in newer versions).
# - SUN397: not in torchvision; may need external loader. Here we treat as placeholder.

try:
    from torchvision.datasets import DTD
    HAVE_DTD = True
except ImportError:
    HAVE_DTD = False

def build_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711))  # CLIP mean/std
    ])

def sample_subset(dataset, max_items=400):
    indices = list(range(min(max_items, len(dataset))))
    return torch.utils.data.Subset(dataset, indices)

def evaluate_accuracy(clip_model, visual_sd: Dict[str, torch.Tensor],
                      dataset, preprocess: Callable,
                      device="cpu"):
    # Load merged weights into model (visual part only)
    with torch.no_grad():
        for k, v in visual_sd.items():
            if k in clip_model.state_dict():
                clip_model.state_dict()[k].copy_(v)
    clip_model.eval()

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, "targets"):
                imgs, labels = batch.data, batch.targets
            elif isinstance(batch, tuple):
                imgs, labels = batch
            else:
                continue
            imgs = imgs.to(device)
            labels = labels.to(device)
            image_features = clip_model.encode_image(imgs)
            # Simple linear probe stand-in: nearest to class centroid (placeholder).
            # For real evaluation, train a small classifier on top of merged backbone features.
            logits = image_features @ image_features.T  # dummy self similarity
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            if total >= 400:  # light subset
                break
    return correct / max(total, 1)