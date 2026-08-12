"""Feature extraction helpers for Phikon and ResNet backbones.

Two extraction entry points, and why they are interchangeable
------------------------------------------------------------
:func:`extract_features` loads a model and extracts in one call. It is the bulk
path, used when no feature cache is configured.

:func:`load_model_components` + :func:`extract_features_from_model` split the same
work so the weights are loaded once and reused across slides. This is the cache
path in ``run_all.py``.

**The two produce the same numbers**, and the feature cache is only legitimate
because they do. Their inner loops are identical (same batch size of 32, same
preprocessing, same CLS-token slice), and both backbones are batch-composition
independent at inference: Phikon is a ViT using LayerNorm, and the ResNet variants
run under ``model.eval()`` so BatchNorm uses running statistics rather than batch
statistics. Splitting a cohort into per-slide batches therefore cannot change any
individual patch's embedding. If a backbone with batch-dependent inference is ever
added, that equivalence breaks and the cache becomes invalid.

The one caveat is floating point: differing final-batch sizes can select different
cuDNN kernels, so cached and freshly-extracted features may differ in the last
ULP or so. Comparisons between a cached run and a non-cached run should not expect
bitwise equality; comparisons between two cached runs should.

Output shapes
-------------
Phikon returns the CLS token of the last hidden state, 768-dim. The ResNet
variants return pooled convolutional features (2048-dim for resnet50/101,
512-dim for resnet18). Both return float32 ``(N, D)``.

**Nothing in the returned array records which model produced it.** The arrays are
saved to the cache as bare ``.npy`` with no metadata, and ``run_all.py``'s cache
guard compares N but not D — so a cache populated by one backbone will be accepted
by a run configured for another. See the cache contract comment in
``run_all.py:run_pipeline`` for the full consequences.
"""

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


def get_model(model_name: str, device: torch.device):
    """Load a model and optional processor."""
    print(f"  Loading {model_name}...")

    if model_name in ("resnet18", "resnet50", "resnet101"):
        from torchvision import models
        from torchvision.models import (
            ResNet18_Weights, ResNet50_Weights, ResNet101_Weights,
        )
        weights_map = {
            "resnet18":  (models.resnet18,  ResNet18_Weights.IMAGENET1K_V1),
            "resnet50":  (models.resnet50,  ResNet50_Weights.IMAGENET1K_V2),
            "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
        }
        factory, weights = weights_map[model_name]
        model = factory(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        return model.to(device), None

    if model_name in ("phikon", "phikon-v2"):
        from transformers import AutoModel, AutoImageProcessor
        hub_name = f"owkin/{model_name}"
        model = AutoModel.from_pretrained(hub_name, trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained(hub_name, use_fast=True)
        return model.to(device), processor

    raise ValueError(f"Unsupported model: {model_name}")


def extract_features(
    patches: np.ndarray,
    model_name: str = "phikon",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract feature vectors for a batch of patches.

    Args:
        patches: (N, H, W, 3) uint8 array
        model_name: Model identifier
        batch_size: GPU batch size

    Returns:
        features: (N, D) float32 array  (D=768 for Phikon)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model, processor = get_model(model_name, device)
    model.eval()

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting features"):
            batch = patches[i : i + batch_size]

            if processor:  # Transformers (Phikon)
                pil_batch = [Image.fromarray(p) for p in batch]
                inputs = processor(images=pil_batch, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                batch_feats = outputs.last_hidden_state[:, 0].cpu().numpy()  # CLS token
            else:  # ResNet
                tensors = [
                    torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                    for p in batch
                ]
                batch_input = torch.stack(tensors).to(device)
                output = model(batch_input)
                batch_feats = output.squeeze(-1).squeeze(-1).cpu().numpy()

            features.append(batch_feats)

    return np.vstack(features)


def load_model_components(model_name: str = "phikon"):
    """Load model once for repeated per-slide inference. Returns (model, processor, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_model(model_name, device)
    model.eval()
    return model, processor, device


def extract_features_from_model(
    patches: np.ndarray,
    model,
    processor,
    device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract features using a pre-loaded model (avoids reloading weights each call)."""
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting features"):
            batch = patches[i : i + batch_size]
            if processor:
                pil_batch = [Image.fromarray(p) for p in batch]
                inputs = processor(images=pil_batch, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                batch_feats = outputs.last_hidden_state[:, 0].cpu().numpy()
            else:
                tensors = [
                    torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                    for p in batch
                ]
                batch_input = torch.stack(tensors).to(device)
                output = model(batch_input)
                batch_feats = output.squeeze(-1).squeeze(-1).cpu().numpy()
            features.append(batch_feats)
    return np.vstack(features)
