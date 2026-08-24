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
guard compares N but not D. A cache populated by one backbone is therefore accepted
by a run configured for another. See the cache contract comment in
``run_all.py:run_pipeline`` for the full consequences.
"""

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


def get_model(model_name: str, device: torch.device):
    """Return ``(model, processor)`` for ``model_name``, moved onto ``device``.

    ``processor`` is None for the ResNet variants and an ``AutoImageProcessor``
    for the Phikon ones. Callers branch on it to decide how to preprocess, so a
    None processor means "this is a ResNet" throughout this module.

    The ResNet branch strips the final classification layer, leaving pooled
    convolutional features. The returned model is NOT in eval mode; callers set
    that, and both of them do.

    Downloads weights on first use, from torchvision's cache for ResNet and from
    the HuggingFace hub for Phikon. On a compute node without internet this is
    where a run fails, so populate the caches from a login node first.

    Raises ValueError on an unrecognised name rather than falling back.
    """
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
    """Embed every patch, loading the model for this call only.

    The bulk path, used when no feature cache is configured. Loading weights per
    call is why the cache path exists.

    Args:
        patches: (N, H, W, 3) uint8 array. Patch size is not checked here; both
            backbones resize internally, so a wrong patch size produces plausible
            features rather than an error.
        model_name: passed to :func:`get_model`.
        batch_size: inference batch size, not a scientific parameter. Both
            backbones are batch-composition independent, so this changes speed
            and memory only. See the module docstring.

    Returns:
        (N, D) float32, row i corresponding to ``patches[i]``. D is 768 for
        Phikon, 2048 for resnet50/101, 512 for resnet18.

    Selects CUDA when available and falls back to CPU silently. A cohort-sized
    extraction on CPU takes hours rather than minutes, so check the logged device
    line if a job runs long.

    Assumes ``patches`` is non-empty: ``np.vstack`` raises on an empty list.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model, processor = get_model(model_name, device)
    model.eval()

    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting features"):
            batch = patches[i : i + batch_size]

            if processor:
                # Phikon. The processor handles resizing and normalization to
                # whatever the checkpoint was trained with, so nothing is
                # hardcoded here. [:, 0] takes the CLS token, which is the
                # sequence position ViT reserves for a whole-image summary; the
                # remaining positions are per-tile and are discarded.
                pil_batch = [Image.fromarray(p) for p in batch]
                inputs = processor(images=pil_batch, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                batch_feats = outputs.last_hidden_state[:, 0].cpu().numpy()
            else:
                # ResNet. WARNING: this scales to [0, 1] and stops. The
                # torchvision IMAGENET1K weights were trained on inputs further
                # normalized by the ImageNet channel mean and std, which is not
                # applied here, so the ResNet backbones see out-of-distribution
                # inputs. The features are still deterministic and still cluster,
                # which is why this was never caught, but they are not what those
                # weights were calibrated to produce. Phikon is unaffected: its
                # processor does its own normalization.
                #
                # Fixing this changes every ResNet embedding, so it is a
                # behaviour change, not a cleanup. No recorded result uses a
                # ResNet backbone; every run used Phikon.
                tensors = [
                    torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                    for p in batch
                ]
                batch_input = torch.stack(tensors).to(device)
                # squeeze twice to drop the trailing 1x1 spatial dims left by
                # the average pool.
                output = model(batch_input)
                batch_feats = output.squeeze(-1).squeeze(-1).cpu().numpy()

            features.append(batch_feats)

    return np.vstack(features)


def load_model_components(model_name: str = "phikon"):
    """Load the model once so it can be reused across slides.

    Returns ``(model, processor, device)`` for :func:`extract_features_from_model`.
    The model is already in eval mode, which is what makes the ResNet variants
    batch-composition independent, so callers must not put it back in train mode.

    This is the cache path's entry point: ``run_all.py`` calls it once and then
    calls the extractor per slide.
    """
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
    """Embed patches with an already-loaded model.

    Same numbers as :func:`extract_features` given the same patches; the inner
    loop is identical and the module docstring explains why that equivalence
    holds. It exists to avoid reloading weights once per slide.

    Args:
        patches: (N, H, W, 3) uint8.
        model, processor, device: exactly the triple
            :func:`load_model_components` returned. A None processor selects the
            ResNet preprocessing branch, so passing a Phikon model with a None
            processor silently takes the wrong path.

    Returns:
        (N, D) float32, aligned row-for-row with ``patches``.

    Assumes ``model`` is in eval mode and ``patches`` is non-empty.
    """
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Extracting features"):
            batch = patches[i : i + batch_size]
            # Kept byte-for-byte in step with extract_features. The feature cache
            # is only valid while these two loops agree, so edit both or neither.
            if processor:
                pil_batch = [Image.fromarray(p) for p in batch]
                inputs = processor(images=pil_batch, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                batch_feats = outputs.last_hidden_state[:, 0].cpu().numpy()
            else:
                # Missing ImageNet normalization here too. See the note in
                # extract_features.
                tensors = [
                    torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                    for p in batch
                ]
                batch_input = torch.stack(tensors).to(device)
                output = model(batch_input)
                batch_feats = output.squeeze(-1).squeeze(-1).cpu().numpy()
            features.append(batch_feats)
    return np.vstack(features)
