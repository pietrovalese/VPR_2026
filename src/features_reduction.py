import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats as scipy_stats
from sklearn.feature_selection import mutual_info_regression
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
DEPS_DIR     = ROOT / "deps"
VPR_EVAL_DIR = DEPS_DIR / "VPR-methods-evaluation"
DATA_DIR     = ROOT / "data"
DESC_DIR     = ROOT / "logs" / "descriptors"
OUT_DIR      = ROOT / "logs" / "feature_reduction"

if VPR_EVAL_DIR.exists() and str(VPR_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(VPR_EVAL_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)

GPS_THRESHOLD_M = 25.0
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

TEST_DATASETS = {
    "sf_xs_test": {
        "database": DATA_DIR / "sf_xs" / "test" / "database",
        "queries":  DATA_DIR / "sf_xs" / "test" / "queries",
    },
    "tokyo_xs": {
        "database": DATA_DIR / "tokyo_xs" / "test" / "database",
        "queries":  DATA_DIR / "tokyo_xs" / "test" / "queries",
    },
    "svox_sun": {
        "database": DATA_DIR / "svox" / "images" / "test" / "gallery",
        "queries":  DATA_DIR / "svox" / "images" / "test" / "queries_sun",
    },
    "svox_night": {
        "database": DATA_DIR / "svox" / "images" / "test" / "gallery",
        "queries":  DATA_DIR / "svox" / "images" / "test" / "queries_night",
    },
}

VAL_DATASET = {
    "database": DATA_DIR / "sf_xs" / "val" / "database",
    "queries":  DATA_DIR / "sf_xs" / "val" / "queries",
}

MODEL_LOADERS = {
    "cosplace": lambda: _get_model(method="cosplace", backbone="ResNet18",
                                   descriptors_dimension=512).eval(),
    "megaloc":  lambda: _get_model(method="megaloc").eval(),
}
MODEL_IMAGE_SIZES = {
    "cosplace": (512, 512),
    "megaloc":  (322, 322),
}


def _get_model(*args, **kwargs):
    """Load a VPR model through the vpr_models submodule."""
    from vpr_models import get_model
    return get_model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    """Recursively loads images from a folder and applies resize/normalize for the given model."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, folder: Path, image_size: tuple):
        self.paths = sorted(
            p for p in folder.rglob("*") if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {folder}")
        H, W = image_size
        self.transform = transforms.Compose([
            transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


# ---------------------------------------------------------------------------
# Descriptor extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_raw_descriptors(
    model, folder: Path, image_size: tuple,
    batch_size: int, num_workers: int, device: torch.device,
) -> tuple[np.ndarray, list]:
    """Run the model over all images in `folder` and return raw descriptors + paths."""
    dataset = ImageFolderDataset(folder, image_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=(device.type == "cuda"))
    all_desc = []
    model = model.to(device)
    for images, _ in tqdm(loader, desc=f"  {folder.name}", leave=False):
        desc = model(images.to(device))
        all_desc.append(desc.cpu().numpy())
    return np.concatenate(all_desc, axis=0), dataset.paths


@torch.no_grad()
def extract_normalized_descriptors(
    model, folder: Path, image_size: tuple,
    batch_size: int, num_workers: int, device: torch.device,
) -> tuple[np.ndarray, list]:
    """Same as extract_raw_descriptors but L2-normalizes each descriptor."""
    raw, paths = extract_raw_descriptors(
        model, folder, image_size, batch_size, num_workers, device)
    norm = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    return norm, paths


# ---------------------------------------------------------------------------
# Activation analysis (chunked)
# ---------------------------------------------------------------------------
def analyze_activations(
    descriptors_path: Path, N: int, D: int,
    near_zero_eps: float = 1e-3,
    chunk_size: int = 4096,
) -> dict:
    """Compute per-feature activation rate, variance, mean abs value and entropy over a memmapped descriptor array, in chunks to keep memory bounded."""
    mm         = np.lib.format.open_memmap(str(descriptors_path), mode="r")
    act_count  = np.zeros(D, dtype=np.float64)
    sum_x      = np.zeros(D, dtype=np.float64)
    sum_x2     = np.zeros(D, dtype=np.float64)
    sum_abs    = np.zeros(D, dtype=np.float64)
    n_bins     = 50
    sample     = mm[:min(10000, N)].astype(np.float32)
    feat_min   = sample.min(axis=0)
    feat_max   = sample.max(axis=0)
    feat_max   = np.where(feat_max == feat_min, feat_min + 1e-8, feat_max)
    hists      = np.zeros((D, n_bins), dtype=np.float64)

    log.info(f"  Chunked activation analysis ({N} img, {D} dim) ...")
    for start in tqdm(range(0, N, chunk_size), desc="  activation", leave=False):
        chunk = mm[start:start+chunk_size].astype(np.float32)
        act_count += (np.abs(chunk) > near_zero_eps).sum(axis=0)
        sum_x     += chunk.sum(axis=0)
        sum_x2    += (chunk ** 2).sum(axis=0)
        sum_abs   += np.abs(chunk).sum(axis=0)
        for d in range(D):
            h, _ = np.histogram(chunk[:, d], bins=n_bins,
                                range=(float(feat_min[d]), float(feat_max[d])))
            hists[d] += h
    del mm

    mean_x          = sum_x / N
    variance        = np.maximum(sum_x2 / N - mean_x ** 2, 0.0)
    activation_rate = act_count / N
    mean_abs        = sum_abs / N
    entropy_norm    = np.zeros(D, dtype=np.float32)
    for d in range(D):
        p = hists[d] / (hists[d].sum() + 1e-12)
        p = p[p > 0]
        h = -np.sum(p * np.log(p))
        entropy_norm[d] = h / np.log(n_bins)

    return {
        "activation_rate": activation_rate.astype(np.float32),
        "variance":        variance.astype(np.float32),
        "mean_abs":        mean_abs.astype(np.float32),
        "entropy_norm":    entropy_norm,
    }


def build_activation_mask(stats: dict, act_threshold: float,
                           var_threshold: float) -> np.ndarray:
    """Keep a feature only if its activation rate and variance both exceed the given thresholds."""
    mask = ((stats["activation_rate"] > act_threshold) &
            (stats["variance"] > var_threshold))
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Multi-layer redundancy system: Pearson -> Spearman -> MI
# ---------------------------------------------------------------------------
def _load_sample(descriptors_path: Path, N: int, max_samples: int,
                 seed: int = 42) -> np.ndarray:
    """Load a random sample from the memmap without loading the whole array into RAM."""
    mm = np.lib.format.open_memmap(str(descriptors_path), mode="r")
    if N > max_samples:
        idx = np.random.default_rng(seed).choice(N, max_samples, replace=False)
        idx.sort()
        desc = mm[idx].astype(np.float32)
    else:
        desc = mm[:].astype(np.float32)
    del mm
    return desc


def compute_pearson_matrix(desc: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix (D x D) from an (N, D) array."""
    mean = desc.mean(axis=0)
    std  = desc.std(axis=0)
    std[std < 1e-8] = 1.0
    desc_z = (desc - mean) / std
    corr   = (desc_z.T @ desc_z) / desc.shape[0]
    np.fill_diagonal(corr, 0.0)
    return corr


def compute_spearman_matrix(desc: np.ndarray) -> np.ndarray:
    """Spearman = Pearson on ranks. More robust to non-linear monotone relationships."""
    log.info("    Computing ranks for Spearman ...")
    ranks = np.zeros_like(desc)
    for d in tqdm(range(desc.shape[1]), desc="    ranking", leave=False):
        ranks[:, d] = scipy_stats.rankdata(desc[:, d])
    return compute_pearson_matrix(ranks)


def compute_mi_for_pairs(
    desc: np.ndarray,
    candidate_pairs: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Compute Mutual Information for a specific set of (i, j) pairs, using sklearn's
    mutual_info_regression (Kraskov k-NN estimator).

    candidate_pairs : (M, 2) array of feature index pairs
    Returns : (M,) array of MI values normalized to [0, 1]
    """
    M   = len(candidate_pairs)
    mis = np.zeros(M, dtype=np.float32)

    # Group by target feature j: mutual_info_regression(X, y) computes MI(X_col, y)
    # for every column of X in a single call, which is far more efficient than
    # evaluating pair by pair.
    from collections import defaultdict
    j_to_indices = defaultdict(list)
    for idx, (i, j) in enumerate(candidate_pairs):
        j_to_indices[int(j)].append((idx, int(i)))

    log.info(f"    MI over {M} pairs ({len(j_to_indices)} unique targets) ...")
    for j, pairs_for_j in tqdm(j_to_indices.items(), desc="    MI", leave=False):
        y      = desc[:, j]
        i_idxs = [p[1] for p in pairs_for_j]
        X      = desc[:, i_idxs]   # (N, len(i_idxs))
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mi_vals = mutual_info_regression(X, y, n_neighbors=n_neighbors,
                                         random_state=42)
        for k, (pair_idx, _) in enumerate(pairs_for_j):
            mis[pair_idx] = float(mi_vals[k])

    # Normalize MI to [0, 1] relative to the observed maximum
    max_mi = mis.max()
    if max_mi > 0:
        mis = mis / max_mi
    return mis


def build_multilayer_redundancy(
    descriptors_path: Path,
    N: int,
    D: int,
    variances: np.ndarray,
    w1: float = 0.33,
    w2: float = 0.33,
    w3: float = 0.34,
    pearson_prefilter: float = 0.3,
    spearman_prefilter: float = 0.3,
    score_threshold: float = 0.6,
    max_samples_pearson: int = 50_000,
    max_samples_mi: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Multi-layer system to identify redundant features.

    Layer 1 - Pearson  : candidate pairs with |r| > pearson_prefilter
    Layer 2 - Spearman : further filters with |rho| > spearman_prefilter
    Layer 3 - MI       : computes MI only on the surviving pairs

    Final score: S(i,j) = w1*|P| + w2*|S| + w3*MI_norm

    Returns:
        keep_mask     : (D,) bool - True = feature to keep
        redund_scores : (D,) float - maximum redundancy score per feature
        layer_stats   : dict with per-layer statistics
    """
    log.info(f"  [Multi-layer] D={D}, N={N}, final threshold={score_threshold}")

    # ---- Layer 1: Pearson ------------------------------------------------
    log.info(f"  Layer 1 - Pearson (sample={min(N, max_samples_pearson)})")
    desc_p  = _load_sample(descriptors_path, N, max_samples_pearson)
    pearson = compute_pearson_matrix(desc_p)

    # Find all pairs (i<j) with |Pearson| > prefilter
    upper        = np.triu(np.abs(pearson), k=1)
    l1_pairs     = np.argwhere(upper > pearson_prefilter)   # (M1, 2)
    l1_n_pairs   = len(l1_pairs)
    log.info(f"    Pairs after Layer 1: {l1_n_pairs} / {D*(D-1)//2}")

    if l1_n_pairs == 0:
        log.info("  No pair passes the Pearson prefilter - no redundant features")
        keep_mask     = np.ones(D, dtype=bool)
        redund_scores = np.zeros(D, dtype=np.float32)
        return keep_mask, redund_scores, {
            "l1_pairs": 0, "l2_pairs": 0, "l3_pairs": 0, "removed": 0
        }

    # ---- Layer 2: Spearman -----------------------------------------------
    log.info(f"  Layer 2 - Spearman (sample={min(N, max_samples_pearson)})")
    spearman     = compute_spearman_matrix(desc_p)
    del desc_p   # free RAM

    # Further filter: keep only pairs where |Spearman| also > prefilter
    l1_p_vals    = np.abs(pearson[l1_pairs[:, 0], l1_pairs[:, 1]])
    l1_s_vals    = np.abs(spearman[l1_pairs[:, 0], l1_pairs[:, 1]])
    l2_mask      = l1_s_vals > spearman_prefilter
    l2_pairs     = l1_pairs[l2_mask]
    l2_p_vals    = l1_p_vals[l2_mask]
    l2_s_vals    = l1_s_vals[l2_mask]
    l2_n_pairs   = len(l2_pairs)
    log.info(f"    Pairs after Layer 2: {l2_n_pairs}")

    # ---- Layer 3: MI -----------------------------------------------------
    mi_vals      = np.zeros(l2_n_pairs, dtype=np.float32)
    if l2_n_pairs > 0:
        log.info(f"  Layer 3 - MI (sample={min(N, max_samples_mi)}, pairs={l2_n_pairs})")
        desc_mi  = _load_sample(descriptors_path, N, max_samples_mi)
        mi_vals  = compute_mi_for_pairs(desc_mi, l2_pairs)
        del desc_mi
    else:
        log.info("  Layer 3 - no pair left to evaluate with MI")

    # ---- Final score -------------------------------------------------------
    # S(i,j) = w1*|P| + w2*|S| + w3*MI
    final_scores = w1 * l2_p_vals + w2 * l2_s_vals + w3 * mi_vals   # (l2_n_pairs,)

    # Maximum score across all pairs involving each feature
    redund_scores = np.zeros(D, dtype=np.float32)
    for k, (i, j) in enumerate(l2_pairs):
        redund_scores[i] = max(redund_scores[i], final_scores[k])
        redund_scores[j] = max(redund_scores[j], final_scores[k])

    # Greedy selection: sort pairs by decreasing score, for each pair drop the
    # feature with the lower variance
    order   = np.argsort(final_scores)[::-1]
    keep    = np.ones(D, dtype=bool)
    removed = 0

    for k in order:
        i, j = l2_pairs[k]
        if final_scores[k] < score_threshold:
            break   # pairs are sorted, none of the remaining ones pass the threshold
        if not keep[i] or not keep[j]:
            continue
        if variances[i] >= variances[j]:
            keep[j] = False
        else:
            keep[i] = False
        removed += 1

    layer_stats = {
        "l1_pairs":   int(l1_n_pairs),
        "l2_pairs":   int(l2_n_pairs),
        "l3_pairs":   int(l2_n_pairs),   # all l2 pairs receive an MI score
        "removed":    removed,
        "score_threshold": score_threshold,
        "weights":    {"w1": w1, "w2": w2, "w3": w3},
    }
    log.info(f"  Features removed (score > {score_threshold}): {removed}/{D}")
    return keep, redund_scores, layer_stats


# ---------------------------------------------------------------------------
# Layer-by-layer comparison (for the report)
# ---------------------------------------------------------------------------
def compare_layers(
    pearson_mat: np.ndarray | None,
    spearman_mat: np.ndarray | None,
    redund_scores: np.ndarray,
    variances: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """
    For each threshold, count how many features would be removed using:
        - Pearson alone
        - Spearman alone
        - the combined multi-layer score
    Useful to show the added value of the multi-layer system in the report.
    """
    rows = []
    D    = len(variances)

    for t in thresholds:
        row = {"threshold": t}

        if pearson_mat is not None:
            keep_p = np.ones(D, dtype=bool)
            pairs  = np.argwhere(np.triu(np.abs(pearson_mat), k=1) > t)
            for i, j in pairs:
                if not keep_p[i] or not keep_p[j]:
                    continue
                if variances[i] >= variances[j]:
                    keep_p[j] = False
                else:
                    keep_p[i] = False
            row["pearson_removed"] = int((~keep_p).sum())
        else:
            row["pearson_removed"] = None

        if spearman_mat is not None:
            keep_s = np.ones(D, dtype=bool)
            pairs  = np.argwhere(np.triu(np.abs(spearman_mat), k=1) > t)
            for i, j in pairs:
                if not keep_s[i] or not keep_s[j]:
                    continue
                if variances[i] >= variances[j]:
                    keep_s[j] = False
                else:
                    keep_s[i] = False
            row["spearman_removed"] = int((~keep_s).sum())
        else:
            row["spearman_removed"] = None

        # Combined multi-layer score
        keep_ml = np.ones(D, dtype=bool)
        feat_order = np.argsort(redund_scores)[::-1]
        for fi in feat_order:
            if redund_scores[fi] < t:
                break
            keep_ml[fi] = False
        row["multilayer_removed"] = int((~keep_ml).sum())

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# KNN + Recall
# ---------------------------------------------------------------------------
def parse_utm(path: str) -> tuple[float, float] | None:
    """Extract the first two '@'-separated numeric fields from a filename (UTM easting/northing)."""
    stem  = Path(path).stem
    parts = stem.split("@")
    nums  = []
    for p in parts:
        try:
            nums.append(float(p.strip()))
        except ValueError:
            pass
        if len(nums) == 2:
            break
    return (nums[0], nums[1]) if len(nums) >= 2 else None


def get_coords(paths) -> np.ndarray:
    """Parse UTM coordinates for a list of paths; unparseable ones become (nan, nan)."""
    coords = []
    for p in paths:
        r = parse_utm(str(p))
        coords.append(r if r else (float("nan"), float("nan")))
    return np.array(coords, dtype=np.float64)


def recall_at_n(
    db_desc: np.ndarray, q_desc: np.ndarray,
    db_coords: np.ndarray, q_coords: np.ndarray,
    k: int = 20, recall_values: list = None,
    threshold_m: float = GPS_THRESHOLD_M,
) -> dict[int, float]:
    """Run a FAISS inner-product KNN search and compute Recall@N for each N in recall_values."""
    import faiss
    if recall_values is None:
        recall_values = [1, 5, 10]
    index = faiss.IndexFlatIP(db_desc.shape[1])
    index.add(db_desc.astype(np.float32))
    _, preds = index.search(q_desc.astype(np.float32), k)
    N_q    = len(q_coords)
    recall = {n: 0 for n in recall_values}
    for i in range(N_q):
        qc = q_coords[i]
        if np.isnan(qc).any():
            continue
        for n in recall_values:
            dists = np.linalg.norm(db_coords[preds[i, :n]] - qc, axis=1)
            if np.any(dists <= threshold_m):
                recall[n] += 1
    return {n: 100.0 * recall[n] / N_q for n in recall_values}


def apply_mask_and_eval(
    db_norm: np.ndarray, q_norm: np.ndarray,
    db_coords: np.ndarray, q_coords: np.ndarray,
    mask: np.ndarray, recall_values: list,
) -> dict[int, float]:
    """Restrict descriptors to the given feature mask, re-normalize, and evaluate Recall@N."""
    db_c = db_norm[:, mask]
    q_c  = q_norm[:, mask]
    db_c = db_c / (np.linalg.norm(db_c, axis=1, keepdims=True) + 1e-8)
    q_c  = q_c  / (np.linalg.norm(q_c,  axis=1, keepdims=True) + 1e-8)
    return recall_at_n(db_c, q_c, db_coords, q_coords, recall_values=recall_values)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def threshold_sweep_multilayer(
    redund_scores: np.ndarray,
    act_stats: dict,
    variances: np.ndarray,
    val_db_norm: np.ndarray, val_q_norm: np.ndarray,
    val_db_coords: np.ndarray, val_q_coords: np.ndarray,
    score_thresholds: list[float],
    act_thresholds: list[float],
    var_thresholds: list[float],
    recall_values: list[int],
) -> list[dict]:
    """Grid-search redundancy/activation/variance thresholds on the validation set."""
    rows = []
    D = len(variances)

    for score_t in score_thresholds:
        redund_mask = redund_scores < score_t   # True = not redundant

        for act_t in act_thresholds:
            for var_t in var_thresholds:
                act_mask   = build_activation_mask(act_stats, act_t, var_t)
                final_mask = act_mask & redund_mask
                n_kept     = int(final_mask.sum())
                if n_kept < 10:
                    continue

                rec = apply_mask_and_eval(
                    val_db_norm, val_q_norm, val_db_coords, val_q_coords,
                    final_mask, recall_values,
                )
                rows.append({
                    "score_threshold": score_t,
                    "act_threshold":   act_t,
                    "var_threshold":   var_t,
                    "n_kept":          n_kept,
                    "n_total":         D,
                    "compression_pct": round(100 * (1 - n_kept / D), 2),
                    **{f"R@{n}": round(v, 4) for n, v in rec.items()},
                })
                log.info(
                    f"  score={score_t:.2f} act={act_t:.2f} var={var_t:.1e} -> "
                    f"{n_kept}/{D} feat ({100*(1-n_kept/D):.1f}% removed)  "
                    f"R@1={rec[recall_values[0]]:.2f}%"
                )
    return rows


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_gradcam(
    model: torch.nn.Module,
    image_path: str,
    image_size: tuple,
    feature_indices: list[int],
    device: torch.device,
) -> dict[int, np.ndarray]:
    """Compute Grad-CAM heatmaps for the given descriptor dimensions using the last Conv2d layer."""
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        log.warning("  No Conv2d layer found - Grad-CAM unavailable")
        return {}

    activations = {}
    gradients   = {}

    def fwd_hook(module, input, output):
        activations["feat"] = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    H, W = image_size
    transform = transforms.Compose([
        transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)

    heatmaps = {}
    for feat_idx in feature_indices:
        model.zero_grad()
        desc  = model(x)
        score = desc[0, feat_idx]
        score.backward(retain_graph=True)

        if "feat" not in activations or "feat" not in gradients:
            continue

        weights = gradients["feat"][0].mean(dim=(-2, -1))
        cam     = (weights[:, None, None] * activations["feat"][0]).sum(dim=0)
        cam     = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        heatmaps[feat_idx] = cam

    h_fwd.remove()
    h_bwd.remove()
    return heatmaps


# ---------------------------------------------------------------------------
# Size/recall curve by variance (top-K% features)
# ---------------------------------------------------------------------------
def variance_topk_curve(
    variances: np.ndarray,
    val_db_norm: np.ndarray, val_q_norm: np.ndarray,
    val_db_coords: np.ndarray, val_q_coords: np.ndarray,
    test_datasets_descriptors: dict,
    recall_values: list[int],
    topk_fractions: list[float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Evaluate recall keeping only the top-K% features by decreasing variance.

    Returns:
        val_rows  : results on the validation set for each K (used to pick the optimal K)
        test_rows : results on all test sets for the optimal K plus a few fixed reference K's
    """
    if topk_fractions is None:
        # From 100% (baseline) down to 5%, with finer steps near the drop-off region
        topk_fractions = [1.0, 0.90, 0.80, 0.70, 0.60, 0.50,
                          0.40, 0.30, 0.20, 0.15, 0.10, 0.05]

    D            = len(variances)
    sorted_feats = np.argsort(variances)[::-1]   # indices by decreasing variance
    val_rows     = []

    log.info(f"  Size/recall curve (top-K variance, {len(topk_fractions)} points)...")
    for frac in topk_fractions:
        n_keep = max(1, int(D * frac))
        mask   = np.zeros(D, dtype=bool)
        mask[sorted_feats[:n_keep]] = True

        rec = apply_mask_and_eval(
            val_db_norm, val_q_norm, val_db_coords, val_q_coords,
            mask, recall_values,
        )
        row = {
            "topk_fraction": frac,
            "n_kept":        n_keep,
            "n_total":       D,
            "compression_pct": round(100 * (1 - n_keep / D), 2),
            **{f"R@{n}": round(v, 4) for n, v in rec.items()},
        }
        val_rows.append(row)
        log.info(
            f"    top-{frac*100:.0f}% ({n_keep}/{D} feat, "
            f"{100*(1-n_keep/D):.0f}% removed)  "
            f"R@1={rec[recall_values[0]]:.2f}%"
        )

    # Pick the optimal K: maximum compression with R@1 within 2% of the baseline
    baseline_r1 = val_rows[0][f"R@{recall_values[0]}"]   # frac=1.0
    valid = [r for r in val_rows if r[f"R@{recall_values[0]}"] >= baseline_r1 - 2.0]
    best_frac = max(valid, key=lambda r: r["compression_pct"])["topk_fraction"]                 if valid else 1.0
    log.info(f"  Optimal K (within 2% of baseline): top-{best_frac*100:.0f}%")

    # Evaluate on the test sets for the optimal K plus a few fixed reference K's
    test_rows = []
    eval_fracs = sorted(set([best_frac, 1.0, 0.5, 0.25, 0.10]))
    for frac in eval_fracs:
        n_keep = max(1, int(D * frac))
        mask   = np.zeros(D, dtype=bool)
        mask[sorted_feats[:n_keep]] = True

        for ds_name, (db_norm, q_norm, db_coords, q_coords) in test_datasets_descriptors.items():
            rec = apply_mask_and_eval(db_norm, q_norm, db_coords, q_coords, mask, recall_values)
            test_rows.append({
                "dataset":         ds_name,
                "topk_fraction":   frac,
                "n_kept":          n_keep,
                "compression_pct": round(100 * (1 - n_keep / D), 2),
                "is_optimal":      frac == best_frac,
                **{f"R@{n}": round(v, 4) for n, v in rec.items()},
            })

    return val_rows, test_rows, best_frac


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Extension 6.3 - multi-layer feature reduction")
    p.add_argument("--methods",           nargs="+", default=["cosplace", "megaloc"])
    p.add_argument("--w1",                type=float, default=0.33,
                   help="Pearson weight in the final score (default: 0.33)")
    p.add_argument("--w2",                type=float, default=0.33,
                   help="Spearman weight in the final score (default: 0.33)")
    p.add_argument("--w3",                type=float, default=0.34,
                   help="MI weight in the final score (default: 0.34)")
    p.add_argument("--pearson_prefilter", type=float, default=0.3,
                   help="Layer 1 Pearson prefilter threshold (default: 0.3)")
    p.add_argument("--spearman_prefilter",type=float, default=0.3,
                   help="Layer 2 Spearman prefilter threshold (default: 0.3)")
    p.add_argument("--score_threshold",   type=float, default=None,
                   help="Fixed final score threshold. If None, a sweep is run.")
    p.add_argument("--act_threshold",     type=float, default=None)
    p.add_argument("--var_threshold",     type=float, default=None)
    p.add_argument("--recall_values",     nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--device",            type=str,   default="auto")
    p.add_argument("--n_gradcam_imgs",    type=int,   default=5)
    p.add_argument("--overwrite",         action="store_true")
    p.add_argument("--skip_multilayer",   action="store_true",
                   help="Skip the multi-layer system and go straight to the top-K curve")
    p.add_argument("--topk_fractions",    nargs="+", type=float, default=None,
                   help="K fractions for the size/recall curve (e.g. 0.5 0.25 0.1)")
    return p.parse_args()


def resolve_device(s):
    """Resolve the --device argument to a torch.device, auto-detecting CUDA when requested."""
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = resolve_device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rv = args.recall_values

    log.info(f"Device    : {device}")
    log.info(f"Methods   : {args.methods}")
    log.info(f"Weights   : w1(Pearson)={args.w1}  w2(Spearman)={args.w2}  w3(MI)={args.w3}")
    log.info(f"Prefilter : Pearson>{args.pearson_prefilter}  Spearman>{args.spearman_prefilter}")

    summary_rows = []

    for method_name in args.methods:
        log.info(f"\n{'='*65}")
        log.info(f"Method: {method_name.upper()}")

        method_dir = OUT_DIR / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        image_size = MODEL_IMAGE_SIZES[method_name]

        # ----------------------------------------------------------------
        # Load model
        # ----------------------------------------------------------------
        log.info("  Loading model...")
        try:
            model = MODEL_LOADERS[method_name]()
            model = model.to(device)
        except Exception as e:
            log.warning(f"  SKIP - {e}")
            continue

        # ----------------------------------------------------------------
        # GSV-XS: shape only, no RAM
        # ----------------------------------------------------------------
        gsv_dir      = DESC_DIR / "gsv_xs_train" / method_name
        gsv_raw_path = gsv_dir / "database_descriptors_raw.npy"

        if not gsv_raw_path.exists():
            log.error(f"  GSV-XS not extracted. Run: "
                      f"python src/extract_descriptors.py --datasets gsv_xs_train --methods {method_name}")
            continue

        mm_tmp   = np.lib.format.open_memmap(str(gsv_raw_path), mode="r")
        N_gsv, D = mm_tmp.shape
        del mm_tmp
        log.info(f"  GSV-XS: {N_gsv} images, dim={D}")

        # ----------------------------------------------------------------
        # Step 1 - Activation analysis
        # ----------------------------------------------------------------
        act_path = method_dir / "activation_stats.npy"
        var_path = method_dir / "variance_stats.npy"
        ent_path = method_dir / "entropy_stats.npy"

        if act_path.exists() and not args.overwrite:
            log.info("  Activation stats already computed, loading...")
            stats = {
                "activation_rate": np.load(act_path),
                "variance":        np.load(var_path),
                "entropy_norm":    np.load(ent_path),
                "mean_abs":        np.zeros(D, dtype=np.float32),
            }
        else:
            stats = analyze_activations(gsv_raw_path, N_gsv, D)
            np.save(act_path, stats["activation_rate"])
            np.save(var_path, stats["variance"])
            np.save(ent_path, stats["entropy_norm"])

        dead = int((stats["activation_rate"] < 0.01).sum())
        lvar = int((stats["variance"] < 1e-4).sum())
        log.info(f"  Nearly-never-activated features (<1%): {dead}/{D}")
        log.info(f"  Low-variance features (<1e-4): {lvar}/{D}")

        # ----------------------------------------------------------------
        # Step 2 - Multi-layer Pearson + Spearman + MI system
        # ----------------------------------------------------------------
        redund_path = method_dir / "redundancy_scores.npy"
        mask_redund = method_dir / "mask_redundancy.npy"

        if redund_path.exists() and not args.overwrite:
            log.info("  Redundancy scores already computed, loading...")
            redund_scores = np.load(redund_path)
            keep_redund   = np.load(mask_redund) if mask_redund.exists() else (redund_scores < 0.6)
            layer_stats   = {}
        else:
            keep_redund, redund_scores, layer_stats = build_multilayer_redundancy(
                gsv_raw_path, N_gsv, D,
                variances=stats["variance"],
                w1=args.w1, w2=args.w2, w3=args.w3,
                pearson_prefilter=args.pearson_prefilter,
                spearman_prefilter=args.spearman_prefilter,
                score_threshold=args.score_threshold or 0.6,
            )
            np.save(redund_path, redund_scores)
            np.save(mask_redund, keep_redund)

            # Save the Pearson/Spearman matrices when D is small enough
            if D <= 5000:
                desc_p   = _load_sample(gsv_raw_path, N_gsv, 50_000)
                p_mat    = compute_pearson_matrix(desc_p)
                s_mat    = compute_spearman_matrix(desc_p)
                np.save(method_dir / "pearson_matrix.npy",  p_mat.astype(np.float32))
                np.save(method_dir / "spearman_matrix.npy", s_mat.astype(np.float32))
                comp_rows = compare_layers(
                    p_mat, s_mat, redund_scores, stats["variance"],
                    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                )
                with open(method_dir / "layer_comparison.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(comp_rows)
                del desc_p, p_mat, s_mat

        n_redund_removed = int((~keep_redund).sum())
        log.info(f"  Redundant features removed: {n_redund_removed}/{D}")
        log.info(f"  Redundancy score - max={redund_scores.max():.3f}  "
                 f"mean={redund_scores.mean():.3f}  "
                 f">0.5: {(redund_scores>0.5).sum()}")

        # ----------------------------------------------------------------
        # Step 3 - Validation set + threshold sweep
        # ----------------------------------------------------------------
        log.info("  Loading validation set (sf_xs_val)...")
        val_db_norm, val_db_paths = extract_normalized_descriptors(
            model, VAL_DATASET["database"], image_size,
            args.batch_size, args.num_workers, device)
        val_q_norm, val_q_paths = extract_normalized_descriptors(
            model, VAL_DATASET["queries"], image_size,
            args.batch_size, args.num_workers, device)
        val_db_coords = get_coords(val_db_paths)
        val_q_coords  = get_coords(val_q_paths)

        baseline_recall = recall_at_n(
            val_db_norm, val_q_norm, val_db_coords, val_q_coords,
            recall_values=rv)
        baseline_r1 = baseline_recall[rv[0]]
        log.info(f"  Baseline val R@1={baseline_r1:.2f}%")

        if args.score_threshold and args.act_threshold and args.var_threshold:
            score_thresholds = [args.score_threshold]
            act_thresholds   = [args.act_threshold]
            var_thresholds   = [args.var_threshold]
        else:
            score_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
            act_thresholds   = [0.01, 0.05, 0.10]
            var_thresholds   = [1e-5, 1e-4, 1e-3]

        log.info("  Threshold sweep on validation set...")
        sweep_rows = threshold_sweep_multilayer(
            redund_scores, stats, stats["variance"],
            val_db_norm, val_q_norm, val_db_coords, val_q_coords,
            score_thresholds, act_thresholds, var_thresholds, rv,
        )

        if sweep_rows:
            with open(method_dir / "threshold_sweep.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sweep_rows)

        # Optimal threshold: maximum compression with R@1 within 2% of the baseline
        r1_key = f"R@{rv[0]}"
        valid  = [r for r in sweep_rows if r.get(r1_key, 0) >= baseline_r1 - 2.0]
        if valid:
            best = max(valid, key=lambda r: r["compression_pct"])
        elif sweep_rows:
            best = max(sweep_rows, key=lambda r: r["compression_pct"])
            log.warning("  No configuration within 2% of the baseline")
        else:
            best = {"score_threshold": 0.6, "act_threshold": 0.01, "var_threshold": 1e-5}

        best_score_t = best["score_threshold"]
        best_act_t   = best["act_threshold"]
        best_var_t   = best["var_threshold"]
        log.info(f"  Chosen thresholds: score={best_score_t} act={best_act_t} var={best_var_t}")

        # ----------------------------------------------------------------
        # Final mask
        # ----------------------------------------------------------------
        act_mask    = build_activation_mask(stats, best_act_t, best_var_t)
        redund_mask = redund_scores < best_score_t
        final_mask  = act_mask & redund_mask

        n_kept   = int(final_mask.sum())
        comp_pct = 100 * (1 - n_kept / D)

        np.save(method_dir / "mask_activation.npy", act_mask)
        np.save(method_dir / "mask_final.npy",      final_mask)
        log.info(f"  Final mask: {n_kept}/{D} features kept ({comp_pct:.1f}% removed)")

        # ----------------------------------------------------------------
        # Step 4 - Test set evaluation
        # ----------------------------------------------------------------
        log.info("  Evaluating on test sets...")
        test_results = {}
        test_desc_cache = {}   # reused in Step 5 to avoid re-extracting descriptors

        for ds_name, ds_cfg in TEST_DATASETS.items():
            if not ds_cfg["database"].exists():
                continue
            db_norm, db_paths = extract_normalized_descriptors(
                model, ds_cfg["database"], image_size,
                args.batch_size, args.num_workers, device)
            q_norm, q_paths = extract_normalized_descriptors(
                model, ds_cfg["queries"], image_size,
                args.batch_size, args.num_workers, device)
            db_coords = get_coords(db_paths)
            q_coords  = get_coords(q_paths)
            test_desc_cache[ds_name] = (db_norm, q_norm, db_coords, q_coords)

            rec_full = recall_at_n(db_norm, q_norm, db_coords, q_coords, recall_values=rv)
            rec_comp = apply_mask_and_eval(db_norm, q_norm, db_coords, q_coords, final_mask, rv)

            test_results[ds_name] = {
                "recall_full":       {f"R@{n}": round(v, 4) for n, v in rec_full.items()},
                "recall_compressed": {f"R@{n}": round(v, 4) for n, v in rec_comp.items()},
                "recall_delta":      {f"R@{n}": round(rec_comp[n] - rec_full[n], 4) for n in rv},
            }
            log.info(
                f"  [{ds_name}] Full R@1={rec_full[rv[0]]:.2f}%  "
                f"Comp R@1={rec_comp[rv[0]]:.2f}%  "
                f"delta={rec_comp[rv[0]]-rec_full[rv[0]]:+.2f}%"
            )

        # ----------------------------------------------------------------
        # Step 5 - Size/recall curve by variance (top-K%)
        # ----------------------------------------------------------------
        log.info("  Size/recall curve (top-K% by variance)...")
        # Reuses the test-set descriptors already extracted in Step 4.

        topk_val_rows, topk_test_rows, best_topk_frac = variance_topk_curve(
            variances=stats["variance"],
            val_db_norm=val_db_norm, val_q_norm=val_q_norm,
            val_db_coords=val_db_coords, val_q_coords=val_q_coords,
            test_datasets_descriptors=test_desc_cache,
            recall_values=rv,
            topk_fractions=args.topk_fractions,
        )

        if topk_val_rows:
            with open(method_dir / "topk_curve_val.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(topk_val_rows[0].keys()))
                writer.writeheader()
                writer.writerows(topk_val_rows)

        if topk_test_rows:
            with open(method_dir / "topk_curve_test.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(topk_test_rows[0].keys()))
                writer.writeheader()
                writer.writerows(topk_test_rows)

        result_topk = {
            "best_topk_fraction":  best_topk_frac,
            "best_topk_n_kept":    max(1, int(D * best_topk_frac)),
            "best_topk_comp_pct":  round(100 * (1 - best_topk_frac), 2),
        }
        log.info(
            f"  Optimal top-K: {best_topk_frac*100:.0f}% -> "
            f"{max(1, int(D*best_topk_frac))}/{D} features "
            f"({100*(1-best_topk_frac):.0f}% removed)"
        )

        # Save the actual boolean top-K-by-variance mask. Unlike mask_final.npy
        # (activation AND multi-layer redundancy), this is the mask that really
        # reduces dimensionality when descriptor dimensions are weakly correlated
        # with one another (Pearson/Spearman/MI all low -> mask_redundancy almost
        # always True -> mask_final removes almost nothing). uncertainty_estimation.py
        # uses this mask for extension 6.2.
        topk_n_keep = max(1, int(D * best_topk_frac))
        topk_sorted_feats = np.argsort(stats["variance"])[::-1]
        mask_topk_variance = np.zeros(D, dtype=bool)
        mask_topk_variance[topk_sorted_feats[:topk_n_keep]] = True
        np.save(method_dir / "mask_topk_variance.npy", mask_topk_variance)
        log.info(f"  Saved mask_topk_variance.npy: {topk_n_keep}/{D} features kept")

        # ----------------------------------------------------------------
        # Step 6 - Grad-CAM
        # ----------------------------------------------------------------
        log.info(f"  Grad-CAM ({args.n_gradcam_imgs} images)...")
        # Uses mask_topk_variance (not final_mask): it's the mask that actually
        # reduces dimensionality (see note above). The most representative
        # "removed" features are the ones with lowest variance among the
        # discarded ones; the most representative "kept" features are the ones
        # with highest variance among the retained ones.
        removed_feats = np.where(~mask_topk_variance)[0]
        kept_feats    = np.where(mask_topk_variance)[0]
        removed_top   = removed_feats[np.argsort(stats["variance"][removed_feats])][:5] \
                        if len(removed_feats) > 0 else []
        kept_top      = kept_feats[np.argsort(stats["variance"][kept_feats])[::-1]][:5]

        gsv_paths   = list(np.load(gsv_dir / "database_paths.npy"))
        np.random.seed(42)
        sample_idxs = np.random.choice(len(gsv_paths),
                                        min(args.n_gradcam_imgs, len(gsv_paths)),
                                        replace=False)
        sample_paths = [str(gsv_paths[i]) for i in sample_idxs]

        for img_idx, img_path in enumerate(sample_paths):
            if len(removed_top) > 0:
                hm_rem = compute_gradcam(model, img_path, image_size,
                                         list(removed_top.astype(int)), device)
                if hm_rem:
                    np.save(method_dir / f"gradcam_removed_img{img_idx}.npy",
                            np.stack(list(hm_rem.values())))
            hm_kept = compute_gradcam(model, img_path, image_size,
                                       list(kept_top.astype(int)), device)
            if hm_kept:
                np.save(method_dir / f"gradcam_kept_img{img_idx}.npy",
                        np.stack(list(hm_kept.values())))

        with open(method_dir / "gradcam_image_paths.json", "w") as f:
            json.dump(sample_paths, f, indent=2)
        if len(removed_top) > 0:
            np.save(method_dir / "gradcam_removed_feat_indices.npy", removed_top)
        np.save(method_dir / "gradcam_kept_feat_indices.npy", kept_top)

        # ----------------------------------------------------------------
        # Save results
        # ----------------------------------------------------------------
        result = {
            "method":            method_name,
            "descriptor_dim":    D,
            "multilayer": {
                "n_kept":          n_kept,
                "n_removed":       D - n_kept,
                "compression_pct": round(comp_pct, 2),
                "score_threshold": best_score_t,
                "act_threshold":   best_act_t,
                "var_threshold":   best_var_t,
                "weights":         {"w1": args.w1, "w2": args.w2, "w3": args.w3},
                "layer_stats":     layer_stats,
            },
            "topk_variance":     result_topk,
            "topk_test_rows":    topk_test_rows,
            "n_dead_features":   dead,
            "baseline_val":      {f"R@{n}": round(v, 4) for n, v in baseline_recall.items()},
            "test_results":      test_results,
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(method_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)
        summary_rows.append(result)
        log.info(f"  Results saved in {method_dir.relative_to(ROOT)}/")

    if summary_rows:
        _save_summary(summary_rows, rv)


def _save_summary(rows, rv):
    """Flatten the per-method result dicts into logs/feature_reduction/summary.csv and print a readable report."""
    csv_path = OUT_DIR / "summary.csv"
    fieldnames = [
        "method", "descriptor_dim", "n_dead_features",
        "multilayer_n_kept", "multilayer_compression_pct",
        "topk_best_fraction", "topk_n_kept", "topk_compression_pct",
    ]
    for ds in TEST_DATASETS:
        for n in rv:
            fieldnames += [f"{ds}_R@{n}_full", f"{ds}_R@{n}_compressed", f"{ds}_R@{n}_delta"]
            fieldnames += [f"{ds}_R@{n}_full_topk", f"{ds}_R@{n}_compressed_topk", f"{ds}_R@{n}_delta_topk"]

    flat_rows = []
    for r in rows:
        ml  = r.get("multilayer", {})
        tkv = r.get("topk_variance", {})
        flat = {
            "method":                   r.get("method"),
            "descriptor_dim":           r.get("descriptor_dim"),
            "n_dead_features":          r.get("n_dead_features"),
            "multilayer_n_kept":        ml.get("n_kept"),
            "multilayer_compression_pct": ml.get("compression_pct"),
            "topk_best_fraction":       tkv.get("best_topk_fraction"),
            "topk_n_kept":              tkv.get("best_topk_n_kept"),
            "topk_compression_pct":     tkv.get("best_topk_comp_pct"),
        }
        for ds_name, ds_res in r.get("test_results", {}).items():
            for n in rv:
                key = f"R@{n}"
                flat[f"{ds_name}_R@{n}_full"]       = ds_res["recall_full"].get(key)
                flat[f"{ds_name}_R@{n}_compressed"] = ds_res["recall_compressed"].get(key)
                flat[f"{ds_name}_R@{n}_delta"]      = ds_res["recall_delta"].get(key)

        # "_topk" columns: same full/compressed/delta but with the variance mask
        # (the one that actually compresses), read from topk_test_rows - already
        # computed in Step 5, no re-evaluation needed.
        topk_rows = r.get("topk_test_rows", [])
        for ds_name in TEST_DATASETS:
            row_full = next((row for row in topk_rows
                              if row["dataset"] == ds_name and row["topk_fraction"] == 1.0), None)
            row_comp = next((row for row in topk_rows
                              if row["dataset"] == ds_name and row.get("is_optimal")), None)
            if row_full is None or row_comp is None:
                continue
            for n in rv:
                key = f"R@{n}"
                full_v = row_full.get(key)
                comp_v = row_comp.get(key)
                flat[f"{ds_name}_R@{n}_full_topk"]       = full_v
                flat[f"{ds_name}_R@{n}_compressed_topk"] = comp_v
                if full_v is not None and comp_v is not None:
                    flat[f"{ds_name}_R@{n}_delta_topk"] = round(comp_v - full_v, 4)
        flat_rows.append(flat)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    log.info(f"\nSummary -> {csv_path.relative_to(ROOT)}")

    print("\n" + "="*70)
    print("FEATURE REDUCTION - FINAL RESULTS")
    print("="*70)
    for r in rows:
        D   = r["descriptor_dim"]
        ml  = r.get("multilayer", {})
        tkv = r.get("topk_variance", {})

        print(f"\n{r['method'].upper()}  (D={D})")
        print(f"  [Multi-layer]    {ml.get('n_kept', D)}/{D} "
              f"({ml.get('compression_pct', 0):.1f}% removed)  "
              f"- pairs L1:{ml.get('layer_stats', {}).get('l1_pairs', '?')} "
              f"L2:{ml.get('layer_stats', {}).get('l2_pairs', '?')} "
              f"removed:{ml.get('layer_stats', {}).get('removed', 0)}")
        if tkv:
            print(f"  [Top-K variance] optimal top-{tkv.get('best_topk_fraction',1)*100:.0f}% "
                  f"-> {tkv.get('best_topk_n_kept', D)}/{D} "
                  f"({tkv.get('best_topk_comp_pct', 0):.1f}% removed)")
        for ds_name, ds_res in r.get("test_results", {}).items():
            for n in rv:
                key = f"R@{n}"
                b   = ds_res["recall_full"].get(key, 0)
                a   = ds_res["recall_compressed"].get(key, 0)
                print(f"  {ds_name:<18} R@{n}: {b:.2f}% -> {a:.2f}% ({a-b:+.2f}%)")


if __name__ == "__main__":
    main()