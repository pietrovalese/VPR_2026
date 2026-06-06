"""
src/feature_reduction.py  —  Estensione 6.3: How to save Memory?

Pipeline:
    1. Carica descrittori GSV-XS (training) per i 2 metodi scelti
    2. Analisi attivazioni: individua feature raramente attivate o quasi-uniformi
    3. Analisi correlazione: individua feature ridondanti (alta correlazione)
    4. Costruisce maschere binarie "feature utili"
    5. Valida le soglie su sf_xs_val (sweep threshold → Recall@1)
    6. Valuta sui test set con descrittori compressi
    7. Grad-CAM: salva heatmap .npy per feature eliminate vs mantenute
    8. Salva tutto in logs/feature_reduction/

Output:
    logs/feature_reduction/
    └── <method>/
        ├── activation_stats.npy       # (D,) frazione attivazioni non-zero
        ├── variance_stats.npy         # (D,) varianza per feature
        ├── correlation_matrix.npy     # (D, D) — solo se D <= 5000
        ├── mask_activation.npy        # (D,) bool — feature supera soglia attivazione
        ├── mask_correlation.npy       # (D,) bool — feature non ridondante
        ├── mask_final.npy             # (D,) bool — AND delle due maschere
        ├── threshold_sweep.csv        # Recall@1 vs soglia su val set
        ├── gradcam_kept_<img>.npy     # heatmap feature mantenute
        ├── gradcam_removed_<img>.npy  # heatmap feature eliminate
        └── results.json               # compression factor + recall

    logs/feature_reduction/summary.csv  — tabella riassuntiva

Uso:
    python src/feature_reduction.py
    python src/feature_reduction.py --methods cosplace megaloc
    python src/feature_reduction.py --act_threshold 0.05 --corr_threshold 0.95
    python src/feature_reduction.py --overwrite
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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

GPS_THRESHOLD_M  = 25.0
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]

# Dataset di test su cui valutare la recall finale
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
    "cosplace": lambda: _get_model(method="cosplace", backbone="ResNet18", descriptors_dimension=512).eval(),
    "megaloc":  lambda: _get_model(method="megaloc").eval(),
}

MODEL_IMAGE_SIZES = {
    "cosplace": (512, 512),
    "megaloc":  (322, 322),
}


def _get_model(*args, **kwargs):
    from vpr_models import get_model
    return get_model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Dataset generico
# ---------------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, folder: Path, image_size: tuple):
        self.paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in self.EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"Nessuna immagine in {folder}")
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
# Estrazione descrittori (grezzi, NON normalizzati — per analisi statistica)
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_raw_descriptors(
    model, folder: Path, image_size: tuple,
    batch_size: int, num_workers: int, device: torch.device,
) -> tuple[np.ndarray, list[Path]]:
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
) -> tuple[np.ndarray, list[Path]]:
    raw, paths = extract_raw_descriptors(model, folder, image_size, batch_size, num_workers, device)
    norm = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    return norm, paths


# ---------------------------------------------------------------------------
# Step 1 — Analisi attivazioni
# ---------------------------------------------------------------------------
def analyze_activations(descriptors: np.ndarray, near_zero_eps: float = 1e-3) -> dict:
    """
    Per ogni feature (colonna) calcola:
        - activation_rate   : frazione di campioni con |val| > eps  (0 = mai attivata)
        - variance          : varianza sul training set
        - mean_abs          : valore assoluto medio
        - entropy_norm      : entropia normalizzata della distribuzione (0 = uniforme, 1 = concentrata)
    """
    D = descriptors.shape[1]
    activation_rate = np.mean(np.abs(descriptors) > near_zero_eps, axis=0)   # (D,)
    variance        = np.var(descriptors, axis=0)                              # (D,)
    mean_abs        = np.mean(np.abs(descriptors), axis=0)                    # (D,)

    # Entropia normalizzata: discretizza in 50 bin e calcola H / log(n_bin)
    entropy_norm = np.zeros(D, dtype=np.float32)
    n_bins = 50
    for d in range(D):
        counts, _ = np.histogram(descriptors[:, d], bins=n_bins)
        p = counts / (counts.sum() + 1e-12)
        p = p[p > 0]
        h = -np.sum(p * np.log(p))
        entropy_norm[d] = h / np.log(n_bins)

    return {
        "activation_rate": activation_rate,
        "variance":        variance,
        "mean_abs":        mean_abs,
        "entropy_norm":    entropy_norm,
    }


def build_activation_mask(stats: dict, act_threshold: float, var_threshold: float) -> np.ndarray:
    """
    Feature UTILE se:
        activation_rate > act_threshold   AND
        variance        > var_threshold
    """
    mask = (stats["activation_rate"] > act_threshold) & (stats["variance"] > var_threshold)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# Step 2 — Analisi correlazione
# ---------------------------------------------------------------------------
def build_correlation_mask(descriptors: np.ndarray, corr_threshold: float) -> np.ndarray:
    """
    Calcola la matrice di correlazione di Pearson e rimuove le feature ridondanti.
    Per ogni coppia (i, j) con |corr| > threshold, rimuove quella con varianza minore.
    Ritorna maschera bool (D,): True = feature da mantenere.
    """
    D = descriptors.shape[1]
    log.info(f"  Calcolo matrice correlazione ({D}×{D}) ...")

    # Normalizza per colonna (zero mean, unit std) per efficienza
    std = descriptors.std(axis=0)
    std[std < 1e-8] = 1.0
    desc_norm = (descriptors - descriptors.mean(axis=0)) / std

    # Calcolo a batch per efficienza memoria
    # corr[i,j] = mean(desc_norm[:,i] * desc_norm[:,j])
    corr_matrix = (desc_norm.T @ desc_norm) / descriptors.shape[0]   # (D, D)
    np.fill_diagonal(corr_matrix, 0.0)  # ignora autocorrelazione

    variances = descriptors.var(axis=0)
    keep      = np.ones(D, dtype=bool)

    # Greedy: scorre le coppie in ordine di correlazione decrescente
    pairs = np.argwhere(np.abs(corr_matrix) > corr_threshold)
    # Ordina per |corr| decrescente
    pairs_corr = np.abs(corr_matrix[pairs[:, 0], pairs[:, 1]])
    order = np.argsort(pairs_corr)[::-1]
    pairs = pairs[order]

    removed = 0
    for i, j in pairs:
        if i >= j:
            continue  # evita duplicati
        if not keep[i] or not keep[j]:
            continue  # già rimossa
        # Rimuove quella con varianza minore
        if variances[i] >= variances[j]:
            keep[j] = False
        else:
            keep[i] = False
        removed += 1

    log.info(f"  Feature rimosse per alta correlazione: {removed}")
    return keep, corr_matrix


# ---------------------------------------------------------------------------
# Step 3 — KNN + Recall su descrittori compressi
# ---------------------------------------------------------------------------
def parse_utm(path: str) -> tuple[float, float] | None:
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
    db_desc_norm: np.ndarray, q_desc_norm: np.ndarray,
    db_coords: np.ndarray, q_coords: np.ndarray,
    mask: np.ndarray,
    recall_values: list,
) -> dict[int, float]:
    """Applica la maschera, ri-normalizza L2 e calcola recall."""
    db_compressed = db_desc_norm[:, mask]
    q_compressed  = q_desc_norm[:, mask]
    # Ri-normalizza dopo aver rimosso le feature
    db_compressed = db_compressed / (np.linalg.norm(db_compressed, axis=1, keepdims=True) + 1e-8)
    q_compressed  = q_compressed  / (np.linalg.norm(q_compressed,  axis=1, keepdims=True) + 1e-8)
    return recall_at_n(db_compressed, q_compressed, db_coords, q_coords, recall_values=recall_values)


# ---------------------------------------------------------------------------
# Step 4 — Sweep soglie su validation set
# ---------------------------------------------------------------------------
def threshold_sweep(
    gsv_stats: dict,
    val_db_norm: np.ndarray, val_q_norm: np.ndarray,
    val_db_coords: np.ndarray, val_q_coords: np.ndarray,
    act_thresholds: list, var_thresholds: list, corr_threshold: float,
    corr_mask: np.ndarray,
    recall_values: list,
) -> list[dict]:
    rows = []
    for act_t in act_thresholds:
        for var_t in var_thresholds:
            act_mask   = build_activation_mask(gsv_stats, act_t, var_t)
            final_mask = act_mask & corr_mask
            n_kept     = int(final_mask.sum())
            n_total    = len(final_mask)

            if n_kept < 10:
                continue  # troppo aggressivo

            rec = apply_mask_and_eval(
                val_db_norm, val_q_norm, val_db_coords, val_q_coords,
                final_mask, recall_values,
            )
            rows.append({
                "act_threshold": act_t,
                "var_threshold": var_t,
                "corr_threshold": corr_threshold,
                "n_kept": n_kept,
                "n_total": n_total,
                "compression_pct": round(100 * (1 - n_kept / n_total), 2),
                **{f"R@{n}": round(v, 4) for n, v in rec.items()},
            })
            log.info(
                f"  act={act_t:.2f} var={var_t:.4f} → "
                f"{n_kept}/{n_total} feat ({100*(1-n_kept/n_total):.1f}% rimosso)  "
                f"R@1={rec[recall_values[0]]:.2f}%"
            )
    return rows


# ---------------------------------------------------------------------------
# Step 5 — Grad-CAM per feature specifiche
# ---------------------------------------------------------------------------
def compute_gradcam(
    model: torch.nn.Module,
    image_path: str,
    image_size: tuple,
    feature_indices: list[int],
    device: torch.device,
    target_layer_name: str | None = None,
) -> dict[int, np.ndarray]:
    """
    Calcola Grad-CAM per ogni feature_index specificato.
    Ritorna dict {feature_idx: heatmap (H, W)} normalizzata in [0,1].

    Aggancia il gradiente all'ultimo layer convoluzionale trovato nel modello.
    """
    # Trova l'ultimo layer conv del backbone
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            target_layer_name_found = name

    if target_layer is None:
        log.warning("  Nessun layer Conv2d trovato — Grad-CAM non disponibile")
        return {}

    log.debug(f"  Grad-CAM target layer: {target_layer_name_found}")

    # Hook per catturare feature maps e gradienti
    activations = {}
    gradients   = {}

    def fwd_hook(module, input, output):
        activations["feat"] = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    # Preprocessing immagine
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
        x.requires_grad_(False)

        desc = model(x)                     # (1, D)
        score = desc[0, feat_idx]           # scalar
        score.backward(retain_graph=True)

        if "feat" not in activations or "feat" not in gradients:
            continue

        # GAP dei gradienti sui canali spaziali: (C,)
        weights = gradients["feat"][0].mean(dim=(-2, -1))   # (C,)
        cam     = (weights[:, None, None] * activations["feat"][0]).sum(dim=0)  # (H', W')
        cam     = F.relu(cam).cpu().numpy()

        # Normalizza in [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        heatmaps[feat_idx] = cam

    h_fwd.remove()
    h_bwd.remove()

    return heatmaps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Estensione 6.3 — Feature Reduction per VPR")
    p.add_argument("--methods",        nargs="+", default=["cosplace", "megaloc"])
    p.add_argument("--act_threshold",  type=float, default=None,
                   help="Soglia attivazione fissa. Se None, fa sweep e sceglie automaticamente.")
    p.add_argument("--var_threshold",  type=float, default=None,
                   help="Soglia varianza fissa. Se None, fa sweep.")
    p.add_argument("--corr_threshold", type=float, default=0.95,
                   help="Soglia correlazione (default: 0.95)")
    p.add_argument("--recall_values",  nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--device",         type=str, default="auto")
    p.add_argument("--n_gradcam_imgs", type=int, default=5,
                   help="Numero di immagini per Grad-CAM (default: 5)")
    p.add_argument("--overwrite",      action="store_true")
    return p.parse_args()


def resolve_device(s):
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

    log.info(f"Device  : {device}")
    log.info(f"Metodi  : {args.methods}")
    log.info(f"Corr th.: {args.corr_threshold}")

    summary_rows = []

    for method_name in args.methods:
        log.info(f"\n{'='*65}")
        log.info(f"Metodo: {method_name.upper()}")

        method_dir = OUT_DIR / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        image_size = MODEL_IMAGE_SIZES[method_name]

        # ----------------------------------------------------------------
        # Carica modello
        # ----------------------------------------------------------------
        log.info("  Caricamento modello...")
        try:
            model = MODEL_LOADERS[method_name]()
            model = model.to(device)
        except Exception as e:
            log.warning(f"  SKIP — {e}")
            continue

        # ----------------------------------------------------------------
        # Step 1 — Descrittori GSV-XS (grezzi, per analisi statistica)
        # ----------------------------------------------------------------
        gsv_dir = DESC_DIR / "gsv_xs_train" / method_name
        gsv_raw_path = gsv_dir / "database_descriptors_raw.npy"

        if gsv_raw_path.exists() and not args.overwrite:
            log.info("  GSV-XS già estratto, carico da disco...")
            gsv_raw  = np.load(gsv_raw_path)
            gsv_paths = list(np.load(gsv_dir / "database_paths.npy"))
        else:
            log.info("  Estrazione GSV-XS training set...")
            gsv_folder = DATA_DIR / "gsv_xs" / "train"
            if not gsv_folder.exists():
                log.error(f"  {gsv_folder} non trovata. Scarica il dataset prima.")
                continue
            gsv_raw, gsv_paths = extract_raw_descriptors(
                model, gsv_folder, image_size, args.batch_size, args.num_workers, device)
            gsv_dir.mkdir(parents=True, exist_ok=True)
            gsv_norm = gsv_raw / (np.linalg.norm(gsv_raw, axis=1, keepdims=True) + 1e-8)
            np.save(gsv_dir / "database_descriptors_raw.npy", gsv_raw)
            np.save(gsv_dir / "database_descriptors.npy",     gsv_norm)
            np.save(gsv_dir / "database_paths.npy", np.array([str(p) for p in gsv_paths]))
            # Salva placeholder query (stessa cartella — non usato per retrieval)
            np.save(gsv_dir / "query_descriptors.npy",     gsv_norm)
            np.save(gsv_dir / "query_descriptors_raw.npy", gsv_raw)
            np.save(gsv_dir / "query_paths.npy", np.array([str(p) for p in gsv_paths]))

        D = gsv_raw.shape[1]
        log.info(f"  GSV-XS: {gsv_raw.shape[0]} immagini, dim={D}")

        # ----------------------------------------------------------------
        # Step 2 — Analisi attivazioni
        # ----------------------------------------------------------------
        log.info("  Analisi attivazioni...")
        stats = analyze_activations(gsv_raw)
        np.save(method_dir / "activation_stats.npy", stats["activation_rate"])
        np.save(method_dir / "variance_stats.npy",   stats["variance"])
        np.save(method_dir / "entropy_stats.npy",    stats["entropy_norm"])

        dead_features = int((stats["activation_rate"] < 0.01).sum())
        low_var       = int((stats["variance"] < 1e-4).sum())
        log.info(f"  Feature quasi-mai attivate (<1%): {dead_features}/{D}")
        log.info(f"  Feature a bassa varianza (<1e-4): {low_var}/{D}")

        # ----------------------------------------------------------------
        # Step 3 — Analisi correlazione
        # ----------------------------------------------------------------
        corr_path = method_dir / "correlation_matrix.npy"
        corr_mask_path = method_dir / "mask_correlation.npy"

        if corr_mask_path.exists() and not args.overwrite:
            log.info("  Matrice correlazione già calcolata, carico maschera...")
            corr_mask = np.load(corr_mask_path)
            corr_matrix = np.load(corr_path) if corr_path.exists() else None
        else:
            corr_mask, corr_matrix = build_correlation_mask(gsv_raw, args.corr_threshold)
            np.save(corr_mask_path, corr_mask)
            if D <= 5000:  # salva la matrice solo se non troppo grande
                np.save(corr_path, corr_matrix.astype(np.float32))
            else:
                log.info(f"  Matrice {D}×{D} troppo grande da salvare — skip")

        log.info(f"  Feature rimosse per correlazione > {args.corr_threshold}: {(~corr_mask).sum()}")

        # ----------------------------------------------------------------
        # Step 4 — Sweep soglie su validation set
        # ----------------------------------------------------------------
        log.info("  Caricamento val set (sf_xs_val)...")
        val_db_norm, val_db_paths = extract_normalized_descriptors(
            model, VAL_DATASET["database"], image_size,
            args.batch_size, args.num_workers, device)
        val_q_norm, val_q_paths = extract_normalized_descriptors(
            model, VAL_DATASET["queries"], image_size,
            args.batch_size, args.num_workers, device)
        val_db_coords = get_coords(val_db_paths)
        val_q_coords  = get_coords(val_q_paths)

        # Recall baseline (nessuna compressione)
        baseline_recall = recall_at_n(val_db_norm, val_q_norm, val_db_coords, val_q_coords, recall_values=rv)
        log.info(f"  Baseline val R@1={baseline_recall[rv[0]]:.2f}%")

        if args.act_threshold is not None and args.var_threshold is not None:
            act_thresholds = [args.act_threshold]
            var_thresholds = [args.var_threshold]
        else:
            act_thresholds = [0.01, 0.05, 0.10, 0.20]
            var_thresholds = [1e-5, 1e-4, 1e-3, 5e-3]

        log.info("  Sweep soglie su val set...")
        sweep_rows = threshold_sweep(
            stats, val_db_norm, val_q_norm, val_db_coords, val_q_coords,
            act_thresholds, var_thresholds, args.corr_threshold,
            corr_mask, rv,
        )

        # Salva sweep CSV
        if sweep_rows:
            sweep_csv = method_dir / "threshold_sweep.csv"
            with open(sweep_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sweep_rows)

        # Scegli la soglia ottimale:
        # massimizza compressione mantenendo R@1 entro 2% dal baseline
        r1_key = f"R@{rv[0]}"
        baseline_r1 = baseline_recall[rv[0]]
        valid_configs = [
            r for r in sweep_rows
            if r.get(r1_key, 0) >= baseline_r1 - 2.0
        ]
        if valid_configs:
            best = max(valid_configs, key=lambda r: r["compression_pct"])
        elif sweep_rows:
            best = max(sweep_rows, key=lambda r: r["compression_pct"])
            log.warning("  Nessuna config entro 2% dal baseline — uso quella più compressa")
        else:
            log.error("  Nessun risultato sweep — uso soglie di default")
            best = {"act_threshold": 0.05, "var_threshold": 1e-4}

        best_act_t = best["act_threshold"]
        best_var_t = best["var_threshold"]
        log.info(f"  Soglie scelte: act={best_act_t}  var={best_var_t}")

        # ----------------------------------------------------------------
        # Maschera finale
        # ----------------------------------------------------------------
        act_mask   = build_activation_mask(stats, best_act_t, best_var_t)
        final_mask = act_mask & corr_mask

        n_kept   = int(final_mask.sum())
        n_total  = D
        comp_pct = 100 * (1 - n_kept / n_total)

        np.save(method_dir / "mask_activation.npy", act_mask)
        np.save(method_dir / "mask_final.npy",      final_mask)

        log.info(f"  Maschera finale: {n_kept}/{n_total} feature mantenute ({comp_pct:.1f}% rimosso)")

        # ----------------------------------------------------------------
        # Step 5 — Valutazione su test set
        # ----------------------------------------------------------------
        log.info("  Valutazione sui test set...")
        test_results = {}

        for ds_name, ds_cfg in TEST_DATASETS.items():
            if not ds_cfg["database"].exists():
                log.warning(f"  SKIP {ds_name} — cartella non trovata")
                continue

            db_norm, db_paths = extract_normalized_descriptors(
                model, ds_cfg["database"], image_size,
                args.batch_size, args.num_workers, device)
            q_norm, q_paths = extract_normalized_descriptors(
                model, ds_cfg["queries"], image_size,
                args.batch_size, args.num_workers, device)
            db_coords = get_coords(db_paths)
            q_coords  = get_coords(q_paths)

            # Baseline (full dim)
            rec_full = recall_at_n(db_norm, q_norm, db_coords, q_coords, recall_values=rv)
            # Compresso
            rec_comp = apply_mask_and_eval(db_norm, q_norm, db_coords, q_coords, final_mask, rv)

            test_results[ds_name] = {
                "recall_full":       {f"R@{n}": round(v, 4) for n, v in rec_full.items()},
                "recall_compressed": {f"R@{n}": round(v, 4) for n, v in rec_comp.items()},
                "recall_delta":      {f"R@{n}": round(rec_comp[n] - rec_full[n], 4) for n in rv},
            }
            log.info(
                f"  [{ds_name}] Full R@1={rec_full[rv[0]]:.2f}%  "
                f"Compressed R@1={rec_comp[rv[0]]:.2f}%  "
                f"Δ={rec_comp[rv[0]]-rec_full[rv[0]]:+.2f}%"
            )

        # ----------------------------------------------------------------
        # Step 6 — Grad-CAM
        # ----------------------------------------------------------------
        log.info(f"  Grad-CAM su {args.n_gradcam_imgs} immagini...")

        # Seleziona feature eliminate e mantenute (le prime N per varianza)
        removed_feats = np.where(~final_mask)[0]
        kept_feats    = np.where(final_mask)[0]

        # Ordina per varianza decrescente per scegliere le più "rappresentative"
        removed_by_var = removed_feats[np.argsort(stats["variance"][removed_feats])[::-1]][:5]
        kept_by_var    = kept_feats[np.argsort(stats["variance"][kept_feats])[::-1]][:5]

        # Seleziona immagini di esempio da GSV-XS
        np.random.seed(42)
        sample_indices = np.random.choice(len(gsv_paths), min(args.n_gradcam_imgs, len(gsv_paths)), replace=False)
        sample_paths   = [str(gsv_paths[i]) for i in sample_indices]

        for img_idx, img_path in enumerate(sample_paths):
            # Grad-CAM feature eliminate
            heatmaps_removed = compute_gradcam(
                model, img_path, image_size,
                list(removed_by_var.astype(int)), device,
            )
            if heatmaps_removed:
                np.save(
                    method_dir / f"gradcam_removed_img{img_idx}.npy",
                    np.stack(list(heatmaps_removed.values())),  # (n_feats, H', W')
                )

            # Grad-CAM feature mantenute
            heatmaps_kept = compute_gradcam(
                model, img_path, image_size,
                list(kept_by_var.astype(int)), device,
            )
            if heatmaps_kept:
                np.save(
                    method_dir / f"gradcam_kept_img{img_idx}.npy",
                    np.stack(list(heatmaps_kept.values())),
                )

        # Salva anche i path delle immagini usate per Grad-CAM (utile per il report)
        with open(method_dir / "gradcam_image_paths.json", "w") as f:
            json.dump(sample_paths, f, indent=2)
        np.save(method_dir / "gradcam_removed_feat_indices.npy", removed_by_var)
        np.save(method_dir / "gradcam_kept_feat_indices.npy",    kept_by_var)

        # ----------------------------------------------------------------
        # Salva risultati
        # ----------------------------------------------------------------
        result = {
            "method":          method_name,
            "descriptor_dim":  D,
            "n_kept":          n_kept,
            "n_removed":       n_total - n_kept,
            "compression_pct": round(comp_pct, 2),
            "act_threshold":   best_act_t,
            "var_threshold":   best_var_t,
            "corr_threshold":  args.corr_threshold,
            "n_dead_features": dead_features,
            "n_corr_removed":  int((~corr_mask).sum()),
            "baseline_val":    {f"R@{n}": round(v, 4) for n, v in baseline_recall.items()},
            "test_results":    test_results,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(method_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)

        summary_rows.append(result)
        log.info(f"  Risultati salvati in {method_dir.relative_to(ROOT)}/")

    # ----------------------------------------------------------------
    # Summary CSV
    # ----------------------------------------------------------------
    if summary_rows:
        _save_summary(summary_rows, rv)


def _save_summary(rows, rv):
    csv_path = OUT_DIR / "summary.csv"
    fieldnames = [
        "method", "descriptor_dim", "n_kept", "n_removed",
        "compression_pct", "act_threshold", "var_threshold", "corr_threshold",
        "n_dead_features", "n_corr_removed",
    ]
    for ds in TEST_DATASETS:
        for n in rv:
            fieldnames += [f"{ds}_R@{n}_full", f"{ds}_R@{n}_compressed", f"{ds}_R@{n}_delta"]

    flat_rows = []
    for r in rows:
        flat = {k: r.get(k) for k in fieldnames if k in r}
        for ds_name, ds_res in r.get("test_results", {}).items():
            for n in rv:
                key = f"R@{n}"
                flat[f"{ds_name}_R@{n}_full"]       = ds_res["recall_full"].get(key)
                flat[f"{ds_name}_R@{n}_compressed"] = ds_res["recall_compressed"].get(key)
                flat[f"{ds_name}_R@{n}_delta"]      = ds_res["recall_delta"].get(key)
        flat_rows.append(flat)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)

    log.info(f"\nSummary salvato → {csv_path.relative_to(ROOT)}")

    # Stampa tabella
    print("\n" + "="*70)
    print("FEATURE REDUCTION — RISULTATI FINALI")
    print("="*70)
    for r in rows:
        print(f"\n{r['method'].upper()}: {r['descriptor_dim']}D → {r['n_kept']}D "
              f"({r['compression_pct']:.1f}% rimosso)")
        for ds_name, ds_res in r.get("test_results", {}).items():
            for n in rv:
                key = f"R@{n}"
                b = ds_res["recall_full"].get(key, 0)
                a = ds_res["recall_compressed"].get(key, 0)
                print(f"  {ds_name:<18} R@{n}: {b:.2f}% → {a:.2f}% ({a-b:+.2f}%)")


if __name__ == "__main__":
    main()