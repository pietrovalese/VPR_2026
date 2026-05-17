"""
src/extract_descriptors.py

Estrae e salva i descrittori VPR per database e query di tutti i dataset.
Usa i modelli già disponibili in deps/VPR-methods-evaluation.

Output:
    logs/descriptors/
    └── <dataset_name>/
        └── <method_name>/
            ├── database_descriptors.npy   # (N_db, D)
            ├── query_descriptors.npy      # (N_q, D)
            ├── database_paths.npy         # path assoluti immagini database
            └── query_paths.npy            # path assoluti immagini query

Uso (dalla root del progetto):
    python src/extract_descriptors.py
    python src/extract_descriptors.py --methods cosplace megaloc
    python src/extract_descriptors.py --datasets sf_xs_test tokyo_xs --overwrite
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ROOT è la cartella radice del progetto (parent di src/)
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
DEPS_DIR    = ROOT / "deps"
VPR_EVAL_DIR = DEPS_DIR / "VPR-methods-evaluation"
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "logs" / "descriptors"

# Rende importabili i moduli dentro deps/VPR-methods-evaluation.
# Va fatto QUI, a livello di modulo, prima che qualsiasi loader venga definito,
# altrimenti Python non trova vpr_models al momento della chiamata.
if VPR_EVAL_DIR.exists():
    if str(VPR_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(VPR_EVAL_DIR))
else:
    print(f"[WARNING] {VPR_EVAL_DIR} non trovato. "
          "Esegui: git submodule update --init --recursive")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurazione dataset
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "sf_xs_test": {
        "database": DATA_DIR / "sf_xs" / "test" / "database",
        "queries":  DATA_DIR / "sf_xs" / "test" / "queries",
    },
    "sf_xs_val": {
        "database": DATA_DIR / "sf_xs" / "val" / "database",
        "queries":  DATA_DIR / "sf_xs" / "val" / "queries",
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


# ---------------------------------------------------------------------------
# Configurazione modelli
# "loader" è una callable () -> nn.Module in eval mode.
# Per aggiungere un nuovo modello: aggiungi una entry in MODEL_CONFIGS.
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    image_size: tuple          # (H, W)
    descriptor_dim: int
    loader: callable


def _get_model(*args, **kwargs):
    """Wrapper che importa get_model da vpr_models (dentro deps/VPR-methods-evaluation)."""
    try:
        from vpr_models import get_model
        return get_model(*args, **kwargs)
    except ImportError:
        vpr_dir = VPR_EVAL_DIR / "vpr_models"
        content = [p.name for p in vpr_dir.iterdir()] if vpr_dir.exists() else "CARTELLA NON ESISTE"
        raise ImportError(
            f"Impossibile importare vpr_models.\n"
            f"  VPR_EVAL_DIR  = {VPR_EVAL_DIR}\n"
            f"  vpr_models/   = {content}\n"
            f"  sys.path[:3]  = {sys.path[:3]}\n"
            f"Soluzione: git submodule update --init --recursive"
        )


def _load_cosplace():
    return _get_model(method="cosplace", backbone="ResNet18", descriptors_dimension=512).eval()


def _load_megaloc():
    return _get_model(method="megaloc").eval()


def _load_netvlad():
    return _get_model(method="netvlad", backbone="VGG16", descriptors_dimension=4096).eval()


def _load_mixvpr():
    return _get_model(method="mixvpr", descriptors_dimension=4096).eval()


# Registry dei modelli — aggiungi qui senza toccare altro codice
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "cosplace": ModelConfig("cosplace", (512, 512), 512,  _load_cosplace),
    "megaloc":  ModelConfig("megaloc",  (322, 322), 4096, _load_megaloc),
    "netvlad":  ModelConfig("netvlad",  (480, 640), 4096, _load_netvlad),
    "mixvpr":   ModelConfig("mixvpr",   (320, 320), 4096, _load_mixvpr),
}


# ---------------------------------------------------------------------------
# Dataset PyTorch generico per cartelle di immagini
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ImageFolderDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, folder: Path, image_size: tuple):
        self.paths = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"Nessuna immagine trovata in {folder}")
        H, W = image_size
        self.transform = transforms.Compose([
            transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


# ---------------------------------------------------------------------------
# Estrazione
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_descriptors(
    model: nn.Module,
    folder: Path,
    image_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, list[Path]]:
    dataset = ImageFolderDataset(folder, image_size)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    all_desc = []
    model = model.to(device)
    for images, _ in tqdm(loader, desc=f"  {folder.name}", leave=False):
        desc = model(images.to(device))
        all_desc.append(F.normalize(desc, p=2, dim=1).cpu().numpy())
    return np.concatenate(all_desc, axis=0), dataset.paths


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_descriptors(out_dir, db_desc, q_desc, db_paths, q_paths):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "database_descriptors.npy", db_desc)
    np.save(out_dir / "query_descriptors.npy",    q_desc)
    np.save(out_dir / "database_paths.npy",       np.array([str(p) for p in db_paths]))
    np.save(out_dir / "query_paths.npy",          np.array([str(p) for p in q_paths]))
    log.info(f"  → {out_dir.relative_to(ROOT)}  "
             f"[db: {db_desc.shape}, query: {q_desc.shape}]")


def already_extracted(out_dir: Path) -> bool:
    return all((out_dir / f).exists() for f in [
        "database_descriptors.npy", "query_descriptors.npy",
        "database_paths.npy", "query_paths.npy",
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   choices=list(DATASET_CONFIGS.keys()),
                   default=list(DATASET_CONFIGS.keys()))
    p.add_argument("--methods", nargs="+",
                   choices=list(MODEL_CONFIGS.keys()),
                   default=list(MODEL_CONFIGS.keys()))
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      type=str, default="auto",
                   help="'auto' | 'cpu' | 'cuda' | 'cuda:0'")
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def main():
    args   = parse_args()
    device = resolve_device(args.device)

    log.info(f"Root    : {ROOT}")
    log.info(f"Device  : {device}")
    log.info(f"Metodi  : {args.methods}")
    log.info(f"Dataset : {args.datasets}")
    log.info(f"Output  : logs/descriptors/\n")

    for method_name in args.methods:
        cfg = MODEL_CONFIGS[method_name]
        log.info(f"{'='*60}")
        log.info(f"Caricamento: {method_name.upper()}")
        try:
            model = cfg.loader()
        except Exception as e:
            log.warning(f"  SKIP — {e}")
            continue

        for dataset_name in args.datasets:
            ds_cfg  = DATASET_CONFIGS[dataset_name]
            out_dir = OUTPUT_DIR / dataset_name / method_name

            if already_extracted(out_dir) and not args.overwrite:
                log.info(f"  [{dataset_name}] già estratto — usa --overwrite per ricalcolare")
                continue

            db_folder = ds_cfg["database"]
            q_folder  = ds_cfg["queries"]

            if not db_folder.exists():
                log.warning(f"  [{dataset_name}] SKIP — {db_folder} non trovata")
                continue
            if not q_folder.exists():
                log.warning(f"  [{dataset_name}] SKIP — {q_folder} non trovata")
                continue

            log.info(f"\n  Dataset: {dataset_name}")
            t0 = time.time()
            db_desc, db_paths = extract_descriptors(
                model, db_folder, cfg.image_size, args.batch_size, args.num_workers, device)
            q_desc, q_paths = extract_descriptors(
                model, q_folder, cfg.image_size, args.batch_size, args.num_workers, device)
            save_descriptors(out_dir, db_desc, q_desc, db_paths, q_paths)
            log.info(f"  Tempo: {time.time() - t0:.1f}s")

    log.info("\nEstrazione completata.")


if __name__ == "__main__":
    main()