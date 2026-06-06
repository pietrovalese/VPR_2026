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
            ├── database_paths.npy
            └── query_paths.npy

    logs/results/extraction_metrics.csv   — tempi e throughput per ogni run

Uso (dalla root del progetto):
    python src/extract_descriptors.py
    python src/extract_descriptors.py --methods cosplace megaloc
    python src/extract_descriptors.py --datasets sf_xs_test tokyo_xs --overwrite
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
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
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
DEPS_DIR     = ROOT / "deps"
VPR_EVAL_DIR = DEPS_DIR / "VPR-methods-evaluation"
DATA_DIR     = ROOT / "data"
OUTPUT_DIR   = ROOT / "logs" / "descriptors"
RESULTS_DIR  = ROOT / "logs" / "results"

if VPR_EVAL_DIR.exists():
    if str(VPR_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(VPR_EVAL_DIR))
else:
    print(f"[WARNING] {VPR_EVAL_DIR} non trovato.")

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
    "gsv_xs_train": {
        "database": DATA_DIR / "gsv_xs" / "train",
        "queries":  DATA_DIR / "gsv_xs" / "train",
    },
}

# ---------------------------------------------------------------------------
# Configurazione modelli
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    image_size: tuple
    descriptor_dim: int
    loader: callable


def _get_model(*args, **kwargs):
    try:
        from vpr_models import get_model
        return get_model(*args, **kwargs)
    except ImportError:
        raise ImportError("Impossibile importare vpr_models. Esegui: git submodule update --init --recursive")


def _load_cosplace():  return _get_model(method="cosplace", backbone="ResNet18", descriptors_dimension=512).eval()
def _load_megaloc():   return _get_model(method="megaloc").eval()
def _load_netvlad():   return _get_model(method="netvlad", backbone="VGG16", descriptors_dimension=4096).eval()
def _load_mixvpr():    return _get_model(method="mixvpr", descriptors_dimension=4096).eval()
def _load_convap():    return _get_model(method="convap", descriptors_dimension=4096).eval()
def _load_sfrs():      return _get_model(method="sfrs").eval()
def _load_boq_resnet(): return _get_model(method="boq", backbone="ResNet50", descriptors_dimension=16384).eval()
def _load_boq_dino():  return _get_model(method="boq", backbone="Dinov2", descriptors_dimension=12288).eval()
def _load_dinomix():   return _get_model(method="dinomix").eval()
def _load_clique_mining(): return _get_model(method="clique-mining").eval()


MODEL_CONFIGS: dict[str, ModelConfig] = {
    #"clique_mining": ModelConfig("clique_mining", (322, 322), 8448,  _load_clique_mining),
    "cosplace":      ModelConfig("cosplace",      (512, 512), 512,   _load_cosplace),
    "megaloc":       ModelConfig("megaloc",        (322, 322), 8448,  _load_megaloc),
    "netvlad":       ModelConfig("netvlad",        (480, 640), 4096,  _load_netvlad),
    "mixvpr":        ModelConfig("mixvpr",         (320, 320), 4096,  _load_mixvpr),
    #"convap":        ModelConfig("convap",         (320, 320), 4096,  _load_convap),
    #"sfrs":          ModelConfig("sfrs",           (480, 640), 4096,  _load_sfrs),
    #"boq_resnet":    ModelConfig("boq_resnet",     (322, 322), 16384, _load_boq_resnet),
    #"boq_dino":      ModelConfig("boq_dino",       (322, 322), 12288, _load_boq_dino),
    #"dinomix":       ModelConfig("dinomix",        (224, 224), 4096,  _load_dinomix),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
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
# Estrazione — ritorna anche statistiche sui descrittori
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_descriptors(
    model: nn.Module,
    folder: Path,
    image_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, list[Path], dict]:
    dataset = ImageFolderDataset(folder, image_size)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    model = model.to(device)
    t0    = time.time()

    all_desc_raw = []
    for images, _ in tqdm(loader, desc=f"  {folder.name}", leave=False):
        desc = model(images.to(device))
        all_desc_raw.append(desc.cpu().numpy())

    elapsed = time.time() - t0
    desc_raw   = np.concatenate(all_desc_raw, axis=0)
    desc_array = desc_raw / (np.linalg.norm(desc_raw, axis=1, keepdims=True) + 1e-8)  # L2-norm
    n_images   = len(dataset)

    stats = {
        "n_images":          n_images,
        "descriptor_dim":    desc_array.shape[1],
        "time_s":            round(elapsed, 3),
        "throughput_img_s":  round(n_images / elapsed, 2) if elapsed > 0 else 0,
        "memory_mb":         round(desc_array.nbytes / 1e6, 3),
        "desc_mean":         round(float(desc_array.mean()), 6),
        "desc_std":          round(float(desc_array.std()),  6),
        "desc_min":          round(float(desc_array.min()),  6),
        "desc_max":          round(float(desc_array.max()),  6),
    }
    return desc_array, desc_raw, dataset.paths, stats


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_descriptors(out_dir, db_desc, db_desc_raw, q_desc, q_desc_raw, db_paths, q_paths):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "database_descriptors.npy",     db_desc)      # L2-normalizzati (per dot product)
    np.save(out_dir / "database_descriptors_raw.npy", db_desc_raw)  # grezzi (per L2 distance)
    np.save(out_dir / "query_descriptors.npy",        q_desc)
    np.save(out_dir / "query_descriptors_raw.npy",    q_desc_raw)
    np.save(out_dir / "database_paths.npy",           np.array([str(p) for p in db_paths]))
    np.save(out_dir / "query_paths.npy",              np.array([str(p) for p in q_paths]))
    log.info(f"  → {out_dir.relative_to(ROOT)}  "
             f"[db: {db_desc.shape}, query: {q_desc.shape}]")


def already_extracted(out_dir: Path) -> bool:
    return all((out_dir / f).exists() for f in [
        "database_descriptors.npy", "database_descriptors_raw.npy",
        "query_descriptors.npy",    "query_descriptors_raw.npy",
        "database_paths.npy",       "query_paths.npy",
    ])


def append_metrics_csv(rows: list[dict]):
    """Aggiunge righe al CSV cumulativo di metriche di estrazione."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "extraction_metrics.csv"
    fieldnames = [
        "timestamp", "method", "dataset", "split",
        "n_images", "descriptor_dim", "time_s", "throughput_img_s",
        "memory_mb", "desc_mean", "desc_std", "desc_min", "desc_max",
        "device", "batch_size",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    log.info(f"  Metriche salvate → {csv_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",    nargs="+", choices=list(DATASET_CONFIGS.keys()),
                   default=list(DATASET_CONFIGS.keys()))
    p.add_argument("--methods",     nargs="+", choices=list(MODEL_CONFIGS.keys()),
                   default=list(MODEL_CONFIGS.keys()))
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--overwrite",   action="store_true")
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def main():
    args   = parse_args()
    device = resolve_device(args.device)
    ts     = time.strftime("%Y-%m-%dT%H:%M:%S")

    log.info(f"Root    : {ROOT}")
    log.info(f"Device  : {device}")
    log.info(f"Metodi  : {args.methods}")
    log.info(f"Dataset : {args.datasets}")

    all_metric_rows = []

    for method_name in args.methods:
        cfg = MODEL_CONFIGS[method_name]
        log.info(f"\n{'='*60}\nCaricamento: {method_name.upper()}")

        t_load = time.time()
        try:
            model = cfg.loader()
        except Exception as e:
            log.warning(f"  SKIP — {e}")
            continue
        load_time = round(time.time() - t_load, 2)
        log.info(f"  Modello caricato in {load_time}s")

        for dataset_name in args.datasets:
            ds_cfg  = DATASET_CONFIGS[dataset_name]
            out_dir = OUTPUT_DIR / dataset_name / method_name

            if already_extracted(out_dir) and not args.overwrite:
                log.info(f"  [{dataset_name}] già estratto — usa --overwrite")
                continue

            db_folder = ds_cfg["database"]
            q_folder  = ds_cfg["queries"]
            if not db_folder.exists() or not q_folder.exists():
                log.warning(f"  [{dataset_name}] SKIP — cartelle non trovate")
                continue

            log.info(f"\n  Dataset: {dataset_name}")

            db_desc, db_desc_raw, db_paths, db_stats = extract_descriptors(
                model, db_folder, cfg.image_size, args.batch_size, args.num_workers, device)
            q_desc, q_desc_raw, q_paths, q_stats = extract_descriptors(
                model, q_folder, cfg.image_size, args.batch_size, args.num_workers, device)

            save_descriptors(out_dir, db_desc, db_desc_raw, q_desc, q_desc_raw, db_paths, q_paths)

            # Salva metadata JSON per questo run
            meta = {
                "method": method_name, "dataset": dataset_name,
                "image_size": list(cfg.image_size),
                "descriptor_dim": cfg.descriptor_dim,
                "model_load_time_s": load_time,
                "device": str(device),
                "batch_size": args.batch_size,
                "database": db_stats,
                "queries": q_stats,
                "timestamp": ts,
            }
            with open(out_dir / "extraction_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            # Righe per CSV cumulativo
            for split, stats in [("database", db_stats), ("queries", q_stats)]:
                all_metric_rows.append({
                    "timestamp": ts, "method": method_name,
                    "dataset": dataset_name, "split": split,
                    "device": str(device), "batch_size": args.batch_size,
                    **stats,
                })

            log.info(
                f"  DB: {db_stats['n_images']} img, "
                f"{db_stats['time_s']}s, {db_stats['throughput_img_s']} img/s, "
                f"{db_stats['memory_mb']} MB"
            )
            log.info(
                f"  Q : {q_stats['n_images']} img, "
                f"{q_stats['time_s']}s, {q_stats['throughput_img_s']} img/s"
            )

    if all_metric_rows:
        append_metrics_csv(all_metric_rows)

    log.info("\nEstrazione completata.")


if __name__ == "__main__":
    main()