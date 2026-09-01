import argparse
import csv
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
VPR_EVAL_DIR = ROOT / "deps" / "VPR-methods-evaluation"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "logs" / "descriptors"
RESULTS_DIR = ROOT / "logs" / "results"

if VPR_EVAL_DIR.exists():
    if str(VPR_EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(VPR_EVAL_DIR))
else:
    print(f"[WARNING] {VPR_EVAL_DIR} not found.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATASET_CONFIGS = {
    "sf_xs_test":   {"database": DATA_DIR / "sf_xs" / "test" / "database", "queries": DATA_DIR / "sf_xs" / "test" / "queries"},
    "sf_xs_val":    {"database": DATA_DIR / "sf_xs" / "val" / "database",  "queries": DATA_DIR / "sf_xs" / "val" / "queries"},
    "tokyo_xs":     {"database": DATA_DIR / "tokyo_xs" / "test" / "database", "queries": DATA_DIR / "tokyo_xs" / "test" / "queries"},
    "svox_sun":     {"database": DATA_DIR / "svox" / "images" / "test" / "gallery", "queries": DATA_DIR / "svox" / "images" / "test" / "queries_sun"},
    "svox_night":   {"database": DATA_DIR / "svox" / "images" / "test" / "gallery", "queries": DATA_DIR / "svox" / "images" / "test" / "queries_night"},
    "gsv_xs_train": {"database": DATA_DIR / "gsv_xs" / "train", "queries": DATA_DIR / "gsv_xs" / "train"},
    "svox_sun_train":   {"database": DATA_DIR / "svox" / "images" / "train" / "gallery", "queries": DATA_DIR / "svox" / "images" / "train" / "queries_sun"},
    "svox_night_train": {"database": DATA_DIR / "svox" / "images" / "train" / "gallery", "queries": DATA_DIR / "svox" / "images" / "train" / "queries_night"},
}


@dataclass
class ModelConfig:
    name: str
    image_size: tuple
    descriptor_dim: int
    loader: callable


def _get_model(*args, **kwargs):
    """Import and call vpr_models.get_model, with a clearer error if the submodule is missing."""
    try:
        from vpr_models import get_model
        return get_model(*args, **kwargs)
    except ImportError:
        raise ImportError("Cannot import vpr_models. Run: git submodule update --init --recursive")


def _load_cosplace():  return _get_model(method="cosplace", backbone="ResNet18", descriptors_dimension=512).eval()
def _load_megaloc():   return _get_model(method="megaloc").eval()
def _load_netvlad():   return _get_model(method="netvlad", backbone="VGG16", descriptors_dimension=4096).eval()
def _load_mixvpr():    return _get_model(method="mixvpr", descriptors_dimension=4096).eval()


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "cosplace": ModelConfig("cosplace", (512, 512), 512, _load_cosplace),
    "megaloc":  ModelConfig("megaloc",  (322, 322), 8448, _load_megaloc),
    "netvlad":  ModelConfig("netvlad",  (480, 640), 4096, _load_netvlad),
    "mixvpr":   ModelConfig("mixvpr",   (320, 320), 4096, _load_mixvpr),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageFolderDataset(Dataset):
    """Recursively loads images from a folder and applies resize/normalize for the given model."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, folder: Path, image_size: tuple):
        self.paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in self.EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {folder}")
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


@torch.no_grad()
def extract_descriptors(
    model: nn.Module,
    folder: Path,
    image_size: tuple,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    out_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[Path], dict]:
    """
    Run the model over all images in `folder` and write descriptors to disk as memmaps
    (constant RAM usage regardless of dataset size).

    If out_dir is given, memmaps are written there permanently; otherwise a temp dir
    is used and everything is loaded into RAM before being deleted.

    Returns (desc_norm, desc_raw, paths, stats).
    """
    dataset = ImageFolderDataset(folder, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=(device.type == "cuda"))
    model = model.to(device)
    n_images = len(dataset)

    # infer descriptor dim from first sample
    first_images, _ = next(iter(loader))
    D = model(first_images[:1].to(device)).shape[1]

    tmp_dir = Path(tempfile.mkdtemp(prefix="vpr_desc_")) if out_dir is None else out_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_path = tmp_dir / "_raw_tmp.npy"
    norm_path = tmp_dir / "_norm_tmp.npy"

    mm_raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=(n_images, D))
    mm_norm = np.lib.format.open_memmap(norm_path, mode="w+", dtype=np.float32, shape=(n_images, D))

    t0 = time.time()
    idx = 0
    for images, _ in tqdm(loader, desc=f"  {folder.name}", leave=False):
        desc = model(images.to(device)).cpu().numpy()
        B = desc.shape[0]
        norms = np.linalg.norm(desc, axis=1, keepdims=True)

        mm_raw[idx:idx + B] = desc
        mm_norm[idx:idx + B] = desc / (norms + 1e-8)
        mm_raw.flush()
        mm_norm.flush()
        idx += B
    elapsed = time.time() - t0

    # stats computed in chunks to avoid loading the full array
    chunk_size = 1024
    mean_acc = std_acc = min_val = max_val = None
    for start in range(0, n_images, chunk_size):
        chunk = mm_norm[start:start + chunk_size]
        if mean_acc is None:
            mean_acc, std_acc, min_val, max_val = chunk.mean(), chunk.std(), chunk.min(), chunk.max()
        else:
            mean_acc = (mean_acc + chunk.mean()) / 2
            min_val = min(min_val, chunk.min())
            max_val = max(max_val, chunk.max())

    stats = {
        "n_images": n_images,
        "descriptor_dim": D,
        "time_s": round(elapsed, 3),
        "throughput_img_s": round(n_images / elapsed, 2) if elapsed > 0 else 0,
        "memory_mb": round(n_images * D * 4 / 1e6, 3),
        "desc_mean": round(float(mean_acc), 6),
        "desc_std": round(float(std_acc), 6),
        "desc_min": round(float(min_val), 6),
        "desc_max": round(float(max_val), 6),
    }

    if out_dir is None:
        desc_raw_out, desc_norm_out = np.array(mm_raw), np.array(mm_norm)
        del mm_raw, mm_norm
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return desc_norm_out, desc_raw_out, dataset.paths, stats

    return mm_norm, mm_raw, dataset.paths, stats


def save_descriptors(out_dir, db_desc, db_desc_raw, q_desc, q_desc_raw, db_paths, q_paths):
    """Save descriptors and paths to out_dir; renames memmaps in place instead of copying when possible."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_or_rename(arr, dest: Path):
        if isinstance(arr, np.memmap):
            src = Path(arr.filename)
            if src.parent == out_dir and src != dest:
                arr._mmap.close()
                src.rename(dest)
                return
        np.save(dest, arr)

    _save_or_rename(db_desc, out_dir / "database_descriptors.npy")
    _save_or_rename(db_desc_raw, out_dir / "database_descriptors_raw.npy")
    _save_or_rename(q_desc, out_dir / "query_descriptors.npy")
    _save_or_rename(q_desc_raw, out_dir / "query_descriptors_raw.npy")
    np.save(out_dir / "database_paths.npy", np.array([str(p) for p in db_paths]))
    np.save(out_dir / "query_paths.npy", np.array([str(p) for p in q_paths]))

    for tmp in out_dir.glob("_*_tmp.npy"):
        tmp.unlink(missing_ok=True)

    log.info(f"  -> {out_dir.relative_to(ROOT)}  [db: {db_desc.shape}, query: {q_desc.shape}]")


def already_extracted(out_dir: Path) -> bool:
    """Check whether all expected output files already exist for this dataset/method."""
    files = ["database_descriptors.npy", "database_descriptors_raw.npy",
             "query_descriptors.npy", "query_descriptors_raw.npy",
             "database_paths.npy", "query_paths.npy"]
    return all((out_dir / f).exists() for f in files)


def append_metrics_csv(rows: list[dict]):
    """Append extraction stats to the cumulative results CSV, creating it with a header if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "extraction_metrics.csv"
    fieldnames = ["timestamp", "method", "dataset", "split", "n_images", "descriptor_dim",
                  "time_s", "throughput_img_s", "memory_mb", "desc_mean", "desc_std",
                  "desc_min", "desc_max", "device", "batch_size"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    log.info(f"  Metrics saved -> {csv_path.relative_to(ROOT)}")


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=list(DATASET_CONFIGS.keys()))
    p.add_argument("--methods", nargs="+", choices=list(MODEL_CONFIGS.keys()), default=list(MODEL_CONFIGS.keys()))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_device(s: str) -> torch.device:
    """Resolve the --device argument to a torch.device, auto-detecting CUDA when requested."""
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def main():
    """Extract and save descriptors for every requested method/dataset combination."""
    args = parse_args()
    device = resolve_device(args.device)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    log.info(f"Root    : {ROOT}")
    log.info(f"Device  : {device}")
    log.info(f"Methods : {args.methods}")
    log.info(f"Datasets: {args.datasets}")

    all_metric_rows = []

    for method_name in args.methods:
        cfg = MODEL_CONFIGS[method_name]
        log.info(f"\n{'='*60}\nLoading: {method_name.upper()}")

        t_load = time.time()
        try:
            model = cfg.loader()
        except Exception as e:
            log.warning(f"  SKIP - {e}")
            continue
        load_time = round(time.time() - t_load, 2)
        log.info(f"  Model loaded in {load_time}s")

        for dataset_name in args.datasets:
            ds_cfg = DATASET_CONFIGS[dataset_name]
            out_dir = OUTPUT_DIR / dataset_name / method_name

            if already_extracted(out_dir) and not args.overwrite:
                log.info(f"  [{dataset_name}] already extracted - use --overwrite")
                continue

            db_folder, q_folder = ds_cfg["database"], ds_cfg["queries"]
            if not db_folder.exists() or not q_folder.exists():
                log.warning(f"  [{dataset_name}] SKIP - folders not found")
                continue

            log.info(f"\n  Dataset: {dataset_name}")

            # write directly to out_dir via memmap, then finalize with save_descriptors
            db_tmp, q_tmp = out_dir / "_db_tmp", out_dir / "_q_tmp"

            db_desc, db_desc_raw, db_paths, db_stats = extract_descriptors(
                model, db_folder, cfg.image_size, args.batch_size, args.num_workers, device, out_dir=db_tmp)
            q_desc, q_desc_raw, q_paths, q_stats = extract_descriptors(
                model, q_folder, cfg.image_size, args.batch_size, args.num_workers, device, out_dir=q_tmp)

            save_descriptors(out_dir, db_desc, db_desc_raw, q_desc, q_desc_raw, db_paths, q_paths)

            for tmp in (db_tmp, q_tmp):
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)

            meta = {
                "method": method_name, "dataset": dataset_name,
                "image_size": list(cfg.image_size), "descriptor_dim": cfg.descriptor_dim,
                "model_load_time_s": load_time, "device": str(device), "batch_size": args.batch_size,
                "database": db_stats, "queries": q_stats, "timestamp": ts,
            }
            with open(out_dir / "extraction_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            for split, stats in [("database", db_stats), ("queries", q_stats)]:
                all_metric_rows.append({
                    "timestamp": ts, "method": method_name, "dataset": dataset_name, "split": split,
                    "device": str(device), "batch_size": args.batch_size, **stats,
                })

            log.info(f"  DB: {db_stats['n_images']} img, {db_stats['time_s']}s, {db_stats['throughput_img_s']} img/s, {db_stats['memory_mb']} MB")
            log.info(f"  Q : {q_stats['n_images']} img, {q_stats['time_s']}s, {q_stats['throughput_img_s']} img/s")

    if all_metric_rows:
        append_metrics_csv(all_metric_rows)

    log.info("\nDone.")


if __name__ == "__main__":
    main()