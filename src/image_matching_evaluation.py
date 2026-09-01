import argparse
import csv
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT               = Path(__file__).resolve().parent.parent
DEPS_DIR           = ROOT / "deps"
IMAGE_MATCHING_DIR = DEPS_DIR / "image-matching-models"
DESC_DIR           = ROOT / "logs" / "descriptors"
PREDS_DIR          = ROOT / "logs" / "results" / "predictions"
RESULTS_DIR        = ROOT / "logs" / "results" / "matching"

if IMAGE_MATCHING_DIR.exists():
    if str(IMAGE_MATCHING_DIR) not in sys.path:
        sys.path.insert(0, str(IMAGE_MATCHING_DIR))
else:
    print(f"[WARNING] {IMAGE_MATCHING_DIR} not found.", flush=True)

# Logging with forced flush - avoids buffering that would stall the output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)


class FlushHandler(logging.StreamHandler):
    """StreamHandler that flushes immediately after every emitted record."""

    def emit(self, record):
        super().emit(record)
        self.flush()


for h in logging.root.handlers:
    h.__class__ = FlushHandler

REQUIRED_MATCHERS = ["superglue", "loftr", "superpoint-lg"]
GPS_THRESHOLD_M   = 25.0


def pprint(msg: str):
    """Print with a guaranteed flush, so progress is visible immediately in redirected logs."""
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# GPS coordinates
# ---------------------------------------------------------------------------
def parse_utm_from_path(path: str) -> tuple[float, float] | None:
    """Extract the first two '@'-separated numeric fields from a filename (UTM easting/northing)."""
    stem  = Path(path).stem
    parts = stem.split("@")
    nums  = []
    for p in parts:
        p = p.strip()
        if p:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        if len(nums) == 2:
            break
    return (nums[0], nums[1]) if len(nums) >= 2 else None


def extract_utm_coords(paths: np.ndarray) -> np.ndarray:
    """Parse UTM coords for every path; unparseable ones become (nan, nan)."""
    coords = []
    for p in paths:
        r = parse_utm_from_path(str(p))
        coords.append(r if r is not None else (float("nan"), float("nan")))
    return np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# Recall@N
# ---------------------------------------------------------------------------
def compute_recall_and_correct(
    predictions: np.ndarray,
    q_coords: np.ndarray,
    db_coords: np.ndarray,
    recall_values: list[int],
    threshold_m: float = GPS_THRESHOLD_M,
) -> tuple[dict[int, float], np.ndarray]:
    """Compute Recall@N and the per-query R@1 correctness mask."""
    N_q       = len(q_coords)
    correct   = {n: 0 for n in recall_values}
    correct_r1 = np.zeros(N_q, dtype=bool)

    for i in range(N_q):
        qc = q_coords[i]
        if np.isnan(qc).any():
            continue
        for n in recall_values:
            top_n = predictions[i, :n]
            dists = np.linalg.norm(db_coords[top_n] - qc, axis=1)
            if np.any(dists <= threshold_m):
                correct[n] += 1
                if n == 1:
                    correct_r1[i] = True

    recall = {n: 100.0 * correct[n] / N_q if N_q > 0 else 0.0 for n in recall_values}
    return recall, correct_r1


# ---------------------------------------------------------------------------
# Inlier counting
# ---------------------------------------------------------------------------
def count_inliers(result: dict) -> int:
    """Extract the inlier count from a matcher result dict, trying a few known key names."""
    if "num_inliers" in result:
        v = result["num_inliers"]
        return int(v) if v is not None else 0
    for key in ("keypoints0", "mkpts0", "matched_kpts0"):
        if key in result and result[key] is not None:
            t = result[key]
            return int(t.shape[0]) if hasattr(t, "shape") else 0
    return 0


# ---------------------------------------------------------------------------
# Matching for a single query
# ---------------------------------------------------------------------------
def run_matching_for_query(
    matcher,
    q_path: str,
    db_paths: np.ndarray,
    pred_indices: np.ndarray,
    img_size: int,
    num_preds: int,
) -> tuple[np.ndarray, float, float]:
    """Match one query image against its top-`num_preds` KNN predictions and return inlier counts and timing."""
    img0    = matcher.load_image(q_path, resize=img_size)
    inliers = np.zeros(num_preds, dtype=np.int32)
    t_start = time.time()

    for rank, db_idx in enumerate(pred_indices[:num_preds]):
        db_path = str(db_paths[db_idx])
        try:
            img1   = matcher.load_image(db_path, resize=img_size)
            result = matcher(deepcopy(img0), img1)
            inliers[rank] = count_inliers(result)
        except Exception as e:
            log.debug(f"    Matching failed [{rank}]: {e}")
            inliers[rank] = 0

    elapsed = time.time() - t_start
    return inliers, elapsed, elapsed / num_preds if num_preds > 0 else 0.0


# ---------------------------------------------------------------------------
# Combination discovery
# ---------------------------------------------------------------------------
def discover_combinations(datasets_filter, methods_filter, metric):
    """Walk logs/descriptors/<dataset>/<method>/ and collect combos that have matching KNN predictions."""
    combos = []
    if not DESC_DIR.exists():
        return combos
    for ds_dir in sorted(DESC_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        if datasets_filter and ds_dir.name not in datasets_filter:
            continue
        for m_dir in sorted(ds_dir.iterdir()):
            if not m_dir.is_dir():
                continue
            if methods_filter and m_dir.name not in methods_filter:
                continue
            required = ["database_paths.npy", "query_paths.npy"]
            if not all((m_dir / f).exists() for f in required):
                continue
            preds_file = PREDS_DIR / f"{ds_dir.name}_{m_dir.name}_{metric}_preds.npy"
            if not preds_file.exists():
                pprint(
                    f"[WARN] Predictions not found: {preds_file.name}\n"
                    f"       Run first: python src/knn_evaluation.py --metrics {metric}"
                )
                continue
            combos.append((ds_dir.name, m_dir.name, m_dir, preds_file))
    return combos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",      nargs="+", default=None)
    p.add_argument("--methods",       nargs="+", default=None)
    p.add_argument("--matchers",      nargs="+", default=REQUIRED_MATCHERS)
    p.add_argument("--metric",        type=str,  default="dot", choices=["l2", "dot"])
    p.add_argument("--num_preds",     type=int,  default=20)
    p.add_argument("--img_size",      type=int,  default=512)
    p.add_argument("--recall_values", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--device",        type=str,  default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--overwrite",     action="store_true")
    return p.parse_args()


def resolve_device(s: str) -> str:
    """Resolve the --device argument, auto-detecting CUDA when requested."""
    if s == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Re-rank KNN predictions with every requested image-matching method and collect metrics."""
    args   = parse_args()
    device = resolve_device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pprint(f"[INFO] Loading 'matching' module from {IMAGE_MATCHING_DIR} ...")
    try:
        from matching import get_matcher
        pprint("[INFO] 'matching' module loaded.")
    except ImportError as e:
        pprint(f"[ERROR] Cannot import 'matching': {e}")
        return

    combos = discover_combinations(args.datasets, args.methods, args.metric)
    if not combos:
        pprint(
            "[ERROR] No combination available.\n"
            "Run first:\n"
            "  python src/extract_descriptors.py\n"
            f"  python src/knn_evaluation.py --metrics {args.metric}"
        )
        return

    pprint(f"[INFO] Device   : {device}")
    pprint(f"[INFO] Matchers : {args.matchers}")
    pprint(f"[INFO] Combos   : {len(combos)} (dataset x method)")
    pprint(f"[INFO] Num preds: {args.num_preds}  |  Img size: {args.img_size}x{args.img_size}")
    pprint(f"[INFO] Recall@N : {args.recall_values}")
    pprint("")

    summary_rows = []

    for matcher_name in args.matchers:
        pprint(f"\n{'='*65}")
        pprint(f"[STEP] Matcher: {matcher_name.upper()}")
        pprint(f"[INFO] Loading weights... (may take a while on first run)")
        sys.stdout.flush()

        t_load = time.time()
        try:
            matcher = get_matcher(matcher_name, device=device)
        except Exception as e:
            pprint(f"[WARN] SKIP {matcher_name} - could not load: {e}")
            continue
        matcher_load_time = round(time.time() - t_load, 2)
        pprint(f"[OK]   Matcher loaded in {matcher_load_time}s")

        for dataset_name, method_name, desc_dir, preds_file in combos:
            out_dir = RESULTS_DIR / f"{dataset_name}_{method_name}_{matcher_name}"

            if (out_dir / "results.json").exists() and not args.overwrite:
                pprint(f"[SKIP] {dataset_name} | {method_name} - already computed (use --overwrite)")
                with open(out_dir / "results.json") as f:
                    summary_rows.append(json.load(f))
                continue

            pprint(f"\n[INFO] Dataset: {dataset_name}  |  VPR: {method_name}")

            db_paths  = np.load(desc_dir / "database_paths.npy")
            q_paths   = np.load(desc_dir / "query_paths.npy")
            knn_preds = np.load(preds_file)

            db_coords = extract_utm_coords(db_paths)
            q_coords  = extract_utm_coords(q_paths)
            N_q       = len(q_paths)
            K         = knn_preds.shape[1]
            num_preds = min(args.num_preds, K)

            pprint(f"[INFO] {N_q} queries  |  K={K}  |  re-ranking top-{num_preds}")

            # Recall BEFORE re-ranking
            recall_before, correct_r1_before = compute_recall_and_correct(
                knn_preds, q_coords, db_coords, args.recall_values
            )
            before_str = "  ".join(f"R@{n}={v:.2f}%" for n, v in recall_before.items())
            pprint(f"[INFO] Recall before re-ranking: {before_str}")

            # Result matrices
            all_inliers        = np.zeros((N_q, num_preds), dtype=np.int32)
            all_reranked_preds = knn_preds.copy()
            times_query        = []
            times_pair         = []

            t_total = time.time()

            for q_idx in tqdm(
                range(N_q),
                desc=f"  {matcher_name} | {dataset_name} | {method_name}",
                unit="query",
                dynamic_ncols=True,
                file=sys.stdout,
            ):
                inliers, t_q, t_p = run_matching_for_query(
                    matcher, str(q_paths[q_idx]), db_paths,
                    knn_preds[q_idx], args.img_size, num_preds,
                )
                all_inliers[q_idx] = inliers
                times_query.append(t_q)
                times_pair.append(t_p)

                rerank_order = np.argsort(inliers)[::-1]
                all_reranked_preds[q_idx, :num_preds] = knn_preds[q_idx, :num_preds][rerank_order]

            total_time = time.time() - t_total

            # Recall AFTER re-ranking
            recall_after, _ = compute_recall_and_correct(
                all_reranked_preds, q_coords, db_coords, args.recall_values
            )

            # Inlier statistics
            top1_inliers      = all_inliers[:, 0]
            inliers_correct   = top1_inliers[correct_r1_before]
            inliers_incorrect = top1_inliers[~correct_r1_before]

            inlier_stats = {
                "mean_all":            round(float(top1_inliers.mean()), 3),
                "std_all":             round(float(top1_inliers.std()),  3),
                "mean_correct":        round(float(inliers_correct.mean()),   3) if len(inliers_correct)   > 0 else None,
                "mean_incorrect":      round(float(inliers_incorrect.mean()), 3) if len(inliers_incorrect) > 0 else None,
                "median_all":          round(float(np.median(top1_inliers)), 3),
                "p25":                 round(float(np.percentile(top1_inliers, 25)), 3),
                "p75":                 round(float(np.percentile(top1_inliers, 75)), 3),
                "p90":                 round(float(np.percentile(top1_inliers, 90)), 3),
            }

            timing_stats = {
                "total_time_s":         round(total_time, 2),
                "avg_time_per_query_s": round(float(np.mean(times_query)), 4),
                "std_time_per_query_s": round(float(np.std(times_query)),  4),
                "avg_time_per_pair_s":  round(float(np.mean(times_pair)),  4),
                "matcher_load_time_s":  matcher_load_time,
            }

            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "inliers.npy",           all_inliers)
            np.save(out_dir / "reranked_preds.npy",    all_reranked_preds)
            np.save(out_dir / "per_query_inliers.npy", top1_inliers)
            np.save(out_dir / "correct_mask.npy",      correct_r1_before)

            row = {
                "dataset":       dataset_name,
                "vpr_method":    method_name,
                "matcher":       matcher_name,
                "num_preds":     num_preds,
                "img_size":      args.img_size,
                "metric":        args.metric,
                "n_queries":     N_q,
                "recall_before": {f"R@{n}": round(v, 4) for n, v in recall_before.items()},
                "recall_after":  {f"R@{n}": round(v, 4) for n, v in recall_after.items()},
                "recall_delta":  {f"R@{n}": round(recall_after[n] - recall_before[n], 4) for n in args.recall_values},
                "inlier_stats":  inlier_stats,
                "timing":        timing_stats,
                "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(out_dir / "results.json", "w") as f:
                json.dump(row, f, indent=2)
            summary_rows.append(row)

            after_str = "  ".join(f"R@{n}={v:.2f}%" for n, v in recall_after.items())
            delta_str = "  ".join(f"dR@{n}={recall_after[n]-recall_before[n]:+.2f}%" for n in args.recall_values)
            pprint(f"[OK]   Recall after re-ranking : {after_str}")
            pprint(f"[OK]   Delta                   : {delta_str}")
            pprint(
                f"[OK]   Time: {total_time:.1f}s total  |  "
                f"{timing_stats['avg_time_per_query_s']:.3f}s/query  |  "
                f"{timing_stats['avg_time_per_pair_s']:.4f}s/pair"
            )
            pprint(
                f"[OK]   Top-1 inliers: mean={inlier_stats['mean_all']:.1f}  "
                f"correct={inlier_stats['mean_correct']}  "
                f"incorrect={inlier_stats['mean_incorrect']}"
            )

    if not summary_rows:
        pprint("[WARN] No result produced.")
        return

    _print_summary(summary_rows, args.recall_values)
    _save_summary_csv(summary_rows, args.recall_values)


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------
def _print_summary(rows, recall_values):
    """Print a formatted before/after Recall@N summary table to stdout."""
    pprint("\n" + "="*90)
    pprint("FINAL SUMMARY - Re-ranking with Image Matching")
    pprint("="*90)
    header = f"{'VPR':<15} {'Matcher':<16} {'Dataset':<18}"
    for n in recall_values:
        header += f"  R@{n}(B) R@{n}(A)    D"
    header += "  t/query(s)"
    pprint(header)
    pprint("-"*len(header))

    for r in sorted(rows, key=lambda x: (x["dataset"], x["vpr_method"], x["matcher"])):
        line = f"{r['vpr_method']:<15} {r['matcher']:<16} {r['dataset']:<18}"
        for n in recall_values:
            key = f"R@{n}"
            b   = r["recall_before"].get(key, 0)
            a   = r["recall_after"].get(key, 0)
            d   = a - b
            line += f"  {b:6.2f} {a:6.2f} {d:+5.2f}"
        line += f"  {r['timing']['avg_time_per_query_s']:.3f}"
        pprint(line)


def _save_summary_csv(rows, recall_values):
    """Flatten all per-combo result dicts into logs/results/matching_summary.csv."""
    csv_path = RESULTS_DIR.parent / "matching_summary.csv"
    fieldnames = [
        "dataset", "vpr_method", "matcher", "num_preds", "img_size", "metric",
        "n_queries", "total_time_s", "avg_time_per_query_s", "avg_time_per_pair_s",
        "inliers_mean_all", "inliers_mean_correct", "inliers_mean_incorrect",
        "inliers_median", "inliers_p25", "inliers_p75", "inliers_p90",
    ]
    for n in recall_values:
        fieldnames += [f"R@{n}_before", f"R@{n}_after", f"R@{n}_delta"]

    flat_rows = []
    for r in rows:
        flat = {k: r.get(k) for k in ["dataset", "vpr_method", "matcher",
                                        "num_preds", "img_size", "metric", "n_queries"]}
        flat.update({
            "total_time_s":           r["timing"]["total_time_s"],
            "avg_time_per_query_s":   r["timing"]["avg_time_per_query_s"],
            "avg_time_per_pair_s":    r["timing"]["avg_time_per_pair_s"],
            "inliers_mean_all":       r["inlier_stats"]["mean_all"],
            "inliers_mean_correct":   r["inlier_stats"]["mean_correct"],
            "inliers_mean_incorrect": r["inlier_stats"]["mean_incorrect"],
            "inliers_median":         r["inlier_stats"]["median_all"],
            "inliers_p25":            r["inlier_stats"]["p25"],
            "inliers_p75":            r["inlier_stats"]["p75"],
            "inliers_p90":            r["inlier_stats"]["p90"],
        })
        for n in recall_values:
            key = f"R@{n}"
            flat[f"R@{n}_before"] = r["recall_before"].get(key)
            flat[f"R@{n}_after"]  = r["recall_after"].get(key)
            flat[f"R@{n}_delta"]  = r["recall_delta"].get(key)
        flat_rows.append(flat)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)

    pprint(f"\n[OK] matching_summary.csv saved -> {csv_path}")


if __name__ == "__main__":
    main()