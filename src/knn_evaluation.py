import argparse
import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DESC_DIR = ROOT / "logs" / "descriptors"
RESULTS_DIR = ROOT / "logs" / "results"

GPS_THRESHOLD_M = 25.0


def parse_utm_from_path(path: str) -> tuple[float, float] | None:
    """Extract the first two '@'-separated numeric fields from a filename (UTM easting/northing)."""
    stem = Path(path).stem
    nums = []
    for p in stem.split("@"):
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
    coords = [parse_utm_from_path(str(p)) or (float("nan"), float("nan")) for p in paths]
    return np.array(coords, dtype=np.float64)


def knn_l2(db_desc: np.ndarray, q_desc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """KNN search via FAISS flat L2 index. Returns (indices, distances), ascending distance."""
    index = faiss.IndexFlatL2(db_desc.shape[1])
    index.add(db_desc.astype(np.float32))
    distances, indices = index.search(q_desc.astype(np.float32), k)
    return indices, distances


def knn_dot(db_desc: np.ndarray, q_desc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """KNN search via FAISS flat inner-product index. Returns (indices, scores), descending score."""
    index = faiss.IndexFlatIP(db_desc.shape[1])
    index.add(db_desc.astype(np.float32))
    scores, indices = index.search(q_desc.astype(np.float32), k)
    return indices, scores


def compute_recall_and_per_query(
    predictions: np.ndarray,
    distances: np.ndarray,
    q_coords: np.ndarray,
    db_coords: np.ndarray,
    recall_values: list[int],
    threshold_m: float = GPS_THRESHOLD_M,
) -> tuple[dict[int, float], list[dict]]:
    """
    Compute Recall@N (a query counts as a hit if any of its top-N predictions
    is within `threshold_m` meters) plus a per-query breakdown for later analysis.

    Returns (recall: {N: percentage}, per_query: list of per-query dicts).
    """
    N_q = len(q_coords)
    correct = {n: 0 for n in recall_values}
    per_query = []

    for i in range(N_q):
        qc = q_coords[i]
        has_gps = not np.isnan(qc).any()

        top1_idx = int(predictions[i, 0])
        top1_dist = float(np.linalg.norm(db_coords[top1_idx] - qc)) if has_gps else float("nan")
        top1_score = float(distances[i, 0])

        is_correct = {}
        for n in recall_values:
            if not has_gps:
                is_correct[n] = False
                continue
            top_n = predictions[i, :n]
            dists = np.linalg.norm(db_coords[top_n] - qc, axis=1)
            hit = bool(np.any(dists <= threshold_m))
            is_correct[n] = hit
            if hit:
                correct[n] += 1

        per_query.append({
            "query_idx": i,
            "has_gps": has_gps,
            "top1_geo_dist_m": round(top1_dist, 2) if has_gps else None,
            "top1_score": round(top1_score, 6),
            "correct_r1": is_correct.get(1, False),
            "correct_r5": is_correct.get(5, False),
            "correct_r10": is_correct.get(10, False),
            "correct_r20": is_correct.get(20, False),
        })

    recall = {n: 100.0 * correct[n] / N_q if N_q > 0 else 0.0 for n in recall_values}
    return recall, per_query


def load_descriptors(desc_dir: Path):
    """
    Load database/query descriptors (normalized + raw) and their paths.

    _norm : L2-normalized, used for dot product (= cosine similarity)
    _raw  : original model output, used for true L2 distance

    Falls back to normalized descriptors for both if the raw files are missing
    (older extraction run) — L2 becomes approximate cosine distance in that case.
    """
    db_norm = np.load(desc_dir / "database_descriptors.npy")
    q_norm = np.load(desc_dir / "query_descriptors.npy")

    raw_db_path = desc_dir / "database_descriptors_raw.npy"
    raw_q_path = desc_dir / "query_descriptors_raw.npy"
    if raw_db_path.exists() and raw_q_path.exists():
        db_raw = np.load(raw_db_path)
        q_raw = np.load(raw_q_path)
    else:
        log.warning(
            f"  Raw descriptors not found in {desc_dir.name}. "
            "Re-run extract_descriptors.py with --overwrite for correct L2 vs dot comparison. "
            "Falling back to normalized descriptors (L2 ~= dot)."
        )
        db_raw, q_raw = db_norm, q_norm

    return (
        db_norm, db_raw,
        q_norm, q_raw,
        np.load(desc_dir / "database_paths.npy"),
        np.load(desc_dir / "query_paths.npy"),
    )


def descriptors_available(d: Path) -> bool:
    """Check that the required (non-raw) descriptor files exist. Raw files are optional."""
    files = ["database_descriptors.npy", "query_descriptors.npy", "database_paths.npy", "query_paths.npy"]
    return all((d / f).exists() for f in files)


def discover_combinations(datasets_filter, methods_filter):
    """Walk logs/descriptors/<dataset>/<method>/ and collect valid, filtered combos."""
    combos = []
    if not DESC_DIR.exists():
        return combos
    for ds_dir in sorted(DESC_DIR.iterdir()):
        if not ds_dir.is_dir() or (datasets_filter and ds_dir.name not in datasets_filter):
            continue
        for m_dir in sorted(ds_dir.iterdir()):
            if not m_dir.is_dir() or (methods_filter and m_dir.name not in methods_filter):
                continue
            if descriptors_available(m_dir):
                combos.append((ds_dir.name, m_dir.name, m_dir))
    return combos


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--metrics", nargs="+", default=["l2", "dot"], choices=["l2", "dot"])
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--recall_values", nargs="+", type=int, default=[1, 5, 10, 20])
    p.add_argument("--threshold_m", type=float, default=GPS_THRESHOLD_M)
    p.add_argument("--save_predictions", action="store_true", default=True,
                    help="Always save KNN predictions (needed by image_matching_evaluation.py)")
    return p.parse_args()


def main():
    """Run KNN retrieval and Recall@N evaluation for every dataset/method/metric combination."""
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_dir = RESULTS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    combos = discover_combinations(args.datasets, args.methods)
    if not combos:
        log.error(f"No descriptors found in {DESC_DIR}.\nRun first: python src/extract_descriptors.py")
        return

    log.info(f"Root      : {ROOT}")
    log.info(f"Combos    : {len(combos)}")
    log.info(f"Metrics   : {args.metrics}  K={args.k}  Recall@{args.recall_values}\n")

    all_results = []
    all_per_query = []

    for dataset_name, method_name, desc_dir in combos:
        log.info(f"{'='*60}")
        log.info(f"Dataset: {dataset_name}  |  Method: {method_name}")

        db_norm, db_raw, q_norm, q_raw, db_paths, q_paths = load_descriptors(desc_dir)
        db_coords = extract_utm_coords(db_paths)
        q_coords = extract_utm_coords(q_paths)
        valid_q = int(np.sum(~np.isnan(q_coords[:, 0])))
        log.info(f"  db: {db_norm.shape}  query: {q_norm.shape}  valid GPS: {valid_q}/{len(q_coords)}")

        for metric in args.metrics:
            log.info(f"\n  [{metric.upper()}]")

            # l2  -> raw descriptors: true Euclidean distance
            # dot -> normalized descriptors: equivalent to cosine similarity
            db_desc = db_raw if metric == "l2" else db_norm
            q_desc = q_raw if metric == "l2" else q_norm

            t_knn = time.time()
            if metric == "l2":
                preds, dists = knn_l2(db_desc, q_desc, args.k)
            else:
                preds, dists = knn_dot(db_desc, q_desc, args.k)
            knn_time = time.time() - t_knn
            knn_time_per_query = knn_time / len(q_coords) if len(q_coords) > 0 else 0

            # predictions are always saved: image_matching_evaluation.py depends on them
            np.save(preds_dir / f"{dataset_name}_{method_name}_{metric}_preds.npy", preds)
            np.save(preds_dir / f"{dataset_name}_{method_name}_{metric}_scores.npy", dists)

            recall, per_query = compute_recall_and_per_query(
                preds, dists, q_coords, db_coords,
                recall_values=args.recall_values, threshold_m=args.threshold_m)

            recall_str = "  ".join(f"R@{n}={v:.2f}%" for n, v in recall.items())
            log.info(f"  {recall_str}")
            log.info(f"  KNN time: {knn_time:.3f}s total  |  {knn_time_per_query*1000:.3f}ms/query")

            top1_scores = dists[:, 0]
            correct_mask = np.array([pq["correct_r1"] for pq in per_query])

            row = {
                "dataset": dataset_name,
                "method": method_name,
                "metric": metric,
                "k": args.k,
                "n_queries": len(q_coords),
                "n_db": len(db_paths),
                "n_valid_gps": valid_q,
                "descriptor_dim": db_desc.shape[1],
                "knn_time_s": round(knn_time, 4),
                "knn_time_ms_per_query": round(knn_time_per_query * 1000, 4),
                "top1_score_mean": round(float(top1_scores.mean()), 6),
                "top1_score_std": round(float(top1_scores.std()), 6),
                "top1_score_correct_mean": round(float(top1_scores[correct_mask].mean()), 6) if correct_mask.any() else None,
                "top1_score_incorrect_mean": round(float(top1_scores[~correct_mask].mean()), 6) if (~correct_mask).any() else None,
                "threshold_m": args.threshold_m,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            row.update({f"R@{n}": round(v, 4) for n, v in recall.items()})
            all_results.append(row)

            for pq in per_query:
                all_per_query.append({"dataset": dataset_name, "method": method_name, "metric": metric, **pq})

    if not all_results:
        log.warning("No results produced.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "recall_table.csv", index=False)

    with open(RESULTS_DIR / "knn_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    df_pq = pd.DataFrame(all_per_query)
    df_pq.to_csv(RESULTS_DIR / "knn_per_query.csv", index=False)

    log.info(f"\nFiles saved in {RESULTS_DIR.relative_to(ROOT)}:")
    log.info("  recall_table.csv, knn_results.json, knn_per_query.csv")
    log.info(f"  predictions/  ({len(list((RESULTS_DIR/'predictions').glob('*.npy')))} .npy files)")

    recall_cols = [f"R@{n}" for n in args.recall_values if f"R@{n}" in df.columns]
    print("\n" + "="*70)
    print("RECALL@N SUMMARY")
    print("="*70)
    print(df[["dataset", "method", "metric"] + recall_cols + ["knn_time_ms_per_query"]].to_string(index=False))

    if set(args.metrics) == {"l2", "dot"} and "R@1" in df.columns:
        print("\nL2 vs DOT COMPARISON (R@1)")
        print("-"*50)
        for (ds, m), grp in df.groupby(["dataset", "method"]):
            l2 = grp.loc[grp.metric == "l2", "R@1"].values
            dot = grp.loc[grp.metric == "dot", "R@1"].values
            if l2.size and dot.size:
                diff = dot[0] - l2[0]
                winner = "DOT" if diff > 0 else ("L2" if diff < 0 else "TIE")
                print(f"  {ds:20s} | {m:12s} | L2={l2[0]:.2f}%  DOT={dot[0]:.2f}%  -> {winner} (d{abs(diff):.2f}%)")


if __name__ == "__main__":
    main()