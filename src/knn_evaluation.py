"""
src/knn_evaluation.py

Carica i descrittori da logs/descriptors/, esegue KNN con L2 e dot product
e calcola Recall@N per ogni combinazione metodo × dataset × metrica.

Output:
    logs/results/
        recall_table.csv          — tabella riassuntiva (una riga per combo)
        knn_results.json          — tutti i risultati machine-readable
        knn_per_query.csv         — correttezza per singola query (utile per analisi inliers)
        predictions/              — indici KNN salvati sempre
            <dataset>_<method>_<metric>_preds.npy

Uso:
    python3 knn_evaluation.py
    python3 knn_evaluation.py --methods cosplace --recall_values 1 5 10 20
    python3 knn_evaluation.py --metrics l2 dot   # confronto entrambe
"""

import argparse
import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
DESC_DIR    = ROOT / "logs" / "descriptors"
RESULTS_DIR = ROOT / "logs" / "results"

GPS_THRESHOLD_M = 25.0


# ---------------------------------------------------------------------------
# Coordinate GPS
# ---------------------------------------------------------------------------
def parse_utm_from_path(path: str) -> tuple[float, float] | None:
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
    coords = []
    for p in paths:
        r = parse_utm_from_path(str(p))
        coords.append(r if r is not None else (float("nan"), float("nan")))
    return np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# KNN con FAISS
# ---------------------------------------------------------------------------
def knn_l2(db_desc: np.ndarray, q_desc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (indici, distanze) ordinati per L2 crescente."""
    index = faiss.IndexFlatL2(db_desc.shape[1])
    index.add(db_desc.astype(np.float32))
    distances, indices = index.search(q_desc.astype(np.float32), k)
    return indices, distances


def knn_dot(db_desc: np.ndarray, q_desc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (indici, scores) ordinati per dot product decrescente."""
    index = faiss.IndexFlatIP(db_desc.shape[1])
    index.add(db_desc.astype(np.float32))
    scores, indices = index.search(q_desc.astype(np.float32), k)
    return indices, scores


# ---------------------------------------------------------------------------
# Recall@N + statistiche per query
# ---------------------------------------------------------------------------
def compute_recall_and_per_query(
    predictions: np.ndarray,
    distances: np.ndarray,
    q_coords: np.ndarray,
    db_coords: np.ndarray,
    recall_values: list[int],
    threshold_m: float = GPS_THRESHOLD_M,
) -> tuple[dict[int, float], list[dict]]:
    """
    Ritorna:
        recall      : {N: recall_percentage}
        per_query   : lista di dict con info per ogni query
                      (usata poi per correlazione con inliers)
    """
    N_q     = len(q_coords)
    correct = {n: 0 for n in recall_values}
    per_query = []

    for i in range(N_q):
        qc = q_coords[i]
        has_gps = not np.isnan(qc).any()

        # distanza geo con top-1
        top1_idx  = int(predictions[i, 0])
        top1_dist = float(np.linalg.norm(db_coords[top1_idx] - qc)) if has_gps else float("nan")
        top1_score = float(distances[i, 0])

        # correttezza per ogni N
        is_correct = {}
        for n in recall_values:
            if not has_gps:
                is_correct[n] = False
                continue
            top_n = predictions[i, :n]
            dists = np.linalg.norm(db_coords[top_n] - qc, axis=1)
            hit   = bool(np.any(dists <= threshold_m))
            is_correct[n] = hit
            if hit:
                correct[n] += 1

        per_query.append({
            "query_idx":       i,
            "has_gps":         has_gps,
            "top1_geo_dist_m": round(top1_dist, 2) if has_gps else None,
            "top1_score":      round(top1_score, 6),
            "correct_r1":      is_correct.get(1, False),
            "correct_r5":      is_correct.get(5, False),
            "correct_r10":     is_correct.get(10, False),
            "correct_r20":     is_correct.get(20, False),
        })

    recall = {n: 100.0 * correct[n] / N_q if N_q > 0 else 0.0 for n in recall_values}
    return recall, per_query


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_descriptors(desc_dir: Path):
    """
    Ritorna (db_desc_norm, db_desc_raw, q_desc_norm, q_desc_raw, db_paths, q_paths).
    - _norm : L2-normalizzati → usati per dot product (cosine similarity)
    - _raw  : grezzi           → usati per distanza L2 (diversi da cosine se non normalizzati)
    Se i file _raw non esistono (vecchia estrazione), fallback sui normalizzati con warning.
    """
    db_norm = np.load(desc_dir / "database_descriptors.npy")
    q_norm  = np.load(desc_dir / "query_descriptors.npy")

    raw_db_path = desc_dir / "database_descriptors_raw.npy"
    raw_q_path  = desc_dir / "query_descriptors_raw.npy"
    if raw_db_path.exists() and raw_q_path.exists():
        db_raw = np.load(raw_db_path)
        q_raw  = np.load(raw_q_path)
    else:
        log.warning(
            f"  Descrittori grezzi non trovati in {desc_dir.name}. "
            "Riesegui extract_descriptors.py con --overwrite per il confronto L2 vs dot corretto. "
            "Usando normalizzati come fallback (L2 ≈ dot)."
        )
        db_raw, q_raw = db_norm, q_norm

    return (
        db_norm, db_raw,
        q_norm,  q_raw,
        np.load(desc_dir / "database_paths.npy"),
        np.load(desc_dir / "query_paths.npy"),
    )


def descriptors_available(d: Path) -> bool:
    return all((d / f).exists() for f in [
        "database_descriptors.npy", "query_descriptors.npy",
        "database_paths.npy", "query_paths.npy",
    ])  # raw opzionali — warn in load_descriptors se mancanti


def discover_combinations(datasets_filter, methods_filter):
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
            if descriptors_available(m_dir):
                combos.append((ds_dir.name, m_dir.name, m_dir))
    return combos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",      nargs="+", default=None)
    p.add_argument("--methods",       nargs="+", default=None)
    p.add_argument("--metrics",       nargs="+", default=["l2", "dot"], choices=["l2", "dot"])
    p.add_argument("--k",             type=int,  default=20)
    p.add_argument("--recall_values", nargs="+", type=int, default=[1, 5, 10, 20])
    p.add_argument("--threshold_m",   type=float, default=GPS_THRESHOLD_M)
    p.add_argument("--save_predictions", action="store_true", default=True,
                   help="Salva sempre le predizioni KNN (necessario per image matching)")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_dir = RESULTS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    combos = discover_combinations(args.datasets, args.methods)
    if not combos:
        log.error(
            f"Nessun descrittore trovato in {DESC_DIR}.\n"
            "Esegui prima: python src/extract_descriptors.py"
        )
        return

    log.info(f"Root    : {ROOT}")
    log.info(f"Combos  : {len(combos)}")
    log.info(f"Metriche: {args.metrics}  K={args.k}  Recall@{args.recall_values}\n")

    all_results     = []
    all_per_query   = []

    for dataset_name, method_name, desc_dir in combos:
        log.info(f"{'='*60}")
        log.info(f"Dataset: {dataset_name}  |  Metodo: {method_name}")

        db_norm, db_raw, q_norm, q_raw, db_paths, q_paths = load_descriptors(desc_dir)
        db_coords = extract_utm_coords(db_paths)
        q_coords  = extract_utm_coords(q_paths)
        valid_q   = int(np.sum(~np.isnan(q_coords[:, 0])))
        log.info(f"  db: {db_norm.shape}  query: {q_norm.shape}  GPS valido: {valid_q}/{len(q_coords)}")

        for metric in args.metrics:
            log.info(f"\n  [{metric.upper()}]")

            # L2  → descrittori grezzi (non normalizzati): distanza euclidea reale
            # dot → descrittori L2-normalizzati: equivale a cosine similarity
            db_desc = db_raw  if metric == "l2" else db_norm
            q_desc  = q_raw   if metric == "l2" else q_norm

            # Timing KNN
            t_knn = time.time()
            if metric == "l2":
                preds, dists = knn_l2(db_desc, q_desc, args.k)
            else:
                preds, dists = knn_dot(db_desc, q_desc, args.k)
            knn_time = time.time() - t_knn
            knn_time_per_query = knn_time / len(q_coords) if len(q_coords) > 0 else 0

            # Salva predizioni (sempre — necessario per image_matching_evaluation.py)
            fname = preds_dir / f"{dataset_name}_{method_name}_{metric}_preds.npy"
            np.save(fname, preds)
            # Salva anche le distanze/scores per analisi successive
            dfname = preds_dir / f"{dataset_name}_{method_name}_{metric}_scores.npy"
            np.save(dfname, dists)

            # Recall + info per query
            recall, per_query = compute_recall_and_per_query(
                preds, dists, q_coords, db_coords,
                recall_values=args.recall_values,
                threshold_m=args.threshold_m,
            )

            recall_str = "  ".join(f"R@{n}={v:.2f}%" for n, v in recall.items())
            log.info(f"  {recall_str}")
            log.info(f"  KNN time: {knn_time:.3f}s total  |  {knn_time_per_query*1000:.3f}ms/query")

            # Statistiche top-1 score
            top1_scores = dists[:, 0]
            correct_mask = np.array([pq["correct_r1"] for pq in per_query])

            row = {
                "dataset":              dataset_name,
                "method":               method_name,
                "metric":               metric,
                "k":                    args.k,
                "n_queries":            len(q_coords),
                "n_db":                 len(db_paths),
                "n_valid_gps":          valid_q,
                "descriptor_dim":       db_desc.shape[1],
                "knn_time_s":           round(knn_time, 4),
                "knn_time_ms_per_query": round(knn_time_per_query * 1000, 4),
                "top1_score_mean":      round(float(top1_scores.mean()), 6),
                "top1_score_std":       round(float(top1_scores.std()),  6),
                "top1_score_correct_mean":   round(float(top1_scores[correct_mask].mean()), 6) if correct_mask.any() else None,
                "top1_score_incorrect_mean": round(float(top1_scores[~correct_mask].mean()), 6) if (~correct_mask).any() else None,
                "threshold_m":          args.threshold_m,
                "timestamp":            time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            row.update({f"R@{n}": round(v, 4) for n, v in recall.items()})
            all_results.append(row)

            # Per-query rows
            for pq in per_query:
                all_per_query.append({
                    "dataset": dataset_name,
                    "method":  method_name,
                    "metric":  metric,
                    **pq,
                })

    if not all_results:
        log.warning("Nessun risultato prodotto.")
        return

    # --- Salvataggio ---
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "recall_table.csv", index=False)

    with open(RESULTS_DIR / "knn_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    df_pq = pd.DataFrame(all_per_query)
    df_pq.to_csv(RESULTS_DIR / "knn_per_query.csv", index=False)

    log.info(f"\nFile salvati in {RESULTS_DIR.relative_to(ROOT)}:")
    log.info("  recall_table.csv, knn_results.json, knn_per_query.csv")
    log.info(f"  predictions/  ({len(list((RESULTS_DIR/'predictions').glob('*.npy')))} file .npy)")

    # Stampa tabella riassuntiva
    recall_cols = [f"R@{n}" for n in args.recall_values if f"R@{n}" in df.columns]
    print("\n" + "="*70)
    print("RECALL@N SUMMARY")
    print("="*70)
    print(df[["dataset", "method", "metric"] + recall_cols + ["knn_time_ms_per_query"]].to_string(index=False))

    # Confronto L2 vs Dot
    if set(args.metrics) == {"l2", "dot"} and "R@1" in df.columns:
        print("\nCONFRONTO L2 vs DOT (R@1)")
        print("-"*50)
        for (ds, m), grp in df.groupby(["dataset", "method"]):
            l2  = grp.loc[grp.metric == "l2",  "R@1"].values
            dot = grp.loc[grp.metric == "dot", "R@1"].values
            if l2.size and dot.size:
                diff   = dot[0] - l2[0]
                winner = "DOT" if diff > 0 else ("L2" if diff < 0 else "PARI")
                print(f"  {ds:20s} | {m:12s} | L2={l2[0]:.2f}%  DOT={dot[0]:.2f}%  → {winner} (Δ{abs(diff):.2f}%)")


if __name__ == "__main__":
    main()