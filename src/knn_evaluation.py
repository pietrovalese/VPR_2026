"""
src/knn_evaluation.py

Carica i descrittori da logs/descriptors/, esegue KNN con L2 e dot product
e calcola Recall@N per ogni combinazione metodo × dataset × metrica.

Le coordinate GPS sono estratte dai nomi file nel formato:
    @UTM_easting@UTM_northing@...

Output in logs/results/:
    recall_table.csv        — tabella riassuntiva
    knn_results.json        — tutti i risultati machine-readable
    predictions/            — indici KNN (--save_predictions)
        <dataset>_<method>_<metric>_preds.npy

Uso (dalla root del progetto):
    python src/knn_evaluation.py
    python src/knn_evaluation.py --methods cosplace --recall_values 1 5 10 20
    python src/knn_evaluation.py --save_predictions
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
DESC_DIR     = ROOT / "logs" / "descriptors"
RESULTS_DIR  = ROOT / "logs" / "results"

GPS_THRESHOLD_M = 25.0  # soglia standard del progetto


# ---------------------------------------------------------------------------
# Parsing coordinate GPS dai nomi file
# Formato atteso: qualcosa@easting@northing@...
# ---------------------------------------------------------------------------
def parse_utm_from_path(path: str) -> tuple[float, float] | None:
    stem   = Path(path).stem
    parts  = stem.split("@")
    nums   = []
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
    """Ritorna array (N, 2) con [easting, northing]. NaN se parsing fallisce."""
    coords = []
    for p in paths:
        r = parse_utm_from_path(str(p))
        coords.append(r if r is not None else (float("nan"), float("nan")))
    return np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# KNN con FAISS
# ---------------------------------------------------------------------------
def knn_l2(db_desc: np.ndarray, q_desc: np.ndarray, k: int) -> np.ndarray:
    """Ritorna indici (N_q, k) ordinati per distanza L2 crescente."""
    index = faiss.IndexFlatL2(db_desc.shape[1])
    index.add(db_desc.astype(np.float32))
    _, indices = index.search(q_desc.astype(np.float32), k)
    return indices


def knn_dot(db_desc: np.ndarray, q_desc: np.ndarray, k: int) -> np.ndarray:
    """
    Ritorna indici (N_q, k) ordinati per dot product decrescente.
    Con vettori L2-normalizzati equivale a similarità coseno.
    """
    index = faiss.IndexFlatIP(db_desc.shape[1])
    index.add(db_desc.astype(np.float32))
    _, indices = index.search(q_desc.astype(np.float32), k)
    return indices


# ---------------------------------------------------------------------------
# Recall@N
# ---------------------------------------------------------------------------
def compute_recall(
    predictions: np.ndarray,   # (N_q, K)
    q_coords: np.ndarray,      # (N_q, 2)
    db_coords: np.ndarray,     # (N_db, 2)
    recall_values: list[int],
    threshold_m: float = GPS_THRESHOLD_M,
) -> dict[int, float]:
    N_q = len(q_coords)
    recall = {}
    for n in recall_values:
        correct = 0
        for i in range(N_q):
            qc = q_coords[i]
            if np.isnan(qc).any():
                continue
            top_n = predictions[i, :n]
            dists = np.linalg.norm(db_coords[top_n] - qc, axis=1)
            if np.any(dists <= threshold_m):
                correct += 1
        recall[n] = 100.0 * correct / N_q if N_q > 0 else 0.0
    return recall


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_descriptors(desc_dir: Path):
    """Ritorna (db_desc, q_desc, db_paths, q_paths)."""
    return (
        np.load(desc_dir / "database_descriptors.npy"),
        np.load(desc_dir / "query_descriptors.npy"),
        np.load(desc_dir / "database_paths.npy"),
        np.load(desc_dir / "query_paths.npy"),
    )


def descriptors_available(d: Path) -> bool:
    return all((d / f).exists() for f in [
        "database_descriptors.npy", "query_descriptors.npy",
        "database_paths.npy", "query_paths.npy",
    ])


def discover_combinations(
    datasets_filter: list[str] | None,
    methods_filter: list[str] | None,
) -> list[tuple[str, str, Path]]:
    """Scansiona logs/descriptors/ e ritorna tutte le (dataset, method, path) disponibili."""
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
    p.add_argument("--datasets",        nargs="+", default=None)
    p.add_argument("--methods",         nargs="+", default=None)
    p.add_argument("--metrics",         nargs="+", default=["l2", "dot"],
                   choices=["l2", "dot"])
    p.add_argument("--k",               type=int,  default=20)
    p.add_argument("--recall_values",   nargs="+", type=int, default=[1, 5, 10, 20])
    p.add_argument("--threshold_m",     type=float, default=GPS_THRESHOLD_M)
    p.add_argument("--save_predictions", action="store_true",
                   help="Salva indici KNN (utili per il re-ranking)")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_dir = RESULTS_DIR / "predictions"
    if args.save_predictions:
        preds_dir.mkdir(parents=True, exist_ok=True)

    combos = discover_combinations(args.datasets, args.methods)
    if not combos:
        log.error(
            f"Nessun descrittore trovato in {DESC_DIR}.\n"
            "Esegui prima: python src/extract_descriptors.py"
        )
        return

    log.info(f"Root    : {ROOT}")
    log.info(f"Combos  : {len(combos)} (dataset × metodo)")
    log.info(f"Metriche: {args.metrics}  K={args.k}  "
             f"Recall@{args.recall_values}  soglia={args.threshold_m}m\n")

    all_results = []

    for dataset_name, method_name, desc_dir in combos:
        log.info(f"{'='*60}")
        log.info(f"Dataset: {dataset_name}  |  Metodo: {method_name}")

        db_desc, q_desc, db_paths, q_paths = load_descriptors(desc_dir)
        log.info(f"  db: {db_desc.shape}  query: {q_desc.shape}")

        db_coords = extract_utm_coords(db_paths)
        q_coords  = extract_utm_coords(q_paths)
        valid_q   = int(np.sum(~np.isnan(q_coords[:, 0])))
        log.info(f"  Query con GPS valido: {valid_q}/{len(q_coords)}")

        for metric in args.metrics:
            log.info(f"\n  [{metric.upper()}]")
            t0 = time.time()

            preds = knn_l2(db_desc, q_desc, args.k) if metric == "l2" \
                    else knn_dot(db_desc, q_desc, args.k)

            elapsed = time.time() - t0

            if args.save_predictions:
                fname = preds_dir / f"{dataset_name}_{method_name}_{metric}_preds.npy"
                np.save(fname, preds)

            recall = compute_recall(
                preds, q_coords, db_coords,
                recall_values=args.recall_values,
                threshold_m=args.threshold_m,
            )

            recall_str = "  ".join(f"R@{n}={v:.2f}%" for n, v in recall.items())
            log.info(f"  {recall_str}  ({elapsed:.3f}s)")

            row = {
                "dataset": dataset_name,
                "method":  method_name,
                "metric":  metric,
                "k":       args.k,
                "time_s":  round(elapsed, 3),
            }
            row.update({f"R@{n}": round(v, 2) for n, v in recall.items()})
            all_results.append(row)

    if not all_results:
        log.warning("Nessun risultato prodotto.")
        return

    # Salvataggio
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "recall_table.csv", index=False)
    with open(RESULTS_DIR / "knn_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nRisultati salvati in {RESULTS_DIR.relative_to(ROOT)}")

    # Stampa tabella riassuntiva
    recall_cols = [f"R@{n}" for n in args.recall_values if f"R@{n}" in df.columns]
    print("\n" + "="*70)
    print("RECALL@N")
    print("="*70)
    print(df[["dataset", "method", "metric"] + recall_cols].to_string(index=False))

    # Confronto L2 vs Dot su R@1
    if set(args.metrics) == {"l2", "dot"} and "R@1" in df.columns:
        print("\nCONFRONTO L2 vs DOT (R@1)")
        print("-"*50)
        for (ds, m), grp in df.groupby(["dataset", "method"]):
            l2  = grp.loc[grp.metric == "l2",  "R@1"].values
            dot = grp.loc[grp.metric == "dot", "R@1"].values
            if l2.size and dot.size:
                diff   = dot[0] - l2[0]
                winner = "DOT" if diff > 0 else ("L2" if diff < 0 else "PARI")
                print(f"  {ds:20s} | {m:10s} | "
                      f"L2={l2[0]:.2f}%  DOT={dot[0]:.2f}%  → {winner} "
                      f"(Δ{abs(diff):.2f}%)")


if __name__ == "__main__":
    main()