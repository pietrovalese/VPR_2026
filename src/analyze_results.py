"""
src/analyze_results.py

Legge tutti i CSV/JSON prodotti dalla pipeline e genera:
  - Tabelle riassuntive formattate per il report (stile paper)
  - Analisi della correlazione inliers ↔ correttezza R@1
  - Statistiche sui tempi (VPR extraction + KNN + matching)
  - Array .npy degli istogrammi inliers (corretto vs sbagliato)
  - Confronto L2 vs dot product
  - Performance/efficiency trade-off summary

Output in logs/analysis/:
    report_table.csv            — Tabella 1 del PDF (metodo + matcher + recall)
    timing_summary.csv          — tutti i tempi in un unico posto
    inlier_correlation.csv      — correlazione inliers/correttezza per dataset×metodo×matcher
    inliers_correct_<tag>.npy   — distribuzione inliers per query corrette
    inliers_incorrect_<tag>.npy — distribuzione inliers per query errate
    l2_vs_dot.csv               — confronto L2 vs dot product
    full_summary.json           — tutto in uno, machine-readable

Uso (dalla root del progetto):
    python src/analyze_results.py
    python src/analyze_results.py --recall_values 1 5 10
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent
RESULTS_DIR  = ROOT / "logs" / "results"
MATCHING_DIR = RESULTS_DIR / "matching"
ANALYSIS_DIR = ROOT / "logs" / "analysis"


# ---------------------------------------------------------------------------
# Caricamento dati
# ---------------------------------------------------------------------------
def load_knn_results() -> pd.DataFrame | None:
    p = RESULTS_DIR / "recall_table.csv"
    if not p.exists():
        log.warning(f"  recall_table.csv non trovato in {RESULTS_DIR}")
        return None
    return pd.read_csv(p)


def load_knn_per_query() -> pd.DataFrame | None:
    p = RESULTS_DIR / "knn_per_query.csv"
    if not p.exists():
        log.warning(f"  knn_per_query.csv non trovato")
        return None
    return pd.read_csv(p)


def load_extraction_metrics() -> pd.DataFrame | None:
    p = RESULTS_DIR / "extraction_metrics.csv"
    if not p.exists():
        log.warning(f"  extraction_metrics.csv non trovato")
        return None
    return pd.read_csv(p)


def load_matching_summary() -> pd.DataFrame | None:
    p = RESULTS_DIR / "matching_summary.csv"
    if not p.exists():
        log.warning(f"  matching_summary.csv non trovato")
        return None
    return pd.read_csv(p)


def load_matching_json_results() -> list[dict]:
    """Carica tutti i results.json dalle cartelle matching/."""
    rows = []
    if not MATCHING_DIR.exists():
        return rows
    for d in sorted(MATCHING_DIR.iterdir()):
        p = d / "results.json"
        if p.exists():
            with open(p) as f:
                rows.append(json.load(f))
    return rows


# ---------------------------------------------------------------------------
# Tabella report principale (stile Tabella 1 del PDF)
# ---------------------------------------------------------------------------
def build_report_table(
    knn_df: pd.DataFrame | None,
    matching_df: pd.DataFrame | None,
    recall_values: list[int],
) -> pd.DataFrame:
    """
    Costruisce la tabella tipo:
        VPR Method | [Matcher] | Dataset1 R@1/R@5/R@10 | Dataset2 ...
    """
    rows = []

    # Righe solo-retrieval (nessun matcher)
    if knn_df is not None:
        # Usa solo la metrica migliore (dot, come da istruzioni del progetto)
        best_metric = _get_best_metric(knn_df, recall_values)
        subset = knn_df[knn_df["metric"] == best_metric]
        for _, r in subset.iterrows():
            row = {
                "vpr_method": r["method"],
                "matcher":    "—",
                "dataset":    r["dataset"],
                "metric":     r["metric"],
            }
            for n in recall_values:
                col = f"R@{n}"
                row[col] = r[col] if col in r.index else None
            rows.append(row)

    # Righe con re-ranking
    if matching_df is not None:
        for _, r in matching_df.iterrows():
            row = {
                "vpr_method": r["vpr_method"],
                "matcher":    r["matcher"],
                "dataset":    r["dataset"],
                "metric":     r.get("metric", "dot"),
            }
            for n in recall_values:
                row[f"R@{n}"] = r.get(f"R@{n}_after")
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["dataset", "vpr_method", "matcher"])


def _get_best_metric(knn_df: pd.DataFrame, recall_values: list[int]) -> str:
    """Sceglie la metrica (l2/dot) con R@1 medio più alto."""
    r1_col = f"R@{recall_values[0]}" if recall_values else "R@1"
    if r1_col not in knn_df.columns:
        return "dot"
    means = knn_df.groupby("metric")[r1_col].mean()
    return str(means.idxmax())


# ---------------------------------------------------------------------------
# Analisi correlazione inliers ↔ correttezza
# ---------------------------------------------------------------------------
def analyze_inlier_correlation(recall_values: list[int]) -> pd.DataFrame:
    """
    Per ogni combinazione dataset×vpr_method×matcher:
      - Carica per_query_inliers.npy e correct_mask.npy
      - Calcola correlazione punto-biseriale, separazione distribuzioni,
        threshold ottimale (massimizza accuracy)
    Salva anche gli array .npy degli istogrammi.
    """
    rows = []

    if not MATCHING_DIR.exists():
        log.warning("  Cartella matching/ non trovata.")
        return pd.DataFrame()

    for combo_dir in sorted(MATCHING_DIR.iterdir()):
        inliers_file = combo_dir / "per_query_inliers.npy"
        correct_file = combo_dir / "correct_mask.npy"

        if not inliers_file.exists() or not correct_file.exists():
            continue

        inliers = np.load(inliers_file)        # (N_q,)
        correct = np.load(correct_file)        # (N_q,) bool

        tag     = combo_dir.name               # dataset_method_matcher
        parts   = tag.split("_", maxsplit=3)   # fragile ma funziona con i nomi usati

        # Statistiche distribuzioni
        inl_correct   = inliers[correct]
        inl_incorrect = inliers[~correct]

        # Salva array per istogrammi
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(ANALYSIS_DIR / f"inliers_correct_{tag}.npy",   inl_correct)
        np.save(ANALYSIS_DIR / f"inliers_incorrect_{tag}.npy", inl_incorrect)

        # Correlazione punto-biseriale: misura associazione inliers ↔ correttezza
        if len(inl_correct) > 1 and len(inl_incorrect) > 1:
            corr, pval = scipy_stats.pointbiserialr(correct.astype(float), inliers.astype(float))
        else:
            corr, pval = float("nan"), float("nan")

        # Threshold ottimale (massimizza accuracy binaria: inliers > t → predico "corretto")
        best_acc, best_thresh = 0.0, 0
        for t in np.unique(inliers):
            pred     = (inliers > t).astype(int)
            acc      = float((pred == correct.astype(int)).mean())
            if acc > best_acc:
                best_acc, best_thresh = acc, int(t)

        row = {
            "combo":                   tag,
            "n_queries":               len(inliers),
            "n_correct":               int(correct.sum()),
            "n_incorrect":             int((~correct).sum()),
            "inliers_mean_correct":    round(float(inl_correct.mean()), 3)   if len(inl_correct) > 0 else None,
            "inliers_mean_incorrect":  round(float(inl_incorrect.mean()), 3) if len(inl_incorrect) > 0 else None,
            "inliers_std_correct":     round(float(inl_correct.std()), 3)    if len(inl_correct) > 1 else None,
            "inliers_std_incorrect":   round(float(inl_incorrect.std()), 3)  if len(inl_incorrect) > 1 else None,
            "inliers_median_correct":  round(float(np.median(inl_correct)), 3)   if len(inl_correct) > 0 else None,
            "inliers_median_incorrect":round(float(np.median(inl_incorrect)), 3) if len(inl_incorrect) > 0 else None,
            "pointbiserial_corr":      round(float(corr), 4),
            "pointbiserial_pval":      round(float(pval), 6),
            "optimal_threshold":       best_thresh,
            "optimal_threshold_acc":   round(best_acc * 100, 2),
        }
        rows.append(row)

        log.info(
            f"  {tag}: corr={corr:.3f} (p={pval:.4f})  "
            f"thresh={best_thresh} → acc={best_acc*100:.1f}%  "
            f"mean(correct)={row['inliers_mean_correct']}  "
            f"mean(incorrect)={row['inliers_mean_incorrect']}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Timing summary unificato
# ---------------------------------------------------------------------------
def build_timing_summary(
    extraction_df: pd.DataFrame | None,
    knn_df: pd.DataFrame | None,
    matching_df: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []

    # Estrazione descrittori
    if extraction_df is not None:
        for _, r in extraction_df.iterrows():
            rows.append({
                "stage":   "extraction",
                "method":  r.get("method"),
                "dataset": r.get("dataset"),
                "split":   r.get("split"),
                "matcher": None,
                "n_items": r.get("n_images"),
                "time_s":  r.get("time_s"),
                "throughput_img_s": r.get("throughput_img_s"),
                "time_ms_per_item": round(1000 * r["time_s"] / r["n_images"], 4) if r.get("n_images", 0) > 0 else None,
            })

    # KNN
    if knn_df is not None:
        for _, r in knn_df.iterrows():
            rows.append({
                "stage":   "knn",
                "method":  r.get("method"),
                "dataset": r.get("dataset"),
                "split":   "query",
                "matcher": r.get("metric"),
                "n_items": r.get("n_queries"),
                "time_s":  r.get("knn_time_s"),
                "throughput_img_s": None,
                "time_ms_per_item": r.get("knn_time_ms_per_query"),
            })

    # Matching
    if matching_df is not None:
        for _, r in matching_df.iterrows():
            rows.append({
                "stage":   "matching",
                "method":  r.get("vpr_method"),
                "dataset": r.get("dataset"),
                "split":   "query",
                "matcher": r.get("matcher"),
                "n_items": r.get("n_queries"),
                "time_s":  r.get("total_time_s"),
                "throughput_img_s": None,
                "time_ms_per_item": round(r["avg_time_per_query_s"] * 1000, 2) if "avg_time_per_query_s" in r.index else None,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Confronto L2 vs Dot
# ---------------------------------------------------------------------------
def build_l2_vs_dot(knn_df: pd.DataFrame | None, recall_values: list[int]) -> pd.DataFrame:
    if knn_df is None or "metric" not in knn_df.columns:
        return pd.DataFrame()

    rows = []
    for (ds, m), grp in knn_df.groupby(["dataset", "method"]):
        l2  = grp[grp.metric == "l2"]
        dot = grp[grp.metric == "dot"]
        if l2.empty or dot.empty:
            continue
        row = {"dataset": ds, "method": m}
        for n in recall_values:
            col = f"R@{n}"
            if col in grp.columns:
                l2_val  = float(l2[col].values[0])
                dot_val = float(dot[col].values[0])
                row[f"l2_R@{n}"]    = round(l2_val, 2)
                row[f"dot_R@{n}"]   = round(dot_val, 2)
                row[f"delta_R@{n}"] = round(dot_val - l2_val, 2)
                row[f"winner_R@{n}"] = "dot" if dot_val > l2_val else ("l2" if l2_val > dot_val else "tie")
        # Tempo
        if "knn_time_ms_per_query" in grp.columns:
            row["l2_time_ms"]  = float(l2["knn_time_ms_per_query"].values[0])
            row["dot_time_ms"] = float(dot["knn_time_ms_per_query"].values[0])
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Performance/efficiency trade-off table
# ---------------------------------------------------------------------------
def build_tradeoff_table(
    knn_df: pd.DataFrame | None,
    matching_df: pd.DataFrame | None,
    recall_values: list[int],
) -> pd.DataFrame:
    """
    Per ogni (dataset, vpr_method):
      - Recall solo retrieval
      - Recall + ogni matcher
      - Guadagno di recall vs costo temporale
    """
    if knn_df is None:
        return pd.DataFrame()

    r1_col = f"R@{recall_values[0]}" if recall_values else "R@1"
    rows   = []

    best_metric = _get_best_metric(knn_df, recall_values)
    base_df     = knn_df[knn_df["metric"] == best_metric]

    for (ds, m), grp in base_df.groupby(["dataset", "method"]):
        base_r1   = float(grp[r1_col].values[0]) if r1_col in grp.columns else None
        base_time = float(grp["knn_time_ms_per_query"].values[0]) if "knn_time_ms_per_query" in grp.columns else None

        row = {
            "dataset": ds, "vpr_method": m,
            "matcher": "retrieval_only",
            f"R@{recall_values[0]}": base_r1,
            "time_ms_per_query": base_time,
            "recall_gain": 0.0,
            "time_overhead_ms": 0.0,
        }
        rows.append(row)

        if matching_df is not None:
            sub = matching_df[(matching_df.dataset == ds) & (matching_df.vpr_method == m)]
            for _, mr in sub.iterrows():
                after_col = f"R@{recall_values[0]}_after"
                after_r1  = float(mr[after_col]) if after_col in mr.index else None
                mtime_ms  = float(mr["avg_time_per_query_s"]) * 1000 if "avg_time_per_query_s" in mr.index else None
                rows.append({
                    "dataset": ds, "vpr_method": m,
                    "matcher": mr["matcher"],
                    f"R@{recall_values[0]}": after_r1,
                    "time_ms_per_query": mtime_ms,
                    "recall_gain":      round(after_r1 - base_r1, 2) if (after_r1 is not None and base_r1 is not None) else None,
                    "time_overhead_ms": round(mtime_ms - (base_time or 0), 2) if mtime_ms is not None else None,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--recall_values", nargs="+", type=int, default=[1, 5, 10])
    return p.parse_args()


def main():
    args = parse_args()
    rv   = args.recall_values
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Caricamento dati...")
    knn_df        = load_knn_results()
    per_query_df  = load_knn_per_query()
    extraction_df = load_extraction_metrics()
    matching_df   = load_matching_summary()

    # --- 1. Tabella report (stile paper) ---
    log.info("\n[1/6] Tabella report principale...")
    report_df = build_report_table(knn_df, matching_df, rv)
    if not report_df.empty:
        report_df.to_csv(ANALYSIS_DIR / "report_table.csv", index=False)
        log.info(f"  Salvata report_table.csv ({len(report_df)} righe)")
        print("\n" + "="*70)
        print("TABELLA REPORT (stile Tabella 1 del PDF)")
        print("="*70)
        print(report_df.to_string(index=False))

    # --- 2. Correlazione inliers ↔ correttezza ---
    log.info("\n[2/6] Analisi correlazione inliers...")
    inlier_corr_df = analyze_inlier_correlation(rv)
    if not inlier_corr_df.empty:
        inlier_corr_df.to_csv(ANALYSIS_DIR / "inlier_correlation.csv", index=False)
        log.info(f"  Salvata inlier_correlation.csv ({len(inlier_corr_df)} righe)")
        print("\n" + "="*70)
        print("CORRELAZIONE INLIERS ↔ CORRETTEZZA R@1")
        print("="*70)
        cols_show = ["combo", "pointbiserial_corr", "optimal_threshold",
                     "optimal_threshold_acc", "inliers_mean_correct", "inliers_mean_incorrect"]
        cols_show = [c for c in cols_show if c in inlier_corr_df.columns]
        print(inlier_corr_df[cols_show].to_string(index=False))

    # --- 3. Timing summary ---
    log.info("\n[3/6] Timing summary...")
    timing_df = build_timing_summary(extraction_df, knn_df, matching_df)
    if not timing_df.empty:
        timing_df.to_csv(ANALYSIS_DIR / "timing_summary.csv", index=False)
        log.info(f"  Salvata timing_summary.csv ({len(timing_df)} righe)")
        print("\n" + "="*70)
        print("TIMING SUMMARY (ms per item)")
        print("="*70)
        cols_show = ["stage", "method", "dataset", "matcher", "n_items", "time_ms_per_item"]
        cols_show = [c for c in cols_show if c in timing_df.columns]
        print(timing_df[cols_show].to_string(index=False))

    # --- 4. L2 vs Dot ---
    log.info("\n[4/6] Confronto L2 vs Dot...")
    l2_dot_df = build_l2_vs_dot(knn_df, rv)
    if not l2_dot_df.empty:
        l2_dot_df.to_csv(ANALYSIS_DIR / "l2_vs_dot.csv", index=False)
        log.info(f"  Salvata l2_vs_dot.csv ({len(l2_dot_df)} righe)")
        print("\n" + "="*70)
        print("L2 vs DOT PRODUCT")
        print("="*70)
        print(l2_dot_df.to_string(index=False))

    # --- 5. Performance/efficiency trade-off ---
    log.info("\n[5/6] Trade-off performance/efficienza...")
    tradeoff_df = build_tradeoff_table(knn_df, matching_df, rv)
    if not tradeoff_df.empty:
        tradeoff_df.to_csv(ANALYSIS_DIR / "tradeoff_table.csv", index=False)
        log.info(f"  Salvata tradeoff_table.csv ({len(tradeoff_df)} righe)")
        print("\n" + "="*70)
        print("PERFORMANCE / EFFICIENCY TRADE-OFF")
        print("="*70)
        print(tradeoff_df.to_string(index=False))

    # --- 6. Full summary JSON ---
    log.info("\n[6/6] Full summary JSON...")
    full_summary = {
        "recall_values":       rv,
        "knn_rows":            knn_df.to_dict("records") if knn_df is not None else [],
        "matching_rows":       load_matching_json_results(),
        "inlier_correlation":  inlier_corr_df.to_dict("records") if not inlier_corr_df.empty else [],
        "l2_vs_dot":           l2_dot_df.to_dict("records") if not l2_dot_df.empty else [],
        "tradeoff":            tradeoff_df.to_dict("records") if not tradeoff_df.empty else [],
    }
    with open(ANALYSIS_DIR / "full_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)
    log.info("  Salvato full_summary.json")

    log.info(f"\nTutti i file di analisi → {ANALYSIS_DIR.relative_to(ROOT)}/")
    log.info("File prodotti:")
    for p in sorted(ANALYSIS_DIR.iterdir()):
        log.info(f"  {p.name}")


if __name__ == "__main__":
    main()
