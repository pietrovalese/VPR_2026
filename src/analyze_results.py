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
# Data loading
# ---------------------------------------------------------------------------
def load_knn_results() -> pd.DataFrame | None:
    """Load the Recall@N table produced by knn_evaluation.py."""
    p = RESULTS_DIR / "recall_table.csv"
    if not p.exists():
        log.warning(f"  recall_table.csv not found in {RESULTS_DIR}")
        return None
    return pd.read_csv(p)


def load_knn_per_query() -> pd.DataFrame | None:
    """Load the per-query correctness breakdown produced by knn_evaluation.py."""
    p = RESULTS_DIR / "knn_per_query.csv"
    if not p.exists():
        log.warning(f"  knn_per_query.csv not found")
        return None
    return pd.read_csv(p)


def load_extraction_metrics() -> pd.DataFrame | None:
    """Load descriptor extraction timing/stats produced by extract_descriptors.py."""
    p = RESULTS_DIR / "extraction_metrics.csv"
    if not p.exists():
        log.warning(f"  extraction_metrics.csv not found")
        return None
    return pd.read_csv(p)


def load_matching_summary() -> pd.DataFrame | None:
    """Load the image-matching (re-ranking) summary produced by image_matching_evaluation.py."""
    p = RESULTS_DIR / "matching_summary.csv"
    if not p.exists():
        log.warning(f"  matching_summary.csv not found")
        return None
    return pd.read_csv(p)


def load_matching_json_results() -> list[dict]:
    """Load every results.json from the matching/ subfolders."""
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
# Main report table (Table 1 style, as in the PDF)
# ---------------------------------------------------------------------------
def build_report_table(
    knn_df: pd.DataFrame | None,
    matching_df: pd.DataFrame | None,
    recall_values: list[int],
) -> pd.DataFrame:
    """
    Build a table shaped like:
        VPR Method | [Matcher] | Dataset1 R@1/R@5/R@10 | Dataset2 ...
    """
    rows = []

    # Retrieval-only rows (no matcher)
    if knn_df is not None:
        # Use only the best metric (dot, as instructed by the project)
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

    # Rows with re-ranking
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
    """Pick the metric (l2/dot) with the highest average R@1."""
    r1_col = f"R@{recall_values[0]}" if recall_values else "R@1"
    if r1_col not in knn_df.columns:
        return "dot"
    means = knn_df.groupby("metric")[r1_col].mean()
    return str(means.idxmax())


# ---------------------------------------------------------------------------
# Inliers <-> correctness correlation analysis
# ---------------------------------------------------------------------------
def analyze_inlier_correlation(recall_values: list[int]) -> pd.DataFrame:
    """
    For every dataset x vpr_method x matcher combination:
      - Load per_query_inliers.npy and correct_mask.npy
      - Compute point-biserial correlation, distribution separation,
        and the optimal accuracy-maximizing threshold.
    Also saves the .npy histogram arrays.
    """
    rows = []

    if not MATCHING_DIR.exists():
        log.warning("  matching/ folder not found.")
        return pd.DataFrame()

    for combo_dir in sorted(MATCHING_DIR.iterdir()):
        inliers_file = combo_dir / "per_query_inliers.npy"
        correct_file = combo_dir / "correct_mask.npy"

        if not inliers_file.exists() or not correct_file.exists():
            continue

        inliers = np.load(inliers_file)        # (N_q,)
        correct = np.load(correct_file)        # (N_q,) bool

        tag     = combo_dir.name               # dataset_method_matcher
        parts   = tag.split("_", maxsplit=3)   # naive split, but works for the names used here

        inl_correct   = inliers[correct]
        inl_incorrect = inliers[~correct]

        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(ANALYSIS_DIR / f"inliers_correct_{tag}.npy",   inl_correct)
        np.save(ANALYSIS_DIR / f"inliers_incorrect_{tag}.npy", inl_incorrect)

        # Point-biserial correlation: measures the association between inliers and correctness
        if len(inl_correct) > 1 and len(inl_incorrect) > 1:
            corr, pval = scipy_stats.pointbiserialr(correct.astype(float), inliers.astype(float))
        else:
            corr, pval = float("nan"), float("nan")

        # Optimal threshold (maximizes binary accuracy: inliers > t -> predict "correct")
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
            f"thresh={best_thresh} -> acc={best_acc*100:.1f}%  "
            f"mean(correct)={row['inliers_mean_correct']}  "
            f"mean(incorrect)={row['inliers_mean_incorrect']}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unified timing summary
# ---------------------------------------------------------------------------
def build_timing_summary(
    extraction_df: pd.DataFrame | None,
    knn_df: pd.DataFrame | None,
    matching_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge extraction, KNN, and matching timings into a single per-item timing table."""
    rows = []

    # Descriptor extraction
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
# L2 vs Dot comparison
# ---------------------------------------------------------------------------
def build_l2_vs_dot(knn_df: pd.DataFrame | None, recall_values: list[int]) -> pd.DataFrame:
    """Compare L2 vs dot-product retrieval, per dataset/method, on Recall@N and KNN time."""
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
    For each (dataset, vpr_method):
      - Retrieval-only recall
      - Recall with each matcher
      - Recall gain vs time cost
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
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--recall_values", nargs="+", type=int, default=[1, 5, 10])
    return p.parse_args()


def main():
    """Load every pipeline output and generate the report tables, correlation analysis and summary JSON."""
    args = parse_args()
    rv   = args.recall_values
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading data...")
    knn_df        = load_knn_results()
    per_query_df  = load_knn_per_query()
    extraction_df = load_extraction_metrics()
    matching_df   = load_matching_summary()

    # --- 1. Report table (paper style) ---
    log.info("\n[1/6] Main report table...")
    report_df = build_report_table(knn_df, matching_df, rv)
    if not report_df.empty:
        report_df.to_csv(ANALYSIS_DIR / "report_table.csv", index=False)
        log.info(f"  Saved report_table.csv ({len(report_df)} rows)")
        print("\n" + "="*70)
        print("REPORT TABLE (Table 1 style)")
        print("="*70)
        print(report_df.to_string(index=False))

    # --- 2. Inliers <-> correctness correlation ---
    log.info("\n[2/6] Inlier correlation analysis...")
    inlier_corr_df = analyze_inlier_correlation(rv)
    if not inlier_corr_df.empty:
        inlier_corr_df.to_csv(ANALYSIS_DIR / "inlier_correlation.csv", index=False)
        log.info(f"  Saved inlier_correlation.csv ({len(inlier_corr_df)} rows)")
        print("\n" + "="*70)
        print("INLIERS <-> R@1 CORRECTNESS CORRELATION")
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
        log.info(f"  Saved timing_summary.csv ({len(timing_df)} rows)")
        print("\n" + "="*70)
        print("TIMING SUMMARY (ms per item)")
        print("="*70)
        cols_show = ["stage", "method", "dataset", "matcher", "n_items", "time_ms_per_item"]
        cols_show = [c for c in cols_show if c in timing_df.columns]
        print(timing_df[cols_show].to_string(index=False))

    # --- 4. L2 vs Dot ---
    log.info("\n[4/6] L2 vs Dot comparison...")
    l2_dot_df = build_l2_vs_dot(knn_df, rv)
    if not l2_dot_df.empty:
        l2_dot_df.to_csv(ANALYSIS_DIR / "l2_vs_dot.csv", index=False)
        log.info(f"  Saved l2_vs_dot.csv ({len(l2_dot_df)} rows)")
        print("\n" + "="*70)
        print("L2 vs DOT PRODUCT")
        print("="*70)
        print(l2_dot_df.to_string(index=False))

    # --- 5. Performance/efficiency trade-off ---
    log.info("\n[5/6] Performance/efficiency trade-off...")
    tradeoff_df = build_tradeoff_table(knn_df, matching_df, rv)
    if not tradeoff_df.empty:
        tradeoff_df.to_csv(ANALYSIS_DIR / "tradeoff_table.csv", index=False)
        log.info(f"  Saved tradeoff_table.csv ({len(tradeoff_df)} rows)")
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
    log.info("  Saved full_summary.json")

    log.info(f"\nAll analysis files -> {ANALYSIS_DIR.relative_to(ROOT)}/")
    log.info("Files produced:")
    for p in sorted(ANALYSIS_DIR.iterdir()):
        log.info(f"  {p.name}")


if __name__ == "__main__":
    main()