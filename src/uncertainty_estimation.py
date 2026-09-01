import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, r2_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from knn_evaluation import parse_utm_from_path, extract_utm_coords, GPS_THRESHOLD_M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup - same conventions as the other pipeline scripts
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
DESC_DIR      = ROOT / "logs" / "descriptors"
PREDS_DIR     = ROOT / "logs" / "results" / "predictions"
KNN_PQ_CSV    = ROOT / "logs" / "results" / "knn_per_query.csv"
MATCH_DIR     = ROOT / "logs" / "results" / "matching"
REDUCTION_DIR = ROOT / "logs" / "feature_reduction"
OUT_DIR       = ROOT / "logs" / "uncertainty"

TRAIN_DATASETS = ["svox_sun_train", "svox_night_train"]
VAL_DATASET    = "sf_xs_val"
TEST_DATASETS  = ["sf_xs_test", "tokyo_xs", "svox_sun", "svox_night"]


# ---------------------------------------------------------------------------
# Loading "full" features (already computed by knn_evaluation / image_matching)
# ---------------------------------------------------------------------------
def load_full_features(dataset: str, method: str, matcher: str, metric: str) -> pd.DataFrame | None:
    """
    Combine, for one dataset, the results already saved by knn_evaluation.py and
    image_matching_evaluation.py into a single per-query table:
        query_idx, geo_dist_m, correct, top1_score, margin, n_inliers
    Nothing is recomputed: if a required file is missing, returns None with an explanatory log.
    """
    scores_path = PREDS_DIR / f"{dataset}_{method}_{metric}_scores.npy"
    if not scores_path.exists():
        log.warning(f"  [{dataset}/{method}] missing {scores_path.name} - run knn_evaluation.py")
        return None
    if not KNN_PQ_CSV.exists():
        log.warning("  missing knn_per_query.csv - run knn_evaluation.py")
        return None

    scores = np.load(scores_path)  # (N_q, K) - decreasing score if metric='dot'
    if scores.shape[1] < 2:
        log.warning(f"  [{dataset}/{method}] at least 2 neighbors are needed for the margin")
        return None

    pq = pd.read_csv(KNN_PQ_CSV)
    pq = pq[(pq["dataset"] == dataset) & (pq["method"] == method) & (pq["metric"] == metric)].reset_index(drop=True)
    if len(pq) != scores.shape[0]:
        log.warning(f"  [{dataset}/{method}] row mismatch between knn_per_query.csv and scores.npy - skip")
        return None

    match_dir = MATCH_DIR / f"{dataset}_{method}_{matcher}"
    inliers_path = match_dir / "per_query_inliers.npy"
    if not inliers_path.exists():
        log.warning(f"  [{dataset}/{method}/{matcher}] missing per_query_inliers.npy - run image_matching_evaluation.py")
        return None

    n_inliers = np.load(inliers_path)
    if len(n_inliers) != len(pq):
        log.warning(f"  [{dataset}/{method}/{matcher}] row mismatch between inliers and knn - skip")
        return None

    # For metric='dot' a higher score means more similar; for 'l2' a lower score means more
    # similar. Everything is normalized to an "uncertainty direction": higher value = more
    # uncertain.
    if metric == "dot":
        top1_score = scores[:, 0]
        margin     = scores[:, 0] - scores[:, 1]
        l2_dist    = -scores[:, 0]      # proxy: high score -> low "distance"
    else:  # l2: true distance, lower score = better
        top1_score = -scores[:, 0]
        margin     = scores[:, 1] - scores[:, 0]
        l2_dist    = scores[:, 0]

    df = pd.DataFrame({
        "query_idx":     pq["query_idx"].values,
        "geo_dist_m":    pq["top1_geo_dist_m"].values,
        "has_gps":       pq["has_gps"].values,
        "correct":       pq["correct_r1"].astype(bool).values,
        "top1_score":    top1_score,
        "l2_dist_top1":  l2_dist,
        "margin":        margin,
        "n_inliers":     n_inliers[:len(pq)].astype(float),
    })
    return df


# ---------------------------------------------------------------------------
# "Compressed" features - require a small re-KNN on masked descriptors
# ---------------------------------------------------------------------------
def compute_compressed_l2_and_margin(dataset: str, method: str) -> pd.DataFrame | None:
    """
    Recompute, using ONLY the dimensions kept by mask_topk_variance.npy (extension 6.3):
    distance to the 1st neighbor, 1st-2nd margin, and - unlike the previous version - also
    the correctness (R@1, 25m threshold) and the geographic distance of the top-1 actually
    retrieved by the compressed retrieval. Reuses the descriptors already extracted by
    extract_descriptors.py (normalized ones), it does not call the model again.

    Why also recompute "correct"/"geo_dist_m": the retrieved top-1 can change once the
    descriptor is masked (consistent with the non-zero delta R@1 observed in 6.3). If we
    reused the "correct" label from the full retrieval to evaluate an uncertainty signal
    computed on the compressed retrieval, we would effectively be asking "does this
    compressed signal predict the correctness of the FULL system?" - a different (and less
    relevant, for an actually deployed compressed system) question from "does it predict its
    own correctness?". Here we answer the latter.

    Returns a DataFrame with columns
    [l2_dist_top1_comp, margin_comp, correct_comp, geo_dist_m_comp],
    or None if the prerequisites are missing.
    """
    mask_path = REDUCTION_DIR / method / "mask_topk_variance.npy"
    if not mask_path.exists():
        log.warning(f"  [{method}] missing mask_topk_variance.npy - run features_reduction.py first "
                    f"(the updated version that also saves this mask)")
        return None
    mask = np.load(mask_path)

    desc_dir = DESC_DIR / dataset / method
    db_path = desc_dir / "database_descriptors.npy"
    q_path  = desc_dir / "query_descriptors.npy"
    db_paths_path = desc_dir / "database_paths.npy"
    q_paths_path  = desc_dir / "query_paths.npy"
    if not db_path.exists() or not q_path.exists():
        log.warning(f"  [{dataset}/{method}] missing normalized descriptors - run extract_descriptors.py")
        return None
    if not db_paths_path.exists() or not q_paths_path.exists():
        log.warning(f"  [{dataset}/{method}] missing paths (for UTM coordinates) - run extract_descriptors.py")
        return None

    db = np.load(db_path)[:, mask].astype(np.float32)
    q  = np.load(q_path)[:, mask].astype(np.float32)

    # Re-normalize after masking (norms are no longer 1 once dimensions are dropped)
    db_n = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-12)
    q_n  = q  / (np.linalg.norm(q,  axis=1, keepdims=True) + 1e-12)

    index = faiss.IndexFlatIP(db_n.shape[1])
    index.add(db_n)
    scores, indices = index.search(q_n, 2)  # top-2 is enough for the margin

    l2_dist = -scores[:, 0]
    margin  = scores[:, 0] - scores[:, 1]

    # Correctness and geo distance of the RECOMPUTED top-1 (same logic/threshold as
    # knn_evaluation.py::compute_recall_and_per_query, GPS_THRESHOLD_M=25m)
    db_paths = np.load(db_paths_path, allow_pickle=True)
    q_paths  = np.load(q_paths_path, allow_pickle=True)
    db_coords = extract_utm_coords(db_paths)
    q_coords  = extract_utm_coords(q_paths)

    top1_idx = indices[:, 0]
    has_gps  = ~np.isnan(q_coords).any(axis=1)
    geo_dist = np.full(len(q_coords), np.nan)
    geo_dist[has_gps] = np.linalg.norm(db_coords[top1_idx[has_gps]] - q_coords[has_gps], axis=1)
    correct = np.where(has_gps, geo_dist <= GPS_THRESHOLD_M, False)

    return pd.DataFrame({
        "l2_dist_top1_comp": l2_dist,
        "margin_comp":       margin,
        "correct_comp":      correct,
        "geo_dist_m_comp":   geo_dist,
    })


def build_feature_table(dataset: str, method: str, matcher: str, metric: str,
                         skip_compressed: bool = False) -> pd.DataFrame | None:
    """Full per-query table: full columns plus (if available) compressed columns."""
    df = load_full_features(dataset, method, matcher, metric)
    if df is None:
        return None

    if not skip_compressed:
        comp = compute_compressed_l2_and_margin(dataset, method)
        if comp is not None and len(comp) == len(df):
            df["l2_dist_top1_comp"] = comp["l2_dist_top1_comp"].values
            df["margin_comp"]       = comp["margin_comp"].values
            df["correct_comp"]      = comp["correct_comp"].values
            df["geo_dist_m_comp"]   = comp["geo_dist_m_comp"].values
        else:
            log.warning(f"  [{dataset}/{method}] compressed features not available for this run")

    return df


# ---------------------------------------------------------------------------
# Uncertainty evaluation metrics
# ---------------------------------------------------------------------------
def evaluate_uncertainty_score(unc_score: np.ndarray, correct: np.ndarray,
                                geo_dist_m: np.ndarray, has_gps: np.ndarray) -> dict:
    """
    unc_score: higher value = more uncertain (already oriented by the caller).
    Returns AUPRC (positive class = wrong query), Spearman's rho vs the continuous geo
    error, R^2 of a linear fit score->error, and AUSC.
    """
    unc_score = np.asarray(unc_score, dtype=np.float64)
    is_wrong  = ~np.asarray(correct, dtype=bool)

    out = {}

    # AUPRC: positive = "wrong query", score = uncertainty
    if is_wrong.any() and (~is_wrong).any():
        out["auprc"] = float(average_precision_score(is_wrong, unc_score))
    else:
        out["auprc"] = float("nan")

    # Spearman and R^2 only on queries with valid GPS
    mask_gps = np.asarray(has_gps, dtype=bool) & ~np.isnan(geo_dist_m)
    if mask_gps.sum() >= 10:
        rho, pval = scipy_stats.spearmanr(unc_score[mask_gps], geo_dist_m[mask_gps])
        out["spearman_rho"] = float(rho)
        out["spearman_p"]   = float(pval)

        lr = LinearRegression().fit(unc_score[mask_gps].reshape(-1, 1), geo_dist_m[mask_gps])
        pred = lr.predict(unc_score[mask_gps].reshape(-1, 1))
        out["r2"] = float(r2_score(geo_dist_m[mask_gps], pred))
    else:
        out["spearman_rho"] = out["spearman_p"] = out["r2"] = float("nan")

    # AUSC: progressively drop the most uncertain queries, look at the mean residual error
    out["ausc"] = area_under_sparsification_curve(unc_score, geo_dist_m, has_gps)

    return out


def area_under_sparsification_curve(unc_score: np.ndarray, geo_dist_m: np.ndarray,
                                      has_gps: np.ndarray, n_steps: int = 20) -> float:
    """
    Sort queries by decreasing uncertainty and drop them in slices (0%, 5%, ..., 95%). At
    each step compute the mean error (in meters) over the remaining queries. AUSC = area
    under the mean-error vs fraction-removed curve (trapezoidal rule, x axis in [0,1]). A
    lower value means uncertainty is good at flagging high-error queries (removing them
    quickly lowers the residual mean error).
    """
    mask_gps = np.asarray(has_gps, dtype=bool) & ~np.isnan(geo_dist_m)
    if mask_gps.sum() < n_steps:
        return float("nan")

    score = np.asarray(unc_score)[mask_gps]
    err   = np.asarray(geo_dist_m)[mask_gps]
    order = np.argsort(-score)  # most uncertain first
    err_sorted = err[order]

    fractions = np.linspace(0, 0.95, n_steps)
    N = len(err_sorted)
    mean_errs = []
    for f in fractions:
        n_removed = int(f * N)
        remaining = err_sorted[n_removed:]
        mean_errs.append(remaining.mean() if len(remaining) > 0 else 0.0)

    return float(np.trapz(mean_errs, fractions) / (fractions[-1] - fractions[0]))


# ---------------------------------------------------------------------------
# Logistic regressor (trained on SVOX train, validated on sf_xs_val)
# ---------------------------------------------------------------------------
def train_logreg(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str],
                  label_col: str = "correct") -> tuple:
    """
    Train a LogisticRegression on train_df, selecting C for the best AUPRC on val_df. The
    features live on very different scales (n_inliers: tens-hundreds; l2_dist/margin: range
    ~[-1,1]): without standardization the L2 penalty would disproportionately penalize the
    small-scale features, effectively collapsing the model onto n_inliers alone. The scaler
    is fit ONLY on the training set and then reused (never refit) on val/test.
    label_col: correctness column to use as the target - "correct" (full) or "correct_comp"
    (compressed, recomputed on the masked retrieval).
    Returns (chosen_model, scaler, best_C, val_auprc).
    """
    train_df = train_df.dropna(subset=feature_cols + [label_col])
    val_df   = val_df.dropna(subset=feature_cols + [label_col])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df[label_col].astype(int).values
    X_val   = scaler.transform(val_df[feature_cols].values)
    y_val   = val_df[label_col].astype(int).values

    best_model, best_c, best_auprc = None, None, -1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        p_wrong = 1.0 - model.predict_proba(X_val)[:, 1]  # probability of being wrong
        if (~y_val.astype(bool)).any() and y_val.astype(bool).any():
            auprc = average_precision_score(~y_val.astype(bool), p_wrong)
        else:
            auprc = float("nan")
        if not np.isnan(auprc) and auprc > best_auprc:
            best_model, best_c, best_auprc = model, C, auprc

    return best_model, scaler, best_c, best_auprc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--methods",  nargs="+", default=["cosplace", "megaloc"])
    p.add_argument("--matchers", nargs="+", default=["superglue", "loftr"])
    p.add_argument("--metric",   type=str, default="dot", choices=["l2", "dot"])
    p.add_argument("--skip_compressed", action="store_true",
                    help="Skip the comparison against compressed descriptors (6.3)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    """Train and evaluate every uncertainty-estimation variant for each method/matcher combination."""
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for method in args.methods:
        for matcher in args.matchers:
            log.info(f"\n{'='*70}\n[COMBO] method={method}  matcher={matcher}\n{'='*70}")
            run_dir = OUT_DIR / f"{method}_{matcher}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # --- Train (SVOX train, sun+night concatenated) ---
            train_frames = []
            for ds in TRAIN_DATASETS:
                df = build_feature_table(ds, method, matcher, args.metric, args.skip_compressed)
                if df is not None:
                    df["train_source"] = ds
                    train_frames.append(df)
            if not train_frames:
                log.warning(f"  No training data available for {method}/{matcher} - skip combo")
                continue
            train_df = pd.concat(train_frames, ignore_index=True)
            train_df.to_csv(run_dir / "train_features.csv", index=False)

            # --- Val (sf_xs_val) ---
            val_df = build_feature_table(VAL_DATASET, method, matcher, args.metric, args.skip_compressed)
            if val_df is None:
                log.warning(f"  Validation set not available for {method}/{matcher} - skip combo")
                continue
            val_df.to_csv(run_dir / "val_features.csv", index=False)

            # --- Train the "full" logreg ---
            feat_full = ["n_inliers", "l2_dist_top1", "margin"]
            model_full, scaler_full, c_full, auprc_full = train_logreg(
                train_df, val_df, feat_full, label_col="correct")
            log.info(f"  logreg FULL: best C={c_full}  val AUPRC={auprc_full:.4f}")

            # --- Train the "compressed" logreg, if the columns exist ---
            # NOTE: n_inliers is deliberately kept identical between full/compressed (it does
            # not depend on the descriptor). This way the full vs compressed comparison
            # isolates ONLY the effect of compression on the descriptor-derived features,
            # instead of conflating "compression hurts" with "removing the inliers hurts".
            # The target is "correct_comp" (correctness of the retrieval RECOMPUTED on the
            # masked descriptors), not "correct" (full) - otherwise we would be training the
            # compressed model to predict the errors of a different system.
            model_comp, scaler_comp, c_comp = None, None, None
            has_comp_cols = "l2_dist_top1_comp" in train_df.columns and "margin_comp" in train_df.columns
            if not args.skip_compressed and has_comp_cols:
                feat_comp = ["n_inliers", "l2_dist_top1_comp", "margin_comp"]
                model_comp, scaler_comp, c_comp, auprc_comp = train_logreg(
                    train_df, val_df, feat_comp, label_col="correct_comp")
                log.info(f"  logreg COMPRESSED: best C={c_comp}  val AUPRC={auprc_comp:.4f}")

            # --- Evaluation on the test sets ---
            for ds in TEST_DATASETS:
                test_df = build_feature_table(ds, method, matcher, args.metric, args.skip_compressed)
                if test_df is None:
                    continue
                test_df.to_csv(run_dir / f"test_{ds}_features.csv", index=False)

                # variant_name -> (score, "correct" column to use, "geo_dist_m" column to use)
                # The "full" variants (n_inliers, l2_dist, margin, logreg_full) are evaluated
                # against the ground truth of the full retrieval; the "compressed" variants
                # against the ground truth of the retrieval RECOMPUTED on the masked
                # descriptors.
                variants = {
                    "n_inliers":  (-test_df["n_inliers"].values,        "correct", "geo_dist_m"),
                    "l2_dist":    ( test_df["l2_dist_top1"].values,     "correct", "geo_dist_m"),
                    "margin":     (-test_df["margin"].values,           "correct", "geo_dist_m"),
                }

                if model_full is not None:
                    Xf = test_df[feat_full].fillna(test_df[feat_full].mean())
                    Xf_scaled = scaler_full.transform(Xf.values)
                    variants["logreg_full"] = (1.0 - model_full.predict_proba(Xf_scaled)[:, 1],
                                                "correct", "geo_dist_m")

                if not args.skip_compressed and "l2_dist_top1_comp" in test_df.columns:
                    variants["l2_dist_compressed"] = (test_df["l2_dist_top1_comp"].values,
                                                       "correct_comp", "geo_dist_m_comp")
                    variants["margin_compressed"]  = (-test_df["margin_comp"].values,
                                                       "correct_comp", "geo_dist_m_comp")

                if model_comp is not None and has_comp_cols and "l2_dist_top1_comp" in test_df.columns:
                    Xc = test_df[["n_inliers", "l2_dist_top1_comp", "margin_comp"]].fillna(
                        test_df[["n_inliers", "l2_dist_top1_comp", "margin_comp"]].mean())
                    Xc_scaled = scaler_comp.transform(Xc.values)
                    variants["logreg_compressed"] = (1.0 - model_comp.predict_proba(Xc_scaled)[:, 1],
                                                      "correct_comp", "geo_dist_m_comp")

                for variant_name, (score, correct_col, geo_col) in variants.items():
                    metrics = evaluate_uncertainty_score(
                        score, test_df[correct_col].values,
                        test_df[geo_col].values, test_df["has_gps"].values,
                    )
                    summary_rows.append({
                        "method": method, "matcher": matcher, "dataset": ds,
                        "variant": variant_name, **metrics,
                    })
                    log.info(f"  [{ds}] {variant_name:20s} AUPRC={metrics['auprc']:.4f}  "
                             f"rho={metrics['spearman_rho']:.4f}  R2={metrics['r2']:.4f}  "
                             f"AUSC={metrics['ausc']:.2f}")

            for tag, model, C in [("full", model_full, c_full), ("compressed", model_comp, c_comp)]:
                if model is not None:
                    with open(run_dir / f"logreg_{tag}.json", "w") as f:
                        json.dump({"C": C, "coef": model.coef_.tolist(), "intercept": model.intercept_.tolist()}, f, indent=2)

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(OUT_DIR / "summary.csv", index=False)
        log.info(f"\nSaved {OUT_DIR / 'summary.csv'}  ({len(df_summary)} rows)")
    else:
        log.warning("No result produced - check that knn_evaluation.py, image_matching_evaluation.py "
                    "and features_reduction.py have been run for the requested methods/matchers.")


if __name__ == "__main__":
    main()