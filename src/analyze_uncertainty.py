import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths - same conventions as plot_results.py
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parent.parent
UNCERTAINTY_DIR = ROOT / "logs" / "uncertainty"
FEAT_DIR        = ROOT / "logs" / "feature_reduction"
ANALYSIS_DIR    = UNCERTAINTY_DIR / "analysis"
PLOTS_DIR       = ROOT / "logs" / "plots"

PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

RAW_VARIANTS   = ["n_inliers", "l2_dist", "margin", "logreg_full"]
COMP_PAIRS     = [("l2_dist", "l2_dist_compressed"),
                   ("margin", "margin_compressed"),
                   ("logreg_full", "logreg_compressed")]


def save(fig, name, dpi=300):
    """Save a matplotlib figure under PLOTS_DIR and print its relative path."""
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. Standalone 6.2 analysis
# ---------------------------------------------------------------------------
def load_uncertainty_summary() -> pd.DataFrame | None:
    """Load logs/uncertainty/summary.csv produced by uncertainty_estimation.py."""
    p = UNCERTAINTY_DIR / "summary.csv"
    if not p.exists():
        print(f"[!] Missing {p} - run src/uncertainty_estimation.py first")
        return None
    return pd.read_csv(p)


def verify_compression_no_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify how much full and compressed variants actually differ (not just 'look the same')."""
    rows = []
    for full, comp in COMP_PAIRS:
        a = df[df.variant == full].set_index(["method", "matcher", "dataset"])[["auprc", "r2", "ausc"]]
        b = df[df.variant == comp].set_index(["method", "matcher", "dataset"])[["auprc", "r2", "ausc"]]
        diff = (a - b).abs()
        rows.append({
            "variant_pair":     f"{full} vs {comp}",
            "max_abs_diff_auprc": diff["auprc"].max(),
            "max_abs_diff_r2":    diff["r2"].max(),
            "max_abs_diff_ausc":  diff["ausc"].max(),
            "mean_abs_diff_auprc": diff["auprc"].mean(),
        })
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS_DIR / "table_compression_no_degradation.csv", index=False)
    print("\n=== Full vs Compressed - does compression degrade the uncertainty estimate? ===")
    print(out.to_string(index=False))
    return out


def pivot_by_variant(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot a metric (auprc/r2/...) into a variant x method table and save it as CSV."""
    piv = df.pivot_table(index="variant", columns="method", values=metric, aggfunc="mean")
    piv = piv.reindex(RAW_VARIANTS + [v for v in piv.index if v not in RAW_VARIANTS])
    piv.to_csv(ANALYSIS_DIR / f"table_{metric}_by_variant.csv")
    return piv


def plot_auprc_by_variant(piv_auprc: pd.DataFrame, dpi: int):
    """Bar plot of mean AUPRC per uncertainty variant, grouped by VPR method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    piv_auprc.loc[RAW_VARIANTS].plot(kind="bar", ax=ax, color=PALETTE[:len(piv_auprc.columns)])
    ax.set_ylabel("AUPRC (averaged over dataset/matcher)")
    ax.set_xlabel("")
    ax.set_title("6.2 - Uncertainty measure quality by VPR method")
    ax.set_ylim(0, 1)
    ax.legend(title="VPR Method")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    save(fig, "09_auprc_by_variant.png", dpi)


# ---------------------------------------------------------------------------
# 2. Sparsification Error (oracle-normalized AUSC)
# ---------------------------------------------------------------------------
def sparsification_curve(unc_score: np.ndarray, err: np.ndarray, n_steps: int = 20) -> np.ndarray:
    """Compute the sparsification curve: mean residual error as increasing fractions of the
    highest-uncertainty queries are discarded."""
    order = np.argsort(-unc_score)
    err_sorted = err[order]
    N = len(err_sorted)
    fractions = np.linspace(0, 0.95, n_steps)
    return np.array([
        err_sorted[int(f * N):].mean() if int(f * N) < N else 0.0
        for f in fractions
    ])


def compute_sparsification_error(method: str, matcher: str, dataset: str) -> list[dict] | None:
    """
    Read test_<dataset>_features.csv (saved by uncertainty_estimation.py) and compute, for
    each available variant, the REAL sparsification curve (ordering by the uncertainty score)
    vs the ORACLE curve (ordering by the true geographic error, the best possible case). The
    area between the two curves (Sparsification Error) isolates how much worse the uncertainty
    measure is than a perfect ranking, independent of the model's baseline average error -
    unlike the raw AUSC, this is comparable across different methods (e.g. CosPlace vs
    MegaLoc), which have very different average errors.

    "Compressed" variants use their own geographic error (geo_dist_m_comp, from the retrieval
    recomputed on the masked descriptors) and their own oracle - not the full retrieval's -
    otherwise we'd be measuring how well a compressed signal ranks the errors of a different
    system than the one that produced it.
    """
    feat_path = UNCERTAINTY_DIR / f"{method}_{matcher}" / f"test_{dataset}_features.csv"
    if not feat_path.exists():
        return None
    df = pd.read_csv(feat_path)

    fractions = np.linspace(0, 0.95, 20)

    def _ausc(score: np.ndarray, err: np.ndarray) -> float:
        curve = sparsification_curve(score, err)
        return np.trapz(curve, fractions) / (fractions[-1] - fractions[0])

    rows = []

    # --- "Full" variants: error/oracle of the full retrieval ---
    mask_full = df["has_gps"].astype(bool) & df["geo_dist_m"].notna()
    df_full = df[mask_full]
    if len(df_full) >= 20:
        err_full = df_full["geo_dist_m"].values
        ausc_oracle_full = _ausc(err_full, err_full)
        full_variants = {
            "n_inliers": -df_full["n_inliers"].values,
            "l2_dist":    df_full["l2_dist_top1"].values,
            "margin":    -df_full["margin"].values,
        }
        for name, score in full_variants.items():
            ausc_obs = _ausc(score, err_full)
            rows.append({
                "method": method, "matcher": matcher, "dataset": dataset, "variant": name,
                "ausc_observed": ausc_obs, "ausc_oracle": ausc_oracle_full,
                "sparsification_error": ausc_obs - ausc_oracle_full,
            })

    # --- "Compressed" variants: error/oracle of the compressed retrieval ---
    if "geo_dist_m_comp" in df.columns:
        mask_comp = df["has_gps"].astype(bool) & df["geo_dist_m_comp"].notna()
        df_comp = df[mask_comp]
        if len(df_comp) >= 20 and "l2_dist_top1_comp" in df_comp.columns:
            err_comp = df_comp["geo_dist_m_comp"].values
            ausc_oracle_comp = _ausc(err_comp, err_comp)
            comp_variants = {
                "l2_dist_compressed": df_comp["l2_dist_top1_comp"].values,
                "margin_compressed": -df_comp["margin_comp"].values,
            }
            for name, score in comp_variants.items():
                ausc_obs = _ausc(score, err_comp)
                rows.append({
                    "method": method, "matcher": matcher, "dataset": dataset, "variant": name,
                    "ausc_observed": ausc_obs, "ausc_oracle": ausc_oracle_comp,
                    "sparsification_error": ausc_obs - ausc_oracle_comp,
                })

    return rows if rows else None


def build_sparsification_error_table(methods, matchers, datasets) -> pd.DataFrame:
    """Compute and save the Sparsification Error table across all method/matcher/dataset combos."""
    all_rows = []
    for method in methods:
        for matcher in matchers:
            for dataset in datasets:
                rows = compute_sparsification_error(method, matcher, dataset)
                if rows:
                    all_rows.extend(rows)
    if not all_rows:
        print("[!] No test_<dataset>_features.csv found - run uncertainty_estimation.py first")
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df.to_csv(ANALYSIS_DIR / "table_sparsification_error.csv", index=False)
    print("\n=== Sparsification Error (lower = uncertainty ranking closer to the oracle) ===")
    print(df.groupby(["method", "variant"])["sparsification_error"].mean().round(2).to_string())
    return df


def plot_sparsification_error(df: pd.DataFrame, dpi: int):
    """Bar plot of mean Sparsification Error per uncertainty variant, grouped by VPR method."""
    if df.empty:
        return
    piv = df.pivot_table(index="variant", columns="method", values="sparsification_error", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    piv.plot(kind="bar", ax=ax, color=PALETTE[:len(piv.columns)])
    ax.set_ylabel("Sparsification Error (m)  -  lower = better")
    ax.set_xlabel("")
    ax.set_title("6.2 - Uncertainty ranking quality (normalized vs oracle)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    save(fig, "10_sparsification_error.png", dpi)


# ---------------------------------------------------------------------------
# 3. Combining 6.2 + 6.3
# ---------------------------------------------------------------------------
def load_feature_reduction_summary() -> pd.DataFrame | None:
    """Load logs/feature_reduction/summary.csv produced by features_reduction.py."""
    p = FEAT_DIR / "summary.csv"
    if not p.exists():
        print(f"[!] Missing {p} - run src/features_reduction.py first")
        return None
    return pd.read_csv(p)


def build_combined_table(df_unc: pd.DataFrame, df_feat: pd.DataFrame, datasets: list) -> pd.DataFrame:
    """
    For each method x test dataset: % compression (6.3), R@1 delta full->compressed (6.3),
    AUPRC delta full->compressed on logreg (6.2, averaged over matchers). A single row answers
    both "how much retrieval performance is lost" and "how much uncertainty reliability is
    lost" for the same compression cut.
    """
    rows = []
    for _, r in df_feat.iterrows():
        method = r["method"]
        # Use the recall/compression based on the top-K-by-variance mask
        # (mask_topk_variance), not "multilayer_compression_pct": the latter reflects the
        # redundancy mask, which with weakly correlated VPR descriptors removes almost
        # nothing (see the note in features_reduction.py).
        comp_pct = r["topk_compression_pct"]
        for ds in datasets:
            r1_full = r.get(f"{ds}_R@1_full_topk")
            r1_comp = r.get(f"{ds}_R@1_compressed_topk")
            r1_delta = r.get(f"{ds}_R@1_delta_topk")
            if r1_full is None or pd.isna(r1_full):
                continue

            sub = df_unc[(df_unc.method == method) & (df_unc.dataset == ds) &
                         (df_unc.variant.isin(["logreg_full", "logreg_compressed"]))]
            if sub.empty:
                auprc_full = auprc_comp = auprc_delta = None
            else:
                auprc_full = sub[sub.variant == "logreg_full"]["auprc"].mean()
                auprc_comp = sub[sub.variant == "logreg_compressed"]["auprc"].mean()
                auprc_delta = auprc_comp - auprc_full

            rows.append({
                "method": method, "dataset": ds,
                "compression_pct": comp_pct,
                "R@1_full": r1_full, "R@1_compressed": r1_comp, "R@1_delta": r1_delta,
                "AUPRC_full": auprc_full, "AUPRC_compressed": auprc_comp, "AUPRC_delta": auprc_delta,
            })
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS_DIR / "table_combined_6_2_6_3.csv", index=False)
    print("\n=== Combined 6.3 (recall) + 6.2 (uncertainty) ===")
    print(out.round(4).to_string(index=False))
    return out


def plot_combined(df_combined: pd.DataFrame, dpi: int):
    """Bar plot comparing R@1 delta (6.3) against AUPRC delta (6.2), one panel per method."""
    if df_combined.empty:
        return
    methods = df_combined["method"].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 4.5), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = df_combined[df_combined.method == method]
        x = np.arange(len(sub))
        w = 0.35
        # AUPRC is in probability units (typical range 0.001-0.2), while R@1 is in
        # percentage points (typical range 1-11): on the same scale the AUPRC bars would be
        # invisible. Both are expressed in "percentage points" by multiplying the AUPRC
        # delta by 100.
        ax.bar(x - w/2, sub["R@1_delta"], width=w, label="Delta R@1 (6.3)", color=PALETTE[0])
        ax.bar(x + w/2, sub["AUPRC_delta"] * 100, width=w, label="Delta AUPRC logreg x100 (6.2)", color=PALETTE[1])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["dataset"], rotation=20, ha="right")
        ax.set_title(f"{method}  ({sub['compression_pct'].iloc[0]:.1f}% compressed)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Delta in percentage points (compressed - full)")
    axes[0].legend()
    fig.suptitle("Effect of compression (6.3): retrieval vs uncertainty quality (6.2)")
    save(fig, "11_combined_compression_vs_uncertainty.png", dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--methods",  nargs="+", default=["cosplace", "megaloc"])
    p.add_argument("--matchers", nargs="+", default=["superglue", "loftr"])
    p.add_argument("--datasets", nargs="+", default=["sf_xs_test", "tokyo_xs", "svox_sun", "svox_night"])
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    """Run the standalone 6.2 analysis and combine it with the 6.3 feature-reduction results."""
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("6.2 ANALYSIS (Uncertainty) + combination with 6.3 (Feature Reduction)")
    print("="*70)

    df_unc = load_uncertainty_summary()
    if df_unc is None:
        return

    # --- 1. Standalone 6.2 ---
    verify_compression_no_degradation(df_unc)
    piv_auprc = pivot_by_variant(df_unc, "auprc")
    pivot_by_variant(df_unc, "r2")
    print("\n=== Mean AUPRC per method x variant ===")
    print(piv_auprc.round(3).to_string())
    plot_auprc_by_variant(piv_auprc, args.dpi)

    # --- 2. Sparsification Error ---
    df_sparse = build_sparsification_error_table(args.methods, args.matchers, args.datasets)
    plot_sparsification_error(df_sparse, args.dpi)

    # --- 3. Combination with 6.3 ---
    df_feat = load_feature_reduction_summary()
    if df_feat is not None:
        df_combined = build_combined_table(df_unc, df_feat, args.datasets)
        plot_combined(df_combined, args.dpi)

    print(f"\nAll tables in {ANALYSIS_DIR.relative_to(ROOT)}/, figures in {PLOTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()