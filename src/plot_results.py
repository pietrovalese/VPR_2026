import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
RESULTS_DIR  = ROOT / "logs" / "results"
MATCHING_DIR = RESULTS_DIR / "matching"
FEAT_DIR     = ROOT / "logs" / "feature_reduction"
DESC_DIR     = ROOT / "logs" / "descriptors"
PLOTS_DIR    = ROOT / "logs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Stile
# ---------------------------------------------------------------------------
PALETTE  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
METHODS  = ["cosplace", "netvlad", "mixvpr", "megaloc"]
MATCHERS = ["superglue", "loftr", "superpoint-lg"]
DATASETS = ["sf_xs_test", "tokyo_xs", "svox_sun", "svox_night"]

DS_LABELS = {
    "sf_xs_test":  "SF-XS test",
    "sf_xs_val":   "SF-XS val",
    "tokyo_xs":    "Tokyo-XS",
    "svox_sun":    "SVOX sun",
    "svox_night":  "SVOX night",
}
METHOD_LABELS = {
    "cosplace": "CosPlace",
    "netvlad":  "NetVLAD",
    "mixvpr":   "MixVPR",
    "megaloc":  "MegaLoc",
}
MATCHER_LABELS = {
    "superglue":     "SuperGlue",
    "loftr":         "LoFTR",
    "superpoint-lg": "SP+LG",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize":10,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "savefig.pad_inches": 0.1,
})


def save(fig, name, dpi=300):
    """Save a matplotlib figure under PLOTS_DIR and print its relative path."""
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")


def load_knn():
    """Load the Recall@N table produced by knn_evaluation.py."""
    p = RESULTS_DIR / "recall_table.csv"
    return pd.read_csv(p) if p.exists() else None

def load_matching():
    """Load the image-matching summary produced by image_matching_evaluation.py."""
    p = RESULTS_DIR / "matching_summary.csv"
    return pd.read_csv(p) if p.exists() else None

def load_feat_results(method):
    """Load results.json for a given method from the feature-reduction output."""
    p = FEAT_DIR / method / "results.json"
    return json.loads(p.read_text()) if p.exists() else None

def load_topk_curve(method, split="val"):
    """Load the top-K-by-variance size/recall curve for a given method and split."""
    p = FEAT_DIR / method / f"topk_curve_{split}.csv"
    return pd.read_csv(p) if p.exists() else None

def get_n_db(dataset, method):
    """Return the number of database images for a dataset/method, or None if not extracted."""
    p = DESC_DIR / dataset / method / "database_paths.npy"
    if not p.exists():
        return None
    return len(np.load(p))


# ===========================================================================
# 01 — Recall bar chart
# ===========================================================================
def plot_recall_bar(knn_df, dpi):
    """Plot Recall@1/5/10 bar chart per method x dataset."""
    print("  [01] recall bar chart...")
    metric = "dot" if "dot" in knn_df["metric"].values else knn_df["metric"].iloc[0]
    df = knn_df[(knn_df["metric"] == metric) &
                knn_df["method"].isin(METHODS) &
                knn_df["dataset"].isin(DATASETS)]
    if df.empty:
        print("    skip: no data")
        return

    datasets = [d for d in DATASETS if d in df["dataset"].unique()]
    methods  = [m for m in METHODS  if m in df["method"].unique()]
    rv_list  = [1, 5, 10]
    alphas   = [1.0, 0.65, 0.40]

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    bar_w = 0.8 / len(methods)

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        for m_i, method in enumerate(methods):
            row = sub[sub["method"] == method]
            if row.empty:
                continue
            for rv_i, rv in enumerate(rv_list):
                col = f"R@{rv}"
                if col not in row.columns:
                    continue
                val   = float(row[col].values[0])
                x     = m_i + rv_i * bar_w - (len(rv_list)-1) * bar_w / 2
                color = PALETTE[m_i % len(PALETTE)]
                ax.bar(x, val, width=bar_w*0.9, color=color, alpha=alphas[rv_i])

        ax.set_title(DS_LABELS.get(ds, ds))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=20, ha="right")
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Recall (%)")
    fig.suptitle(f"Recall@N ({metric.upper()} metric)", fontsize=14, y=1.01)

    patches = []
    for rv_i, rv in enumerate(rv_list):
        patches.append(mpatches.Patch(color="gray", alpha=alphas[rv_i], label=f"R@{rv}"))
    for m_i, method in enumerate(methods):
        patches.append(mpatches.Patch(color=PALETTE[m_i % len(PALETTE)],
                                      label=METHOD_LABELS.get(method, method)))
    fig.legend(handles=patches, loc="lower center",
               ncol=len(methods)+len(rv_list), bbox_to_anchor=(0.5, -0.08))
    save(fig, "01_recall_bar.png", dpi)


# ===========================================================================
# 02 — L2 vs Dot
# ===========================================================================
def plot_l2_vs_dot(knn_df, dpi):
    """Scatter plot comparing R@1 under the L2 metric vs the dot-product metric."""
    print("  [02] L2 vs dot...")
    if "dot" not in knn_df["metric"].values or "l2" not in knn_df["metric"].values:
        print("    skip: both metrics are required")
        return

    df       = knn_df[knn_df["method"].isin(METHODS) & knn_df["dataset"].isin(DATASETS)]
    datasets = [d for d in DATASETS if d in df["dataset"].unique()]
    methods  = [m for m in METHODS  if m in df["method"].unique()]

    fig, axes = plt.subplots(1, len(datasets), figsize=(4*len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        l2df  = df[(df["metric"]=="l2")  & (df["dataset"]==ds)]
        dotdf = df[(df["metric"]=="dot") & (df["dataset"]==ds)]
        for m_i, method in enumerate(methods):
            r_l2  = l2df[l2df["method"]==method]
            r_dot = dotdf[dotdf["method"]==method]
            if r_l2.empty or r_dot.empty:
                continue
            v_l2  = float(r_l2["R@1"].values[0])
            v_dot = float(r_dot["R@1"].values[0])
            color = PALETTE[m_i % len(PALETTE)]
            label = METHOD_LABELS.get(method, method)
            ax.scatter(v_l2, v_dot, color=color, s=120, zorder=5, label=label)
            ax.annotate(label, (v_l2, v_dot), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)

        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)
        ax.set_xlabel("L2 — R@1 (%)")
        ax.set_title(DS_LABELS.get(ds, ds))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Dot product — R@1 (%)")
    fig.suptitle("L2 vs Dot Product: R@1 comparison", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(methods), bbox_to_anchor=(0.5, -0.08))
    save(fig, "02_l2_vs_dot.png", dpi)


# ===========================================================================
# 03 — Re-ranking delta
# ===========================================================================
def plot_reranking_delta(matching_df, dpi):
    """Bar plot of R@1 gain after re-ranking, per method x matcher x dataset."""
    print("  [03] re-ranking delta...")
    df = matching_df[matching_df["vpr_method"].isin(METHODS) &
                     matching_df["dataset"].isin(DATASETS) &
                     matching_df["matcher"].isin(MATCHERS)]
    if df.empty:
        print("    skip: no matching data")
        return

    methods  = [m for m in METHODS  if m in df["vpr_method"].unique()]
    matchers = [m for m in MATCHERS if m in df["matcher"].unique()]
    datasets = [d for d in DATASETS if d in df["dataset"].unique()]

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    bar_w = 0.8 / len(matchers)

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"]==ds]
        for mt_i, matcher in enumerate(matchers):
            msub = sub[sub["matcher"]==matcher]
            for m_i, method in enumerate(methods):
                row = msub[msub["vpr_method"]==method]
                if row.empty:
                    continue
                if "R@1_delta" in row.columns:
                    delta = float(row["R@1_delta"].values[0])
                else:
                    delta = float(row["R@1_after"].values[0]) - float(row["R@1_before"].values[0])
                x     = m_i + mt_i * bar_w - (len(matchers)-1) * bar_w / 2
                color = PALETTE[mt_i % len(PALETTE)]
                ax.bar(x, delta, width=bar_w*0.85, color=color,
                       label=MATCHER_LABELS.get(matcher, matcher)
                             if ds == datasets[0] and m_i == 0 else "")

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(DS_LABELS.get(ds, ds))
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=20, ha="right")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Delta R@1 (%)")
    fig.suptitle("R@1 gain after re-ranking", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(matchers), bbox_to_anchor=(0.5, -0.06))
    save(fig, "03_reranking_delta.png", dpi)


# ===========================================================================
# 04 — Timing trade-off
# ===========================================================================
def plot_timing_tradeoff(knn_df, matching_df, dpi):
    """Scatter plot of R@1 vs time per query, for retrieval-only and each matcher."""
    print("  [04] timing trade-off...")

    metric = "dot" if knn_df is not None and "dot" in knn_df["metric"].values else "l2"
    points = []

    if knn_df is not None:
        df = knn_df[(knn_df["metric"]==metric) &
                    knn_df["method"].isin(METHODS) &
                    knn_df["dataset"].isin(DATASETS)]
        for _, row in df.iterrows():
            points.append({
                "method":  row["method"],
                "matcher": "retrieval",
                "dataset": row["dataset"],
                "r1":      float(row.get("R@1", 0)),
                "time_ms": float(row.get("knn_time_ms_per_query", 1)),
            })

    if matching_df is not None:
        df = matching_df[matching_df["vpr_method"].isin(METHODS) &
                         matching_df["dataset"].isin(DATASETS) &
                         matching_df["matcher"].isin(MATCHERS)]
        for _, row in df.iterrows():
            points.append({
                "method":  row["vpr_method"],
                "matcher": row["matcher"],
                "dataset": row["dataset"],
                "r1":      float(row.get("R@1_after", 0)),
                "time_ms": float(row.get("avg_time_per_query_s", 0)) * 1000,
            })

    if not points:
        print("    skip: no data")
        return

    pf = pd.DataFrame(points)
    datasets = [d for d in DATASETS if d in pf["dataset"].unique()]
    methods  = [m for m in METHODS  if m in pf["method"].unique()]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    markers = {"retrieval": "o", "superglue": "s", "loftr": "^", "superpoint-lg": "D"}

    for ax, ds in zip(axes, datasets):
        sub = pf[pf["dataset"]==ds]
        for m_i, method in enumerate(methods):
            color = PALETTE[m_i % len(PALETTE)]
            m_sub = sub[sub["method"]==method]
            ret   = m_sub[m_sub["matcher"]=="retrieval"]
            for _, row in m_sub.iterrows():
                mk = markers.get(row["matcher"], "o")
                ax.scatter(row["time_ms"], row["r1"],
                           color=color, s=100, zorder=5, marker=mk,
                           label=METHOD_LABELS.get(method, method)
                                 if row["matcher"]=="retrieval" else "")
                if row["matcher"] != "retrieval" and not ret.empty:
                    ax.plot([float(ret["time_ms"].values[0]), row["time_ms"]],
                            [float(ret["r1"].values[0]),      row["r1"]],
                            color=color, alpha=0.3, linewidth=1)

        ax.set_xscale("log")
        ax.set_xlabel("Time per query (ms, log)")
        ax.set_ylabel("R@1 (%)")
        ax.set_title(DS_LABELS.get(ds, ds))
        ax.grid(True, alpha=0.3, which="both")

    # Legenda marker
    marker_patches = [
        plt.scatter([], [], marker="o", color="gray", s=80, label="Retrieval only"),
        plt.scatter([], [], marker="s", color="gray", s=80, label="+ SuperGlue"),
        plt.scatter([], [], marker="^", color="gray", s=80, label="+ LoFTR"),
        plt.scatter([], [], marker="D", color="gray", s=80, label="+ SP+LG"),
    ]
    handles, labels = axes[0].get_legend_handles_labels()
    all_handles = handles + marker_patches
    all_labels  = labels + [h.get_label() for h in marker_patches]
    fig.legend(all_handles, all_labels, loc="lower center",
               ncol=len(methods)+4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Performance / Efficiency Trade-off", fontsize=13)
    save(fig, "04_timing_tradeoff.png", dpi)


# ===========================================================================
# 05 - Inlier histograms
# ===========================================================================
def plot_inlier_histograms(dpi):
    """Plot inlier-count histograms for correct vs wrong top-1 retrievals."""
    print("  [05] inlier histograms...")

    combos = []
    if MATCHING_DIR.exists():
        for d in sorted(MATCHING_DIR.iterdir()):
            if (d/"per_query_inliers.npy").exists() and (d/"correct_mask.npy").exists():
                combos.append(d)

    if not combos:
        print("    skip: no inliers data")
        return

    # Prefer superglue, max 8 combos
    sg = [c for c in combos if "superglue" in c.name]
    show = (sg if sg else combos)[:8]

    n     = len(show)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, d in zip(axes_flat, show):
        inliers = np.load(d / "per_query_inliers.npy")
        correct = np.load(d / "correct_mask.npy")
        inl_c   = inliers[correct]
        inl_w   = inliers[~correct]
        bins    = np.linspace(0, max(inliers.max(), 1), 40)

        ax.hist(inl_c, bins=bins, alpha=0.65, color="#4CAF50",
                label=f"Correct (n={len(inl_c)})", density=True)
        ax.hist(inl_w, bins=bins, alpha=0.65, color="#F44336",
                label=f"Wrong (n={len(inl_w)})", density=True)
        ax.axvline(inl_c.mean(), color="#1B5E20", linestyle="--",
                   linewidth=1.5, label=f"mean_c={inl_c.mean():.1f}")
        ax.axvline(inl_w.mean(), color="#B71C1C", linestyle="--",
                   linewidth=1.5, label=f"mean_w={inl_w.mean():.1f}")
        ax.set_title(d.name, fontsize=8)
        ax.set_xlabel("Inliers")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Inlier distribution: correct vs wrong queries (top-1 retrieved)", fontsize=13)
    plt.tight_layout()
    save(fig, "05_inlier_histograms.png", dpi)


# ===========================================================================
# 06 - Top-K variance curve
# ===========================================================================
def plot_topk_curve(dpi):
    """Plot the size/recall curve for the top-K-by-variance feature reduction."""
    print("  [06] top-K variance curve...")

    methods_avail = [m for m in ["cosplace", "megaloc"]
                     if (FEAT_DIR / m / "topk_curve_val.csv").exists()]
    if not methods_avail:
        print("    skip: topk_curve_val.csv not found")
        return

    fig, axes = plt.subplots(1, len(methods_avail),
                             figsize=(6*len(methods_avail), 4.5))
    if len(methods_avail) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_avail):
        df = load_topk_curve(method, "val")
        if df is None or df.empty:
            continue
        df = df.sort_values("n_kept")
        D  = int(df["n_total"].iloc[0])

        r1_col = "R@1" if "R@1" in df.columns else df.columns[-1]
        ax.plot(df["n_kept"], df[r1_col], "o-",
                color=PALETTE[0], lw=2, ms=6, label="R@1")
        if "R@5" in df.columns:
            ax.plot(df["n_kept"], df["R@5"], "s--",
                    color=PALETTE[1], lw=1.5, ms=5, alpha=0.8, label="R@5")
        if "R@10" in df.columns:
            ax.plot(df["n_kept"], df["R@10"], "^:",
                    color=PALETTE[2], lw=1.5, ms=5, alpha=0.8, label="R@10")

        baseline = df[df["topk_fraction"]==1.0][r1_col].values
        if len(baseline):
            ax.axhline(baseline[0], color="gray", ls="--", lw=1, alpha=0.6,
                       label=f"Baseline {baseline[0]:.1f}%")
            ax.axhline(baseline[0]-2, color="gray", ls=":", lw=1, alpha=0.4,
                       label="Baseline -2%")
            # Optimal point
            valid = df[df[r1_col] >= baseline[0]-2]
            if not valid.empty:
                opt = valid.sort_values("n_kept").iloc[0]
                ax.axvline(opt["n_kept"], color="red", ls="--", lw=1.5, alpha=0.7,
                           label=f"Optimal {int(opt['n_kept'])}D "
                                 f"({opt['compression_pct']:.0f}% saved)")

        ax.set_xlabel(f"Features kept (out of {D})")
        ax.set_ylabel("Recall (%)")
        ax.set_title(f"{METHOD_LABELS.get(method, method)} — Dimension/Recall curve")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Secondary axis with percentage
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ticks = [t for t in ax.get_xticks() if 0 <= t <= D]
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f"{100*t/D:.0f}%" for t in ticks], fontsize=8)
        ax2.set_xlabel("% features kept")

    fig.suptitle("Feature Reduction: Dimension vs Recall (top-K by variance)", fontsize=13)
    plt.tight_layout()
    save(fig, "06_topk_curve.png", dpi)


# ===========================================================================
# 07 - Memory
# ===========================================================================
def plot_memory_saving(dpi):
    """Bar plot comparing database memory usage: full vs compressed descriptors."""
    print("  [07] memory saving...")

    rows = []
    for method in ["cosplace", "megaloc"]:
        res = load_feat_results(method)
        if res is None:
            continue
        D_full = res["descriptor_dim"]
        tkv    = res.get("topk_variance", {})
        n_kept = tkv.get("best_topk_n_kept", D_full)

        for ds in DATASETS:
            n_db = get_n_db(ds, method)
            if n_db is None:
                continue
            mb_full = n_db * D_full * 4 / 1e6
            mb_comp = n_db * n_kept * 4 / 1e6
            rows.append({
                "method":      METHOD_LABELS.get(method, method),
                "dataset":     DS_LABELS.get(ds, ds),
                "D_full":      D_full,
                "D_kept":      n_kept,
                "n_db":        n_db,
                "mb_full":     round(mb_full, 2),
                "mb_comp":     round(mb_comp, 2),
                "saving_pct":  round(100*(1 - mb_comp/mb_full), 1),
            })

    if not rows:
        print("    skip: no data")
        return

    df = pd.DataFrame(rows)
    df.to_csv(PLOTS_DIR / "table_memory.csv", index=False)

    methods_avail = df["method"].unique()
    fig, axes = plt.subplots(1, len(methods_avail),
                             figsize=(6*len(methods_avail), 5))
    if len(methods_avail) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_avail):
        sub = df[df["method"]==method].reset_index(drop=True)
        x   = np.arange(len(sub))
        w   = 0.35
        ax.bar(x-w/2, sub["mb_full"], w, label="Full", color=PALETTE[0], alpha=0.85)
        ax.bar(x+w/2, sub["mb_comp"], w, label="Compressed", color=PALETTE[1], alpha=0.85)
        for i, row in sub.iterrows():
            ax.text(i, row["mb_full"]*1.02, f"-{row['saving_pct']:.0f}%",
                    ha="center", va="bottom", fontsize=9, color=PALETTE[1], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["dataset"], rotation=15, ha="right")
        ax.set_ylabel("Memory (MB)")
        ax.set_title(f"{method}\n{sub['D_full'].iloc[0]}D -> {sub['D_kept'].iloc[0]}D")
        ax.legend()
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Database Memory: Full vs Compressed Descriptors", fontsize=13)
    plt.tight_layout()
    save(fig, "07_memory_saving.png", dpi)


# ===========================================================================
# 08 — Grad-CAM
# ===========================================================================
def plot_gradcam(dpi):
    """Plot Grad-CAM heatmaps for kept vs removed descriptor features."""
    print("  [08] Grad-CAM...")

    for method in ["cosplace", "megaloc"]:
        method_dir = FEAT_DIR / method
        paths_file = method_dir / "gradcam_image_paths.json"
        if not paths_file.exists():
            continue

        img_paths = json.loads(paths_file.read_text())
        n_imgs    = min(3, len(img_paths))
        n_feats   = 3

        kept_idxs = method_dir / "gradcam_kept_feat_indices.npy"
        rem_idxs  = method_dir / "gradcam_removed_feat_indices.npy"
        kept_idx_v = np.load(kept_idxs) if kept_idxs.exists() else np.array([])
        rem_idx_v  = np.load(rem_idxs)  if rem_idxs.exists()  else np.array([])

        has_removed = len(rem_idx_v) > 0
        n_cols = 1 + n_feats + (n_feats if has_removed else 0)

        fig, axes = plt.subplots(n_imgs, n_cols,
                                 figsize=(3*n_cols, 3.2*n_imgs))
        if n_imgs == 1:
            axes = axes[np.newaxis, :]

        for img_i in range(n_imgs):
            # Original image
            try:
                img = Image.open(img_paths[img_i]).convert("RGB")
                axes[img_i, 0].imshow(img)
            except Exception:
                pass
            axes[img_i, 0].set_title("Original" if img_i == 0 else "")
            axes[img_i, 0].axis("off")

            # Kept
            kf = method_dir / f"gradcam_kept_img{img_i}.npy"
            if kf.exists():
                hm = np.load(kf)
                for f_i in range(min(n_feats, hm.shape[0])):
                    axes[img_i, 1+f_i].imshow(hm[f_i], cmap="jet", vmin=0, vmax=1)
                    fidx = kept_idx_v[f_i] if f_i < len(kept_idx_v) else "?"
                    axes[img_i, 1+f_i].set_title(f"Kept #{fidx}" if img_i==0 else "")
                    axes[img_i, 1+f_i].axis("off")

            # Removed
            if has_removed:
                rf = method_dir / f"gradcam_removed_img{img_i}.npy"
                if rf.exists():
                    hm = np.load(rf)
                    for f_i in range(min(n_feats, hm.shape[0])):
                        axes[img_i, 1+n_feats+f_i].imshow(
                            hm[f_i], cmap="hot", vmin=0, vmax=1)
                        fidx = rem_idx_v[f_i] if f_i < len(rem_idx_v) else "?"
                        axes[img_i, 1+n_feats+f_i].set_title(
                            f"Removed #{fidx}" if img_i==0 else "")
                        axes[img_i, 1+n_feats+f_i].axis("off")

        fig.suptitle(f"Grad-CAM: {METHOD_LABELS.get(method, method)} — "
                     f"kept (jet) vs removed (hot)", fontsize=11)
        plt.tight_layout()
        save(fig, f"08_gradcam_{method}.png", dpi)


# ===========================================================================
# Paper-ready CSV tables
# ===========================================================================
def build_tables(knn_df, matching_df):
    """Build paper-ready Recall@N and timing CSV tables."""
    print("  CSV tables...")
    rv = [1, 5, 10, 20]

    # Recall table
    rows = []
    metric = "dot"
    if knn_df is not None:
        df = knn_df[(knn_df["metric"]==metric) &
                    knn_df["method"].isin(METHODS) &
                    knn_df["dataset"].isin(DATASETS)]
        for _, r in df.iterrows():
            row = {
                "VPR Method": METHOD_LABELS.get(r["method"], r["method"]),
                "Matcher":    "—",
                "Dataset":    DS_LABELS.get(r["dataset"], r["dataset"]),
            }
            for n in rv:
                row[f"R@{n}"] = round(float(r[f"R@{n}"]), 2) if f"R@{n}" in r else ""
            rows.append(row)

    if matching_df is not None:
        df = matching_df[matching_df["vpr_method"].isin(METHODS) &
                         matching_df["dataset"].isin(DATASETS) &
                         matching_df["matcher"].isin(MATCHERS)]
        for _, r in df.iterrows():
            row = {
                "VPR Method": METHOD_LABELS.get(r["vpr_method"], r["vpr_method"]),
                "Matcher":    MATCHER_LABELS.get(r["matcher"], r["matcher"]),
                "Dataset":    DS_LABELS.get(r["dataset"], r["dataset"]),
            }
            for n in rv:
                col = f"R@{n}_after"
                row[f"R@{n}"] = round(float(r[col]), 2) if col in r.index else ""
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(PLOTS_DIR / "table_recall.csv", index=False)
        print(f"    table_recall.csv ({len(rows)} rows)")

    # Timing table
    rows = []
    if knn_df is not None:
        df = knn_df[(knn_df["metric"]==metric) & knn_df["method"].isin(METHODS)]
        for _, r in df.iterrows():
            rows.append({
                "Stage":    "Retrieval",
                "Method":   METHOD_LABELS.get(r["method"], r["method"]),
                "Matcher":  "—",
                "Dataset":  DS_LABELS.get(r["dataset"], r["dataset"]),
                "ms/query": round(float(r["knn_time_ms_per_query"]), 3)
                             if "knn_time_ms_per_query" in r else "",
                "R@1":      round(float(r["R@1"]), 2) if "R@1" in r else "",
            })
    if matching_df is not None:
        df = matching_df[matching_df["vpr_method"].isin(METHODS) &
                         matching_df["matcher"].isin(MATCHERS)]
        for _, r in df.iterrows():
            rows.append({
                "Stage":    "Re-ranking",
                "Method":   METHOD_LABELS.get(r["vpr_method"], r["vpr_method"]),
                "Matcher":  MATCHER_LABELS.get(r["matcher"], r["matcher"]),
                "Dataset":  DS_LABELS.get(r["dataset"], r["dataset"]),
                "ms/query": round(float(r["avg_time_per_query_s"])*1000, 2)
                             if "avg_time_per_query_s" in r.index else "",
                "R@1":      round(float(r["R@1_after"]), 2) if "R@1_after" in r.index else "",
            })
    if rows:
        pd.DataFrame(rows).to_csv(PLOTS_DIR / "table_timing.csv", index=False)
        print(f"    table_timing.csv ({len(rows)} rows)")


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None,
                   help="E.g.: --only 01 05 06")
    p.add_argument("--dpi",  type=int, default=300)
    return p.parse_args()

def should_run(name, only):
    """Check whether a plot with the given id prefix should run, based on --only."""
    return only is None or any(name.startswith(o) for o in only)

def main():
    """Load all pipeline results and generate every requested plot/table."""
    args = parse_args()
    print(f"Output -> {PLOTS_DIR.relative_to(ROOT)}/\n")

    knn_df      = load_knn()
    matching_df = load_matching()

    if knn_df is not None:      print(f"KNN:      {len(knn_df)} rows")
    if matching_df is not None: print(f"Matching: {len(matching_df)} rows")
    print()

    if should_run("01", args.only) and knn_df is not None:
        plot_recall_bar(knn_df, args.dpi)
    if should_run("02", args.only) and knn_df is not None:
        plot_l2_vs_dot(knn_df, args.dpi)
    if should_run("03", args.only) and matching_df is not None:
        plot_reranking_delta(matching_df, args.dpi)
    if should_run("04", args.only):
        plot_timing_tradeoff(knn_df, matching_df, args.dpi)
    if should_run("05", args.only):
        plot_inlier_histograms(args.dpi)
    if should_run("06", args.only):
        plot_topk_curve(args.dpi)
    if should_run("07", args.only):
        plot_memory_saving(args.dpi)
    if should_run("08", args.only):
        plot_gradcam(args.dpi)
    if args.only is None or "table" in (args.only or []):
        build_tables(knn_df, matching_df)

    print(f"\nFigure in {PLOTS_DIR.relative_to(ROOT)}/")
    for p in sorted(PLOTS_DIR.iterdir()):
        print(f"  {p.name}")

if __name__ == "__main__":
    main()