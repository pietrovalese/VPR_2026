"""
src/analyze_uncertainty.py

Analisi dei risultati dell'estensione 6.2 (Uncertainty Estimation) e
combinazione con i risultati della 6.3 (Feature/Memory reduction).

Cosa fa
-------
1. Analisi standalone 6.2:
   - Verifica quantitativa full-vs-compressed (quanto davvero cambia AUPRC/R2/AUSC
     rimuovendo le feature scartate dalla 6.3) — conferma numerica, non solo visiva.
   - Tabelle pivot: AUPRC/R2/AUSC medi per (metodo x variante di incertezza).
   - Sparsification Error = AUSC_osservato - AUSC_oracolo. L'AUSC "grezzo" salvato
     da uncertainty_estimation.py è in metri assoluti, quindi è dominato
     dall'errore medio di base del modello (MegaLoc ha errori piccoli di suo,
     quindi AUSC basso anche con un'incertezza poco informativa). Sottraendo la
     curva "oracolo" (che ordina le query per errore VERO, il miglior
     ordinamento possibile) si isola la qualità del ranking di incertezza dalla
     accuratezza di base del modello — è la metrica giusta da riportare nel
     report per confrontare metodi diversi tra loro.

2. Combinazione con la 6.3:
   - Per ciascun metodo, affianca: % di compressione (6.3), delta di R@1
     full->compressed (6.3), e delta di AUPRC full->compressed (6.2) sullo
     stesso dataset di test — un'unica tabella/figura che risponde alla domanda
     "la compressione fa male al retrieval? e all'affidabilità delle sue
     predizioni?" nello stesso posto.

Prerequisiti
------------
    python src/uncertainty_estimation.py        # produce logs/uncertainty/{summary.csv, <method>_<matcher>/test_*.csv}
    python src/features_reduction.py             # produce logs/feature_reduction/summary.csv

Output (in logs/uncertainty/analysis/)
---------------------------------------
    table_auprc_by_variant.csv       — AUPRC medio per metodo x variante
    table_r2_by_variant.csv          — R2 medio per metodo x variante
    table_sparsification_error.csv   — AUSC oracolo-normalizzato per combo x dataset x variante
    table_combined_6_2_6_3.csv       — tabella unica compressione + delta R@1 + delta AUPRC
    09_auprc_by_variant.png
    10_sparsification_error.png
    11_combined_compression_vs_uncertainty.png

Uso
---
    python src/analyze_uncertainty.py
    python src/analyze_uncertainty.py --dpi 200
"""

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
# Paths — stesse convenzioni di plot_results.py
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
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. Analisi standalone 6.2
# ---------------------------------------------------------------------------
def load_uncertainty_summary() -> pd.DataFrame | None:
    p = UNCERTAINTY_DIR / "summary.csv"
    if not p.exists():
        print(f"[!] Manca {p} — esegui prima src/uncertainty_estimation.py")
        return None
    return pd.read_csv(p)


def verify_compression_no_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Quantifica quanto full e compressed differiscono davvero (non solo 'sembrano uguali')."""
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
    print("\n=== Full vs Compressed — la compressione degrada l'incertezza? ===")
    print(out.to_string(index=False))
    return out


def pivot_by_variant(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    piv = df.pivot_table(index="variant", columns="method", values=metric, aggfunc="mean")
    piv = piv.reindex(RAW_VARIANTS + [v for v in piv.index if v not in RAW_VARIANTS])
    piv.to_csv(ANALYSIS_DIR / f"table_{metric}_by_variant.csv")
    return piv


def plot_auprc_by_variant(piv_auprc: pd.DataFrame, dpi: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    piv_auprc.loc[RAW_VARIANTS].plot(kind="bar", ax=ax, color=PALETTE[:len(piv_auprc.columns)])
    ax.set_ylabel("AUPRC (medio su dataset/matcher)")
    ax.set_xlabel("")
    ax.set_title("6.2 — Qualità delle misure di incertezza per metodo VPR")
    ax.set_ylim(0, 1)
    ax.legend(title="Metodo VPR")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    save(fig, "09_auprc_by_variant.png", dpi)


# ---------------------------------------------------------------------------
# 2. Sparsification Error (AUSC oracolo-normalizzato)
# ---------------------------------------------------------------------------
def sparsification_curve(unc_score: np.ndarray, err: np.ndarray, n_steps: int = 20) -> np.ndarray:
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
    Legge test_<dataset>_features.csv (salvato da uncertainty_estimation.py) e
    calcola, per ogni variante disponibile, la curva di sparsificazione REALE
    (ordinando per lo score di incertezza) vs quella ORACOLO (ordinando per
    l'errore geografico vero, il miglior caso possibile). L'area tra le due
    curve (Sparsification Error) isola quanto la misura di incertezza è
    peggiore del ranking perfetto, indipendentemente dall'errore medio di base
    del modello — a differenza dell'AUSC grezzo, è confrontabile tra metodi
    diversi (es. CosPlace vs MegaLoc), che hanno errori medi molto diversi.

    Le varianti "compressed" usano il proprio errore geografico (geo_dist_m_comp,
    dal retrieval ricalcolato sui descrittori mascherati) e il proprio oracolo —
    non quello del retrieval full — altrimenti si misurerebbe quanto bene un
    segnale compressed ordina gli errori di un sistema diverso da quello che
    lo ha prodotto.
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

    # --- Varianti "full": errore/oracolo del retrieval full ---
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

    # --- Varianti "compressed": errore/oracolo del retrieval compressed ---
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
    all_rows = []
    for method in methods:
        for matcher in matchers:
            for dataset in datasets:
                rows = compute_sparsification_error(method, matcher, dataset)
                if rows:
                    all_rows.extend(rows)
    if not all_rows:
        print("[!] Nessun test_<dataset>_features.csv trovato — esegui prima uncertainty_estimation.py")
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df.to_csv(ANALYSIS_DIR / "table_sparsification_error.csv", index=False)
    print("\n=== Sparsification Error (piu' basso = ranking incertezza piu' vicino all'oracolo) ===")
    print(df.groupby(["method", "variant"])["sparsification_error"].mean().round(2).to_string())
    return df


def plot_sparsification_error(df: pd.DataFrame, dpi: int):
    if df.empty:
        return
    piv = df.pivot_table(index="variant", columns="method", values="sparsification_error", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    piv.plot(kind="bar", ax=ax, color=PALETTE[:len(piv.columns)])
    ax.set_ylabel("Sparsification Error (m)  —  piu' basso = meglio")
    ax.set_xlabel("")
    ax.set_title("6.2 — Qualita' del ranking di incertezza (normalizzato vs oracolo)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    save(fig, "10_sparsification_error.png", dpi)


# ---------------------------------------------------------------------------
# 3. Combinazione 6.2 + 6.3
# ---------------------------------------------------------------------------
def load_feature_reduction_summary() -> pd.DataFrame | None:
    p = FEAT_DIR / "summary.csv"
    if not p.exists():
        print(f"[!] Manca {p} — esegui prima src/features_reduction.py")
        return None
    return pd.read_csv(p)


def build_combined_table(df_unc: pd.DataFrame, df_feat: pd.DataFrame, datasets: list) -> pd.DataFrame:
    """
    Per ciascun metodo x dataset di test: % compressione (6.3), delta R@1
    full->compressed (6.3), delta AUPRC full->compressed su logreg (6.2, media
    sui matcher). Un'unica riga risponde a "quanto perdo in retrieval" e
    "quanto perdo in affidabilita' della stima di incertezza" per lo stesso
    taglio di compressione.
    """
    rows = []
    for _, r in df_feat.iterrows():
        method = r["method"]
        # Usa la compressione/recall basata sulla maschera top-K per varianza
        # (mask_topk_variance), non "multilayer_compression_pct": quest'ultima
        # riflette la maschera per ridondanza, che con descrittori VPR poco
        # correlati non rimuove quasi nulla (vedi nota in features_reduction.py).
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
    print("\n=== Combinato 6.3 (recall) + 6.2 (uncertainty) ===")
    print(out.round(4).to_string(index=False))
    return out


def plot_combined(df_combined: pd.DataFrame, dpi: int):
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
        # AUPRC e' in unita' di probabilita' (range tipico 0.001-0.2), mentre
        # R@1 e' in punti percentuali (range tipico 1-11): sulla stessa scala
        # le barre AUPRC risultano invisibili. Le esprimiamo entrambe in
        # "punti percentuali" moltiplicando l'AUPRC delta per 100.
        ax.bar(x - w/2, sub["R@1_delta"], width=w, label="Delta R@1 (6.3)", color=PALETTE[0])
        ax.bar(x + w/2, sub["AUPRC_delta"] * 100, width=w, label="Delta AUPRC logreg x100 (6.2)", color=PALETTE[1])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["dataset"], rotation=20, ha="right")
        ax.set_title(f"{method}  ({sub['compression_pct'].iloc[0]:.1f}% compresso)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Delta in punti percentuali (compressed - full)")
    axes[0].legend()
    fig.suptitle("Effetto della compressione (6.3): retrieval vs qualita' incertezza (6.2)")
    save(fig, "11_combined_compression_vs_uncertainty.png", dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods",  nargs="+", default=["cosplace", "megaloc"])
    p.add_argument("--matchers", nargs="+", default=["superglue", "loftr"])
    p.add_argument("--datasets", nargs="+", default=["sf_xs_test", "tokyo_xs", "svox_sun", "svox_night"])
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("ANALISI 6.2 (Uncertainty) + combinazione con 6.3 (Feature Reduction)")
    print("="*70)

    df_unc = load_uncertainty_summary()
    if df_unc is None:
        return

    # --- 1. Standalone 6.2 ---
    verify_compression_no_degradation(df_unc)
    piv_auprc = pivot_by_variant(df_unc, "auprc")
    pivot_by_variant(df_unc, "r2")
    print("\n=== AUPRC medio per metodo x variante ===")
    print(piv_auprc.round(3).to_string())
    plot_auprc_by_variant(piv_auprc, args.dpi)

    # --- 2. Sparsification Error ---
    df_sparse = build_sparsification_error_table(args.methods, args.matchers, args.datasets)
    plot_sparsification_error(df_sparse, args.dpi)

    # --- 3. Combinazione con 6.3 ---
    df_feat = load_feature_reduction_summary()
    if df_feat is not None:
        df_combined = build_combined_table(df_unc, df_feat, args.datasets)
        plot_combined(df_combined, args.dpi)

    print(f"\nTutte le tabelle in {ANALYSIS_DIR.relative_to(ROOT)}/, le figure in {PLOTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()