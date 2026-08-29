"""
src/uncertainty_estimation.py — Estensione 6.2 (Uncertainty Estimation),
integrata con l'estensione 6.3 (Memory saving / compressione feature).

Idea dell'integrazione
-----------------------
La 6.2 chiede di stimare, in modo post-hoc, quanto ci si può fidare della
predizione R@1 di un metodo VPR. La 6.3 comprime i descrittori scartando
dimensioni ridondanti o poco informative. Domanda naturale che lega le due
estensioni: **la compressione peggiora la qualità della stima di incertezza?**

Per rispondere, ogni misura di incertezza basata sul descrittore (distanza
al 1° vicino, margine 1°-2° vicino, e il regressore logistico che le usa)
viene calcolata due volte:
    - "full"       -> descrittori originali (D dimensioni)
    - "compressed" -> descrittori mascherati con mask_topk_variance.npy
                       prodotta da src/features_reduction.py (n_kept dimensioni,
                       selezione top-K per varianza — vedi nota nel codice sul
                       perché non si usa mask_final.npy)
Il numero di inlier da image matching non dipende dal descrittore, quindi
resta identico nei due casi (serve da controllo/baseline).

Note metodologiche (fix applicati)
------------------------------------
    - Le feature [n_inliers, l2_dist, margin] vivono su scale molto diverse
      (n_inliers: decine-centinaia; l2_dist/margin: range ~[-1,1]). Senza
      standardizzazione, la penalità L2 del regressore logistico penalizza
      sproporzionatamente le feature a piccola scala. Si usa uno
      StandardScaler fittato sul training set, applicato coerentemente a
      train/val/test.
    - La variante "compressed" viene valutata contro il proprio ground
      truth (correttezza e distanza geografica del retrieval RICALCOLATO
      sui descrittori mascherati), non contro quello del retrieval "full".
      Altrimenti si misurerebbe "il segnale compresso predice la
      correttezza del sistema full", che è una domanda diversa da "il
      segnale compresso predice la correttezza del sistema compresso".

Misure di incertezza implementate
----------------------------------
    1. n_inliers        : inlier tra query e 1° retrieved (richiede matching)     [23]
    2. l2_dist_top1      : "distanza L2 in feature space" al 1° vicino            [18]
    3. margin_top1_top2  : score(1°) - score(2°)  (ambiguità/PA-score-like)
    4. logreg            : regressore logistico su [n_inliers, l2_dist, margin]
                            -> P(query corretta). Allenato SOLO sui training set.

Split usati (si veda il testo del progetto, Sez. 6.2)
------------------------------------------------------
    train : svox_sun_train, svox_night_train  (uniche vere "training sets"
            disponibili oltre a GSV-XS, che è escluso come nella 6.1)
    val   : sf_xs_val   (selezione iperparametri: C del logistic regressor)
    test  : sf_xs_test, tokyo_xs, svox_sun, svox_night

Metriche di valutazione
------------------------
    - AUPRC                         (richiesta esplicitamente dal testo)
    - Spearman's rho                (score incertezza vs errore geo continuo)
    - R^2 (coefficient of determination), via regressione lineare score->errore
    - AUSC (Area Under Sparsification Curve): errore medio residuo rimuovendo
      progressivamente le query più incerte; più basso = incertezza più utile

Prerequisiti (in ordine)
-------------------------
    python src/extract_descriptors.py --datasets svox_sun_train svox_night_train sf_xs_val sf_xs_test tokyo_xs svox_sun svox_night --methods cosplace megaloc
    python src/knn_evaluation.py --metrics dot
    python src/image_matching_evaluation.py --datasets svox_sun_train svox_night_train sf_xs_test tokyo_xs svox_sun svox_night --methods cosplace megaloc
    python src/features_reduction.py --methods cosplace megaloc   # per mask_final.npy

Output
------
    logs/uncertainty/<method>_<matcher>/
        train_features.csv, val_features.csv, test_<dataset>_features.csv
        logreg_full.json, logreg_compressed.json
        metrics_summary.csv
        sparsification_<dataset>_<variant>.csv
    logs/uncertainty/summary.csv

Uso
---
    python src/uncertainty_estimation.py --methods cosplace megaloc --matchers superglue loftr
    python src/uncertainty_estimation.py --methods cosplace --matchers superglue --skip_compressed
"""

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
# Path setup — stesse convenzioni degli altri script della pipeline
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
# Caricamento feature "full" (già calcolate da knn_evaluation / image_matching)
# ---------------------------------------------------------------------------
def load_full_features(dataset: str, method: str, matcher: str, metric: str) -> pd.DataFrame | None:
    """
    Combina, per un dataset, i risultati già salvati da knn_evaluation.py e
    image_matching_evaluation.py in un'unica tabella per-query:
        query_idx, geo_dist_m, correct, top1_score, margin, n_inliers
    Non ricalcola nulla: se un file manca, ritorna None con un log esplicativo.
    """
    scores_path = PREDS_DIR / f"{dataset}_{method}_{metric}_scores.npy"
    if not scores_path.exists():
        log.warning(f"  [{dataset}/{method}] manca {scores_path.name} — esegui knn_evaluation.py")
        return None
    if not KNN_PQ_CSV.exists():
        log.warning("  manca knn_per_query.csv — esegui knn_evaluation.py")
        return None

    scores = np.load(scores_path)  # (N_q, K) — score decrescente per bontà se metric='dot'
    if scores.shape[1] < 2:
        log.warning(f"  [{dataset}/{method}] servono almeno 2 vicini per il margine")
        return None

    pq = pd.read_csv(KNN_PQ_CSV)
    pq = pq[(pq["dataset"] == dataset) & (pq["method"] == method) & (pq["metric"] == metric)].reset_index(drop=True)
    if len(pq) != scores.shape[0]:
        log.warning(f"  [{dataset}/{method}] disallineamento righe knn_per_query.csv vs scores.npy — skip")
        return None

    match_dir = MATCH_DIR / f"{dataset}_{method}_{matcher}"
    inliers_path = match_dir / "per_query_inliers.npy"
    if not inliers_path.exists():
        log.warning(f"  [{dataset}/{method}/{matcher}] manca per_query_inliers.npy — esegui image_matching_evaluation.py")
        return None
    n_inliers = np.load(inliers_path)
    if len(n_inliers) != len(pq):
        log.warning(f"  [{dataset}/{method}/{matcher}] disallineamento righe inliers vs knn — skip")
        return None

    # Per metric='dot' punteggio più alto = più simile; per 'l2' più basso = più simile.
    # Normalizziamo tutto in "direzione incertezza": valore più alto = più incerto.
    if metric == "dot":
        top1_score = scores[:, 0]
        margin     = scores[:, 0] - scores[:, 1]
        l2_dist    = -scores[:, 0]      # proxy: score alto -> "distanza" bassa
    else:  # l2: distanza vera, score basso = migliore
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
# Feature "compressed" — richiede un mini re-KNN su descrittori mascherati
# ---------------------------------------------------------------------------
def compute_compressed_l2_and_margin(dataset: str, method: str) -> pd.DataFrame | None:
    """
    Ricalcola, usando SOLO le dimensioni tenute da mask_topk_variance.npy
    (estensione 6.3): distanza al 1° vicino, margine 1°-2°, e — a differenza
    della versione precedente — anche la correttezza (R@1, soglia 25m) e la
    distanza geografica del top-1 EFFETTIVAMENTE ritrovato dal retrieval
    compresso. Riusa i descrittori già estratti da extract_descriptors.py
    (normalizzati), non richiama il modello.

    Perché ricalcolare anche "correct"/"geo_dist_m": il top-1 ritrovato può
    cambiare quando si maschera il descrittore (coerente con il ΔR@1 non
    nullo osservato nella 6.3). Se si riusa il "correct" del retrieval full
    per valutare un segnale di incertezza calcolato sul retrieval compressed,
    si sta di fatto chiedendo "questo segnale compresso predice la
    correttezza del sistema FULL?" — una domanda diversa (e meno rilevante
    per un sistema compresso davvero distribuito) da "predice la propria
    correttezza?". Qui rispondiamo alla seconda.

    Ritorna un DataFrame con colonne
    [l2_dist_top1_comp, margin_comp, correct_comp, geo_dist_m_comp],
    oppure None se mancano i prerequisiti.
    """
    mask_path = REDUCTION_DIR / method / "mask_topk_variance.npy"
    if not mask_path.exists():
        log.warning(f"  [{method}] manca mask_topk_variance.npy — esegui features_reduction.py prima "
                    f"(versione aggiornata che salva anche questa maschera)")
        return None
    mask = np.load(mask_path)

    desc_dir = DESC_DIR / dataset / method
    db_path = desc_dir / "database_descriptors.npy"
    q_path  = desc_dir / "query_descriptors.npy"
    db_paths_path = desc_dir / "database_paths.npy"
    q_paths_path  = desc_dir / "query_paths.npy"
    if not db_path.exists() or not q_path.exists():
        log.warning(f"  [{dataset}/{method}] mancano descrittori normalizzati — esegui extract_descriptors.py")
        return None
    if not db_paths_path.exists() or not q_paths_path.exists():
        log.warning(f"  [{dataset}/{method}] mancano i path (per le coordinate UTM) — esegui extract_descriptors.py")
        return None

    db = np.load(db_path)[:, mask].astype(np.float32)
    q  = np.load(q_path)[:, mask].astype(np.float32)

    # Ri-normalizza dopo il masking (le norme non sono più 1 una volta tolte dimensioni)
    db_n = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-12)
    q_n  = q  / (np.linalg.norm(q,  axis=1, keepdims=True) + 1e-12)

    index = faiss.IndexFlatIP(db_n.shape[1])
    index.add(db_n)
    scores, indices = index.search(q_n, 2)  # top-2: basta per il margine

    l2_dist = -scores[:, 0]
    margin  = scores[:, 0] - scores[:, 1]

    # Correttezza e distanza geo del top-1 RICALCOLATO (stessa logica/soglia
    # di knn_evaluation.py::compute_recall_and_per_query, GPS_THRESHOLD_M=25m)
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
    """Tabella completa per-query: colonne full + (se disponibili) colonne compressed."""
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
            log.warning(f"  [{dataset}/{method}] feature compresse non disponibili per questo run")

    return df


# ---------------------------------------------------------------------------
# Metriche di valutazione dell'incertezza
# ---------------------------------------------------------------------------
def evaluate_uncertainty_score(unc_score: np.ndarray, correct: np.ndarray,
                                geo_dist_m: np.ndarray, has_gps: np.ndarray) -> dict:
    """
    unc_score: valore più alto = più incerto (già orientato dal chiamante).
    Ritorna AUPRC (target = query sbagliata), Spearman rho vs errore geo,
    R^2 di un fit lineare score->errore, e AUSC.
    """
    unc_score = np.asarray(unc_score, dtype=np.float64)
    is_wrong  = ~np.asarray(correct, dtype=bool)

    out = {}

    # AUPRC: positivo = "query sbagliata", score = incertezza
    if is_wrong.any() and (~is_wrong).any():
        out["auprc"] = float(average_precision_score(is_wrong, unc_score))
    else:
        out["auprc"] = float("nan")

    # Spearman e R^2 solo su query con GPS valido
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

    # AUSC: rimuovi progressivamente le query più incerte, guarda l'errore medio residuo
    out["ausc"] = area_under_sparsification_curve(unc_score, geo_dist_m, has_gps)

    return out


def area_under_sparsification_curve(unc_score: np.ndarray, geo_dist_m: np.ndarray,
                                      has_gps: np.ndarray, n_steps: int = 20) -> float:
    """
    Ordina le query per incertezza decrescente e le rimuove a fette (0%, 5%, ..., 95%).
    A ogni step calcola l'errore medio (in metri) sulle query rimanenti.
    AUSC = area sotto la curva errore-medio vs frazione-rimossa (trapezio, asse x in [0,1]).
    Valore più basso = l'incertezza individua bene le query con errore alto
    (rimuoverle fa scendere rapidamente l'errore medio residuo).
    """
    mask_gps = np.asarray(has_gps, dtype=bool) & ~np.isnan(geo_dist_m)
    if mask_gps.sum() < n_steps:
        return float("nan")

    score = np.asarray(unc_score)[mask_gps]
    err   = np.asarray(geo_dist_m)[mask_gps]
    order = np.argsort(-score)  # più incerto prima
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
# Regressore logistico (train su SVOX train, val su sf_xs_val)
# ---------------------------------------------------------------------------
def train_logreg(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str],
                  label_col: str = "correct") -> tuple:
    """
    Allena LogisticRegression su train_df, seleziona C per miglior AUPRC su val_df.
    Le feature vivono su scale molto diverse (n_inliers: decine-centinaia;
    l2_dist/margin: range ~[-1,1]): senza standardizzazione la penalità L2
    penalizzerebbe sproporzionatamente le feature a piccola scala, facendo
    collassare il modello quasi su n_inliers da solo. Lo scaler è fittato SOLO
    sul training set e poi riapplicato (mai rifittato) a val/test.
    label_col: colonna di correttezza da usare come target — "correct" (full)
    o "correct_comp" (compressed, ricalcolata sul retrieval mascherato).
    Ritorna (modello_scelto, scaler, best_C, val_auprc).
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
        p_wrong = 1.0 - model.predict_proba(X_val)[:, 1]  # prob. che sia sbagliata
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
    p = argparse.ArgumentParser()
    p.add_argument("--methods",  nargs="+", default=["cosplace", "megaloc"])
    p.add_argument("--matchers", nargs="+", default=["superglue", "loftr"])
    p.add_argument("--metric",   type=str, default="dot", choices=["l2", "dot"])
    p.add_argument("--skip_compressed", action="store_true",
                    help="Salta il confronto con i descrittori compressi (6.3)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for method in args.methods:
        for matcher in args.matchers:
            log.info(f"\n{'='*70}\n[COMBO] method={method}  matcher={matcher}\n{'='*70}")
            run_dir = OUT_DIR / f"{method}_{matcher}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # --- Train (SVOX train, sun+night concatenati) ---
            train_frames = []
            for ds in TRAIN_DATASETS:
                df = build_feature_table(ds, method, matcher, args.metric, args.skip_compressed)
                if df is not None:
                    df["train_source"] = ds
                    train_frames.append(df)
            if not train_frames:
                log.warning(f"  Nessun dato di training disponibile per {method}/{matcher} — skip combo")
                continue
            train_df = pd.concat(train_frames, ignore_index=True)
            train_df.to_csv(run_dir / "train_features.csv", index=False)

            # --- Val (sf_xs_val) ---
            val_df = build_feature_table(VAL_DATASET, method, matcher, args.metric, args.skip_compressed)
            if val_df is None:
                log.warning(f"  Val set non disponibile per {method}/{matcher} — skip combo")
                continue
            val_df.to_csv(run_dir / "val_features.csv", index=False)

            # --- Allena logreg "full" ---
            feat_full = ["n_inliers", "l2_dist_top1", "margin"]
            model_full, scaler_full, c_full, auprc_full = train_logreg(
                train_df, val_df, feat_full, label_col="correct")
            log.info(f"  logreg FULL: best C={c_full}  val AUPRC={auprc_full:.4f}")

            # --- Allena logreg "compressed", se le colonne esistono ---
            # NOTA: n_inliers resta identico a full/compressed di proposito (non dipende
            # dal descrittore). Così il confronto full vs compressed isola SOLO l'effetto
            # della compressione sulle feature derivate dal descrittore, invece di
            # confondere "la compressione fa male" con "togliere gli inlier fa male".
            # Il target è "correct_comp" (correttezza del retrieval RICALCOLATO sui
            # descrittori mascherati), non "correct" (full) — altrimenti staremmo
            # allenando il modello compressed a predire gli errori di un sistema diverso.
            model_comp, scaler_comp, c_comp = None, None, None
            has_comp_cols = "l2_dist_top1_comp" in train_df.columns and "margin_comp" in train_df.columns
            if not args.skip_compressed and has_comp_cols:
                feat_comp = ["n_inliers", "l2_dist_top1_comp", "margin_comp"]
                model_comp, scaler_comp, c_comp, auprc_comp = train_logreg(
                    train_df, val_df, feat_comp, label_col="correct_comp")
                log.info(f"  logreg COMPRESSED: best C={c_comp}  val AUPRC={auprc_comp:.4f}")

            # --- Valutazione sui test set ---
            for ds in TEST_DATASETS:
                test_df = build_feature_table(ds, method, matcher, args.metric, args.skip_compressed)
                if test_df is None:
                    continue
                test_df.to_csv(run_dir / f"test_{ds}_features.csv", index=False)

                # variant_name -> (score, colonna "correct" da usare, colonna "geo_dist_m" da usare)
                # Le varianti "full" (n_inliers, l2_dist, margin, logreg_full) si valutano
                # contro il ground truth del retrieval full; le varianti "compressed" contro
                # il ground truth del retrieval RICALCOLATO sui descrittori mascherati.
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
        log.info(f"\nSalvato {OUT_DIR / 'summary.csv'}  ({len(df_summary)} righe)")
    else:
        log.warning("Nessun risultato prodotto — controlla i prerequisiti nel docstring del file.")


if __name__ == "__main__":
    main()