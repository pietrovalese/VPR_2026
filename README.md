# VPR_2026

Visual Place Recognition (VPR) project for the 2026 course/year.

## Cloning the repository

```bash
git clone --recurse-submodules https://github.com/pietrovalese/VPR_2026.git
```

Or, if you have already cloned the repo:


```bash
git submodule update --init --recursive
```

## Dependencies

Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Download dependencies and datasets

```bash
bash setup_deps.sh
```


## 🛠️ Project structure

```
VPR_2026/
├── setup_deps.sh                     # downloads submodules/dependencies and datasets
├── src/
│   ├── extract_descriptors.py        # step 1 - VPR descriptor extraction
│   ├── knn_evaluation.py             # step 2 - KNN + Recall@N (L2 vs dot)
│   ├── image_matching_evaluation.py  # step 3 - re-ranking with Image Matching
│   ├── features_reduction.py         # extension 6.3 - dimensionality reduction
│   ├── uncertainty_estimation.py     # extension 6.2 - uncertainty estimation
│   ├── analyze_results.py            # retrieval + re-ranking tables/analysis
│   ├── analyze_uncertainty.py        # 6.2 analysis and combination with 6.3
│   └── plot_results.py               # figures and tables for the report
├── deps/                             # external submodules (VPR-methods-evaluation, image-matching-models)
├── data/                             # datasets (GSV-XS, SF-XS, Tokyo-XS, SVOX)
└── logs/
    ├── descriptors/                  # extracted descriptors (.npy) per dataset/method
    ├── results/
    │   ├── predictions/              # saved KNN indices (needed for re-ranking)
    │   └── matching/                 # re-ranking results per dataset/method/matcher combo
    ├── feature_reduction/            # extension 6.3 output
    ├── uncertainty/                  # extension 6.2 output
    ├── analysis/                     # CSV/JSON tables produced by analyze_*.py
    └── plots/                        # .png figures produced by plot_results.py
```

## Usage

### 1. Extract Descriptors

Extracts and saves VPR descriptors for database and query images of each dataset.

**Basic usage:**
```bash
python3 src/extract_descriptors.py
```

**Full options:**
```bash
python3 src/extract_descriptors.py \
    --methods   cosplace megaloc netvlad mixvpr \  # default: all
--datasets  sf_xs_test sf_xs_val tokyo_xs svox_sun svox_night \  # default: all
--batch_size 32 \     # default: 32 — reduce to 4 for netvlad/mixvpr if OOM
--num_workers 4 \     # default: 4
--device auto \       # auto | cpu | cuda | cuda:0
--overwrite           # recompute even if descriptors already exist
```

**Note:** NetVLAD and MixVPR are memory-intensive (4096-dim). If you get CUDA OOM:
```bash
python3 src/extract_descriptors.py --methods netvlad mixvpr --batch_size 8
```

---

### 2. KNN Evaluation

Loads saved descriptors, runs KNN with L2 and dot product metrics, and computes Recall@N.

**Basic usage:**
```bash
python3 src/knn_evaluation.py
```

**Full options:**
```bash
python3 src/knn_evaluation.py \
    --datasets sf_xs_test sf_xs_val tokyo_xs svox_sun svox_night \  # default: all found
--methods cosplace megaloc netvlad mixvpr \                      # default: all found
--metrics l2 dot \            # default: both
--k 20 \                      # number of neighbors to retrieve, default: 20
--recall_values 1 5 10 20 \   # default: 1 5 10 20
--threshold_m 25.0 \          # GPS distance threshold in meters, default: 25.0
--save_predictions            # save KNN indices to logs/results/predictions/ (needed for re-ranking)
```

**Output:**
- `logs/results/recall_table.csv` — summary table
- `logs/results/knn_results.json` — all results in machine-readable format
- `logs/results/predictions/<dataset>_<method>_<metric>_preds.npy` — KNN indices (with --save_predictions)

---

### 3. Image Matching (re-ranking)

Re-ranks the KNN predictions using Image Matching methods (SuperGlue, LoFTR, SuperPoint+LightGlue), counting inliers between the query and each retrieved candidate. Requires KNN predictions saved with `--save_predictions` (step 2).

**Basic usage:**
```bash
python3 src/image_matching_evaluation.py
```

**Full options:**
```bash
python3 src/image_matching_evaluation.py \
    --datasets sf_xs_test sf_xs_val tokyo_xs svox_sun svox_night \  # default: all found
    --methods cosplace megaloc netvlad mixvpr \                      # default: all found
    --matchers superglue loftr superpoint-lg \                       # default: all three
    --metric dot \                 # l2 | dot, default: dot
    --num_preds 20 \               # number of KNN candidates to re-rank, default: 20
    --img_size 512 \               # input image size for matching, default: 512
    --recall_values 1 5 10 \       # default: 1 5 10
    --device auto \                # auto | cpu | cuda, default: auto
    --overwrite                    # recompute even if results already exist
```

**Output:**
- `logs/results/matching/<dataset>_<method>_<matcher>/results.json` — recall before/after, inlier stats, timing
- `logs/results/matching/<dataset>_<method>_<matcher>/inliers.npy`, `per_query_inliers.npy`, `correct_mask.npy`, `reranked_preds.npy`
- `logs/results/matching_summary.csv` — flat summary across all combinations

---

### 4. Feature Reduction (Extension 6.3)

Reduces descriptor dimensionality by discarding rarely-activated/low-variance features and features redundant with others (multi-layer Pearson → Spearman → Mutual Information system), then evaluates the recall/compression trade-off and visualizes discarded/kept features with Grad-CAM. Requires descriptors already extracted for `gsv_xs_train` and the test datasets (step 1).

**Basic usage:**
```bash
python3 src/features_reduction.py
```

**Full options:**
```bash
python3 src/features_reduction.py \
    --methods cosplace megaloc \        # default: cosplace megaloc
    --w1 0.33 --w2 0.33 --w3 0.34 \     # Pearson / Spearman / MI weights in the final redundancy score
    --pearson_prefilter 0.3 \           # layer 1 Pearson prefilter threshold
    --spearman_prefilter 0.3 \          # layer 2 Spearman prefilter threshold
    --score_threshold 0.6 \             # fixed final-score threshold (if omitted, a sweep is run)
    --act_threshold 0.01 \              # fixed activation-rate threshold (used together with --score_threshold/--var_threshold)
    --var_threshold 1e-5 \              # fixed variance threshold
    --recall_values 1 5 10 \            # default: 1 5 10
    --batch_size 32 --num_workers 4 \   # default: 32 / 4
    --device auto \                     # default: auto
    --n_gradcam_imgs 5 \                # number of sample images for Grad-CAM, default: 5
    --skip_multilayer \                 # skip the multi-layer redundancy system, go straight to the top-K curve
    --topk_fractions 0.5 0.25 0.1 \     # K fractions for the size/recall curve
    --overwrite                         # recompute even if results already exist
```

**Output:**
- `logs/feature_reduction/<method>/results.json` — thresholds, compression, recall before/after
- `logs/feature_reduction/<method>/mask_final.npy`, `mask_topk_variance.npy` — boolean feature masks
- `logs/feature_reduction/<method>/threshold_sweep.csv`, `topk_curve_val.csv`, `topk_curve_test.csv`
- `logs/feature_reduction/<method>/gradcam_*.npy` — Grad-CAM heatmaps for kept/removed features
- `logs/feature_reduction/summary.csv` — flat summary across all methods

---

### 5. Uncertainty Estimation (Extension 6.2)

Trains a logistic regressor (on SVOX train, validated on `sf_xs_val`) to predict query correctness from the number of inliers, the L2 distance and the margin to the top-1 neighbor, and compares it against simpler uncertainty signals (n_inliers, l2_dist, margin) using AUPRC, Spearman's ρ, R² and AUSC. Optionally repeats the comparison on the compressed descriptors from Extension 6.3. Requires KNN + Image Matching results (steps 2–3) and, unless `--skip_compressed` is used, the masks from Extension 6.3 (step 4).

**Basic usage:**
```bash
python3 src/uncertainty_estimation.py
```

**Full options:**
```bash
python3 src/uncertainty_estimation.py \
    --methods cosplace megaloc \        # default: cosplace megaloc
    --matchers superglue loftr \        # default: superglue loftr
    --metric dot \                      # l2 | dot, default: dot
    --skip_compressed \                 # skip the comparison against compressed descriptors (6.3)
    --overwrite
```

**Output:**
- `logs/uncertainty/<method>_<matcher>/train_features.csv`, `val_features.csv`, `test_<dataset>_features.csv`
- `logs/uncertainty/<method>_<matcher>/logreg_full.json`, `logreg_compressed.json` — trained model coefficients
- `logs/uncertainty/summary.csv` — AUPRC/ρ/R²/AUSC for every method × matcher × dataset × variant

---

### 6. Results Analysis

Aggregates the outputs of steps 1–5 into report-ready tables.

**Retrieval + re-ranking analysis:**
```bash
python3 src/analyze_results.py --recall_values 1 5 10
```
Reads `recall_table.csv`, `knn_per_query.csv`, `extraction_metrics.csv` and `matching_summary.csv`, and produces the paper-style report table, the inlier↔correctness correlation analysis, the timing summary, the L2-vs-dot comparison and the performance/efficiency trade-off table in `logs/analysis/`.

**Uncertainty analysis (6.2) + combination with feature reduction (6.3):**
```bash
python3 src/analyze_uncertainty.py \
    --methods cosplace megaloc \
    --matchers superglue loftr \
    --datasets sf_xs_test tokyo_xs svox_sun svox_night \
    --dpi 300
```
Reads `logs/uncertainty/summary.csv` and (if available) `logs/feature_reduction/summary.csv`, and produces the sparsification-error table and the combined recall/uncertainty comparison in `logs/uncertainty/analysis/` and `logs/plots/`.

---

### 7. Plots

Generates all figures and paper-ready CSV tables for the report from the outputs of the previous steps.

**Basic usage (all plots):**
```bash
python3 src/plot_results.py
```

**Full options:**
```bash
python3 src/plot_results.py \
    --only 01 05 06 \   # run only the numbered plots matching these prefixes (default: all)
    --dpi 300           # figure resolution, default: 300
```

**Output:** `logs/plots/*.png` (Recall bar charts, L2 vs dot, re-ranking delta, timing trade-off, inlier histograms, top-K variance curve, memory saving, Grad-CAM) and `logs/analysis/table_recall.csv`, `table_timing.csv`.

---

### Typical full run

```bash
# Step 1 — extract descriptors for all models and datasets
python3 src/extract_descriptors.py --batch_size 8

# Step 2 — evaluate KNN and save predictions for re-ranking
python3 src/knn_evaluation.py --save_predictions

# Step 3 — re-rank with Image Matching methods
python3 src/image_matching_evaluation.py

# Step 4 — feature reduction (extension 6.3)
python3 src/features_reduction.py

# Step 5 — uncertainty estimation (extension 6.2)
python3 src/uncertainty_estimation.py

# Step 6 — aggregate results into report tables
python3 src/analyze_results.py
python3 src/analyze_uncertainty.py

# Step 7 — generate all figures for the report
python3 src/plot_results.py
```
