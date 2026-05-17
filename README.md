# VPR_2026

Progetto relativo a Visual Place Recognition (VPR) per il corso/anno 2026.

## 📦 Clonare la repository

```bash
git clone --recurse-submodules https://github.com/pietrovalese/VPR_2026.git
```

Oppure, se hai già clonato la repo:


```bash
git submodule update --init --recursive
```

## Dependecies

Crea l'ambiente virtuale

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Scarica dipendenze e dataset

```bash
bash setup_deps.sh
```


## 🛠️ Struttura del progetto



## Usage

### 1. Extract Descriptors

Extracts and saves VPR descriptors for database and query images of each dataset.

**Basic usage:**
```bash
python src/extract_descriptors.py
```

**Full options:**
```bash
python src/extract_descriptors.py \
    --methods   cosplace megaloc netvlad mixvpr \  # default: all
    --datasets  sf_xs_test sf_xs_val tokyo_xs svox_sun svox_night \  # default: all
    --batch_size 32 \     # default: 32 — reduce to 4 for netvlad/mixvpr if OOM
    --num_workers 4 \     # default: 4
    --device auto \       # auto | cpu | cuda | cuda:0
    --overwrite           # recompute even if descriptors already exist
```

**Note:** NetVLAD and MixVPR are memory-intensive (4096-dim). If you get CUDA OOM:
```bash
python src/extract_descriptors.py --methods netvlad mixvpr --batch_size 8
```

---

### 2. KNN Evaluation

Loads saved descriptors, runs KNN with L2 and dot product metrics, and computes Recall@N.

**Basic usage:**
```bash
python src/knn_evaluation.py
```

**Full options:**
```bash
python src/knn_evaluation.py \
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

### Typical full run

```bash
# Step 1 — extract descriptors for all models and datasets
python3 src/extract_descriptors.py --batch_size 8

# Step 2 — evaluate KNN and save predictions for re-ranking
python3 src/knn_evaluation.py --save_predictions
```

