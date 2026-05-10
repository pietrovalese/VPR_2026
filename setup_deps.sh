#!/usr/bin/env bash
set -e

# ── Argomenti ────────────────────────────────────────────────────────────────

DOWNLOAD_DATASETS=false

for arg in "$@"; do
    case $arg in
        --dataset) DOWNLOAD_DATASETS=true ;;
        *) echo "Opzione sconosciuta: $arg"; exit 1 ;;
    esac
done

# ── Submodules ───────────────────────────────────────────────────────────────

echo "Initializing git submodules..."
git submodule update --init --recursive

# ── PyTorch ──────────────────────────────────────────────────────────────────

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | tr -d '.')
    if   [ "$CUDA_VERSION" -ge 124 ] 2>/dev/null; then TORCH_INDEX="cu124"
    elif [ "$CUDA_VERSION" -ge 121 ] 2>/dev/null; then TORCH_INDEX="cu121"
    else TORCH_INDEX="cu118"
    fi
    echo "GPU NVIDIA rilevata — installazione PyTorch con CUDA ${TORCH_INDEX}..."
    pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
else
    echo "Nessuna GPU NVIDIA rilevata — installazione PyTorch CPU-only..."
    pip install torch torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/cpu"
fi

# ── Dipendenze Python ────────────────────────────────────────────────────────

echo "Installing image-matching-models..."
pip install -e deps/image-matching-models[all]

echo "Installing faiss-cpu..."
pip install faiss-cpu

# ── Dataset ──────────────────────────────────────────────────────────────────

if [ "$DOWNLOAD_DATASETS" = true ]; then
    # aggiorna gdown (IMPORTANTE)
    pip install -U "gdown>=4.7"

    # crea cartella dati
    mkdir -p data

    echo "Downloading full folder from Google Drive..."
    # scarica tutta la cartella
    gdown --folder "https://drive.google.com/drive/folders/1Ucy9JONT26EjDAjIJFhuL9qeLxgSZKmf" -O data

    echo "Downloading single file..."
    # scarica il file singolo
    gdown "https://drive.google.com/uc?id=16iuk8voW65GaywNUQlWAbDt6HZzAJ_t9" -O data/svox.zip

    echo "Unzipping..."
    # unzip automatico di tutti gli zip
    for f in data/*.zip; do
        unzip -o "$f" -d data
        rm "$f"
    done
else
    echo "Skipping dataset download (usa --dataset per scaricarli)."
fi

echo "Done."