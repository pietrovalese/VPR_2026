#!/usr/bin/env bash

set -e

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

echo "Done."