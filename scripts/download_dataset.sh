#!/bin/bash
# Download Social-IQ 2.0 dataset from Google Drive
# Usage: bash scripts/download_dataset.sh

set -e

FOLDER_ID="1QEQo1meL7l_PcSUZ50fy0tL6JFKWh9ay"
OUTPUT_DIR="datasets/socialiq2/siq2"

# Install gdown if not available
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

mkdir -p "$OUTPUT_DIR"

echo "Downloading Social-IQ dataset to $OUTPUT_DIR..."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$OUTPUT_DIR" --remaining-ok

echo ""
echo "Download complete. Contents:"
ls -R "$OUTPUT_DIR"
