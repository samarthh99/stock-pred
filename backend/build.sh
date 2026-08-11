#!/bin/bash
set -e  # Exit on any error

echo "Installing Git LFS..."
apt-get update
apt-get install -y git-lfs

echo "Initializing Git LFS..."
git lfs install

echo "Pulling LFS files..."
git lfs pull

echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "Build completed successfully!"