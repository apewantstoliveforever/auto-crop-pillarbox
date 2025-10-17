#!/bin/bash
echo "=============================="
echo "   Video Crop Automation"
echo "=============================="
echo

# Check python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

PY=python3
echo "Using $PY"

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PY -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing required packages..."
$PY -m pip install --upgrade pip
$PY -m pip install opencv-python numpy tqdm moviepy==1.0.3

# Ask for folder
read -p "Enter path to folder containing videos (leave empty for current folder): " FOLDER_PATH
if [ -z "$FOLDER_PATH" ]; then
    FOLDER_PATH="$PWD"
fi

echo "Running video crop script on: $FOLDER_PATH"
$PY main.py "$FOLDER_PATH"

echo "=============================="
echo "      Task finished!"
echo "=============================="
