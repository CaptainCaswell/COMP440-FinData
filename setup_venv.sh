#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "Done!"