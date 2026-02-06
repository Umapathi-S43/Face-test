#!/bin/bash
# Face Swap Application - Start Script
# Run this script to start the Face Swap application

echo "🎭 Face Swap Application Starter"
echo "================================"

# Navigate to the app directory
cd "$(dirname "$0")"

# Activate the virtual environment
source ~/face-swap/facefusion/venv/bin/activate

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated"
echo "📁 Working directory: $(pwd)"

# Run the application
python run.py
