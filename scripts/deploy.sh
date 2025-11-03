#!/bin/bash

# Deployment script for Fake News Detection System

echo "================================"
echo "FAKE NEWS DETECTION - DEPLOYMENT"
echo "================================"

# Check environment
if [ -z "$ENVIRONMENT" ]; then
    export ENVIRONMENT=production
fi

echo "Environment: $ENVIRONMENT"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements-deployment.txt

# Download spacy model
echo "Downloading spacy model..."
python -m spacy download en_core_web_sm

# Create required directories
echo "Creating directories..."
mkdir -p outputs/logs outputs/reports data/cache data/models

# Check environment variables
echo "Checking environment variables..."
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  WARNING: GROQ_API_KEY not set!"
fi

if [ -z "$GOOGLE_FACT_CHECK_API_KEY" ]; then
    echo "⚠️  WARNING: GOOGLE_FACT_CHECK_API_KEY not set!"
fi

echo "✓ Deployment setup complete"
echo ""
echo "To start the server:"
echo "  Development: python app.py"
echo "  Production:  gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app"
echo ""
