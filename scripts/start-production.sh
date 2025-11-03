#!/bin/bash

# Production startup script

export ENVIRONMENT=production
export LOG_LEVEL=info

echo "Starting Fake News Detection System (Production)..."
echo "Listening on 0.0.0.0:8000"

gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --threads 2 \
    --worker-class gthread \
    --timeout 120 \
    --access-logfile ./outputs/logs/access.log \
    --error-logfile ./outputs/logs/error.log \
    --log-level info \
    wsgi:app
