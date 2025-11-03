"""
WSGI entry point for production deployment
Use with: gunicorn wsgi:app
"""

import os
import sys
from pathlib import Path

# Set environment
os.environ.setdefault('ENVIRONMENT', 'production')

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import app

if __name__ == "__main__":
    app.run()
