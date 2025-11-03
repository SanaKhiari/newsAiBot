"""
Shared utility functions for the fake news detection pipeline.
"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
from loguru import logger


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text for caching."""
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_url(url: str) -> str:
    """Normalize URL for consistent caching."""
    url = url.lower().strip()
    url = re.sub(r'^https?://(www\.)?', '', url)
    url = re.sub(r'/$', '', url)
    return url


def time_decay_weight(publish_date: Optional[datetime], decay_days: int = 30) -> float:
    """
    Calculate time decay weight for evidence based on publication date.
    Uses exponential decay: exp(-days_old / decay_days)
    
    Args:
        publish_date: Publication datetime
        decay_days: Half-life in days
        
    Returns:
        Weight between 0.1 and 1.0
    """
    if not publish_date:
        return 0.5  # Default weight for unknown dates
    
    days_old = (datetime.now() - publish_date).days
    if days_old < 0:
        days_old = 0
    
    weight = math.exp(-days_old / decay_days)
    return max(weight, 0.1)  # Minimum weight of 0.1


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    pattern = r'(?:https?://)?(?:www\.)?([^/]+)'
    match = re.search(pattern, url)
    return match.group(1) if match else ""


def get_source_weight(url: str, source_weights: Dict[str, float]) -> float:
    """
    Get reliability weight for source based on domain.
    
    Args:
        url: Source URL
        source_weights: Dictionary mapping domain keywords to weights
        
    Returns:
        Weight between 0.0 and 1.0
    """
    domain = extract_domain(url).lower()
    
    for source_key, weight in source_weights.items():
        if source_key in domain:
            return weight
    
    return 0.5  # Default weight for unknown sources


def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Create safe filename from text."""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:max_length].strip('-')


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name: str):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        """End timing and log."""
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"✓ {self.name} completed in {self.elapsed:.2f}s")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_nested(data: Dict, keys: List[str], default: Any = None) -> Any:
    """Safely get nested dictionary value."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
    return data if data is not None else default
