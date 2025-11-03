"""Configuration management for Fake News Detection."""

from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
from datetime import datetime


class Settings(BaseSettings):
    """Application settings."""
    
    # ========== ENVIRONMENT ==========
    environment: str = "development"
    log_level: str = "info"
    
    # ========== API ==========
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # ========== LLM - GROQ ==========
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    
    # ========== MODELS ==========
    claim_detector_model: str = "en_core_web_sm"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_sequence_length: int = 512
    
    # ========== FACT-CHECKING APIs ==========
    # Google Fact Check API - CRITICAL: Must match .env variable name exactly
    google_fact_check_api_key: str = ""
    google_fact_check_enabled: bool = True
    
    # Snopes
    snopes_enabled: bool = True
    snopes_timeout: int = 15
    
    # GDELT
    gdelt_api_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    gdelt_enabled: bool = True
    gdelt_timeout: int = 30
    
    # Evidence retrieval priority
    evidence_source_priority: List[str] = [
        "google_fact_check",
        "snopes",
        "pubmed",
        "gdelt"
    ]
    
    # ========== EVIDENCE RETRIEVAL ==========
    commoncrawl_api_url: str = "https://index.commoncrawl.org/CC-MAIN-2024-10-index"
    max_evidence_per_claim: int = 10
    evidence_timeout: int = 30
    request_delay: float = 1.0
    
    # ========== FAISS ==========
    faiss_index_path: str = "./data/models/faiss_index"
    top_k_evidence: int = 5
    
    # ========== REDIS ==========
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600
    
    # ========== AGGREGATION ==========
    source_weights: Dict[str, float] = {
        "reuters": 1.0,
        "ap": 1.0,
        "bbc": 0.9,
        "nyt": 0.9,
        "cnn": 0.8,
        "fox": 0.8,
        "twitter": 0.3
    }
    time_decay_days: int = 30
    verdict_threshold: float = 0.3
    
    # ========== EXPLAINABILITY ==========
    shap_num_samples: int = 100
    lime_num_features: int = 10
    generate_html_report: bool = True
    enable_llm_explanations: bool = True
    
    # ========== LANGCHAIN ==========
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "fake-news-detection"
    
    # ========== PATHS ==========
    data_dir: str = "./data"
    cache_dir: str = "./data/cache"
    model_dir: str = "./data/models"
    output_dir: str = "./outputs"
    log_dir: str = "./outputs/logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False  # Important: allows both GOOGLE_FACT_CHECK_API_KEY and google_fact_check_api_key
        extra = "ignore"  # Ignore extra fields from .env


# Create singleton settings instance
settings = Settings()

# Debug: Print loaded settings (remove after testing)
if settings.environment == "development":
    import sys
    if "test" in sys.argv[0] or "test" in " ".join(sys.argv):
        print(f"\n[CONFIG DEBUG] Google FC API Key loaded: {'Yes' if settings.google_fact_check_api_key else 'No'}")
        if settings.google_fact_check_api_key:
            print(f"[CONFIG DEBUG] API Key preview: {settings.google_fact_check_api_key[:20]}...")
