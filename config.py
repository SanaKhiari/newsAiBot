"""
Flask configuration for different environments
"""

import os
from datetime import timedelta

class Config:
    """Base configuration."""
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = ENVIRONMENT == 'development'
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = DEBUG
    JSON_MIMETYPE = 'application/json'
    
    # Session
    SESSION_COOKIE_SECURE = ENVIRONMENT == 'production'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Upload
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    UPLOAD_FOLDER = './uploads'

class DevelopmentConfig(Config):
    """Development configuration."""
    ENVIRONMENT = 'development'
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    ENVIRONMENT = 'production'
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration."""
    ENVIRONMENT = 'testing'
    DEBUG = True
    TESTING = True
    SESSION_COOKIE_SECURE = False

# Config by environment
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
