#!/usr/bin/env python3
"""Verify all components for Groq + spaCy setup."""

import sys
from pathlib import Path
import os

# ✅ ADD THIS: Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print("✓ Python version: {}.{}.{}".format(version.major, version.minor, version.micro))
        return True
    print("✗ Python version too old: {}.{}.{}".format(version.major, version.minor, version.micro))
    return False


def check_imports():
    """Check if required packages are installed."""
    packages = [
        ('fastapi', 'FastAPI'),
        ('langchain', 'LangChain'),
        ('langgraph', 'LangGraph'),
        ('groq', 'Groq'),
        ('spacy', 'spaCy'),
        ('sentence_transformers', 'Sentence Transformers'),  # ✅ FIXED: was 'sentence-transformers'
        ('faiss', 'FAISS'),
        ('shap', 'SHAP'),
        ('lime', 'LIME'),
        ('newspaper', 'Newspaper3k'),  # ✅ FIXED: was 'newspaper3k'
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError as e:
            print(f"✗ {name} NOT installed: {e}")
            all_good = False
    
    return all_good


def check_spacy_model():
    """Check if spaCy model is downloaded."""
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        
        # Test basic functionality
        doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
        entities = len(doc.ents)
        tokens = len(doc)
        
        print(f"✓ spaCy model 'en_core_web_sm' loaded")
        print(f"  - Tokens: {tokens}, Entities: {entities}")
        return True
    except Exception as e:
        print(f"✗ spaCy model 'en_core_web_sm' NOT found: {e}")
        return False


def check_groq_api():
    """Check if Groq API is configured."""
    try:
        from src.config import settings
        
        if not settings.groq_api_key:
            print("✗ Groq API key not configured in .env")
            print("  Please add: GROQ_API_KEY=gsk_xxxxx")
            return False
        
        if settings.groq_api_key == "gsk_your_actual_api_key_here":
            print("✗ Groq API key is placeholder - please update .env")
            return False
        
        # Try to initialize client
        from src.llm.groq_client import GroqClient
        client = GroqClient()
        print("✓ Groq API client initialized successfully")
        print(f"  - Model: {settings.groq_model}")
        return True
        
    except ValueError as e:
        print(f"✗ Groq API configuration error: {e}")
        return False
    except Exception as e:
        print(f"✗ Groq API connection failed: {e}")
        return False


def check_directories():
    """Check if required directories exist."""
    dirs = [
        'data/cache',
        'data/models',
        'outputs/reports',
        'outputs/logs',
        'src/agents',
        'src/llm',
        'src/orchestration'
    ]
    
    all_good = True
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_good = False
    
    return all_good


def check_config():
    """Check if .env file exists and is valid."""
    if not Path('.env').exists():
        print("✗ Configuration file (.env) NOT found")
        return False
    
    print("✓ Configuration file (.env) exists")
    
    # Check for required keys
    with open('.env', 'r') as f:
        env_content = f.read()
    
    required_keys = [
        'GROQ_API_KEY',
        'CLAIM_DETECTOR_MODEL=en_core_web_sm',
    ]
    
    all_good = True
    for key in required_keys:
        if key.split('=')[0] in env_content:
            print(f"  ✓ {key.split('=')[0]} configured")
        else:
            print(f"  ✗ {key.split('=')[0]} NOT configured")
            all_good = False
    
    return all_good


def test_imports():
    """Test actual imports to verify they work."""
    print("\n[Additional Import Tests]")
    
    tests = []
    
    # Test sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Sentence Transformers can load models")
        tests.append(True)
    except Exception as e:
        print(f"✗ Sentence Transformers model load failed: {e}")
        tests.append(False)
    
    # Test newspaper
    try:
        from newspaper import Article
        print("✓ Newspaper3k Article class available")
        tests.append(True)
    except Exception as e:
        print(f"✗ Newspaper3k import failed: {e}")
        tests.append(False)
    
    return all(tests)


def main():
    """Run all checks."""
    print("\n==== Fake News Detection Setup Verification ====")
    print("(Using Groq API + spaCy)\n")
    
    checks = [
        ("Python Version", check_python_version()),
        ("Python Packages", check_imports()),
        ("spaCy Model", check_spacy_model()),
        ("Groq API", check_groq_api()),
        ("Directories", check_directories()),
        ("Configuration", check_config()),
        ("Import Tests", test_imports()),  # ✅ ADDED: Additional import tests
    ]
    
    print("\n==== Summary ====\n")
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! System is ready.")
        print("\nNext steps:")
        print("1. Create source code files (remaining agents)")
        print("2. Run: uvicorn src.api.server:app --reload")
        print("3. Test: curl http://localhost:8000/health")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
