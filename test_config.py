#!/usr/bin/env python
"""Test configuration loading."""

import sys
import os
sys.path.insert(0, os.getcwd())

print("\n" + "="*80)
print("TESTING CONFIGURATION LOADING")
print("="*80 + "\n")

# Test 1: Check .env file exists and has the key
print("TEST 1: Checking .env file...")
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        content = f.read()
        if 'GOOGLE_FACT_CHECK_API_KEY' in content:
            print("✓ .env contains GOOGLE_FACT_CHECK_API_KEY")
            # Extract value
            for line in content.split('\n'):
                if 'GOOGLE_FACT_CHECK_API_KEY=' in line:
                    key_value = line.split('=')[1].strip()
                    print(f"  Key found: {key_value[:20]}...")
        else:
            print("✗ .env does NOT contain GOOGLE_FACT_CHECK_API_KEY")
else:
    print("✗ .env file not found")

# Test 2: Check environment variable
print("\nTEST 2: Checking os.getenv()...")
api_key_env = os.getenv('GOOGLE_FACT_CHECK_API_KEY')
if api_key_env:
    print(f"✓ os.getenv('GOOGLE_FACT_CHECK_API_KEY'): {api_key_env[:20]}...")
else:
    print("✗ os.getenv() returned None")

# Test 3: Check settings loading
print("\nTEST 3: Checking settings object...")
from src.config import settings

print(f"✓ settings.google_fact_check_api_key: {settings.google_fact_check_api_key[:20] if settings.google_fact_check_api_key else 'EMPTY'}...")

if settings.google_fact_check_api_key:
    print("\n✅ ALL TESTS PASSED - Config loaded correctly!")
else:
    print("\n❌ CONFIG FAILED - API key not loading!")
    print("\nDEBUG INFO:")
    print(f"  - Settings object type: {type(settings)}")
    print(f"  - Settings attributes: {dir(settings)}")

print("\n" + "="*80 + "\n")
