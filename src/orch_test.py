import sys
import os
sys.path.insert(0, os.getcwd())

print("\n" + "="*80)
print("TESTING PRIORITY-BASED EVIDENCE RETRIEVAL")
print("="*80 + "\n")

# Test 1: Import all agents
print("TEST 1: Importing agents...")
try:
    from src.agents.google_fact_check_agent import GoogleFactCheckAgent
    print("✓ GoogleFactCheckAgent imported")
except Exception as e:
    print(f"✗ GoogleFactCheckAgent failed: {e}")

try:
    from src.agents.snopes_agent import SnopesAgent
    print("✓ SnopesAgent imported")
except Exception as e:
    print(f"✗ SnopesAgent failed: {e}")

try:
    from src.agents.evidence_retrieval_agent import EvidenceRetrievalAgent
    print("✓ EvidenceRetrievalAgent imported")
except Exception as e:
    print(f"✗ EvidenceRetrievalAgent failed: {e}")

# Test 2: Check configuration
print("\nTEST 2: Checking configuration...")
from src.config import settings
print(f"✓ Google FC Enabled: {settings.google_fact_check_enabled}")
print(f"✓ Snopes Enabled: {settings.snopes_enabled}")
print(f"✓ GDELT Enabled: {settings.gdelt_enabled}")
print(f"✓ Evidence Source Priority: {settings.evidence_source_priority}")

# Test 3: Test Google Fact Check API
print("\nTEST 3: Testing Google Fact Check API...")
try:
    gfc = GoogleFactCheckAgent()
    if gfc.available:
        results = gfc.search_claims("vaccines safe", limit=2)
        print(f"✓ Google FC API working: {len(results)} results")
    else:
        print("⚠ Google FC API key not configured")
except Exception as e:
    print(f"⚠ Google FC API test failed: {e}")

# Test 4: Test Snopes
print("\nTEST 4: Testing Snopes...")
try:
    snopes = SnopesAgent()
    results = snopes.search_claims("moon landing fake", limit=1)
    print(f"✓ Snopes working: {len(results)} results")
except Exception as e:
    print(f"⚠ Snopes test failed: {e}")

print("\n" + "="*80)
print("TESTS COMPLETE")
print("="*80 + "\n")