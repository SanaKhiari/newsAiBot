import sys
import os
sys.path.insert(0, os.getcwd())

print("\n" + "="*80)
print("TESTING WITH REAL COFFEE CANCER CLAIM")
print("="*80 + "\n")

from src.agents.google_fact_check_agent import GoogleFactCheckAgent
from src.agents.snopes_agent import SnopesAgent

# Test Google FC
print("TEST 1: Google Fact Check with 'coffee cancer risk'...")
gfc = GoogleFactCheckAgent()
results = gfc.search_claims("coffee reduces cancer risk", limit=3)
print(f"✓ Google FC found {len(results)} results")
for i, r in enumerate(results, 1):
    print(f"  {i}. {r['title'][:80]}")
    print(f"     Stance: {r['stance']} ({r['stance_confidence']:.2f})")

# Test Snopes
print("\nTEST 2: Snopes with 'coffee cancer'...")
snopes = SnopesAgent()
results = snopes.search_claims("coffee cancer", limit=3)
print(f"✓ Snopes found {len(results)} results")
for i, r in enumerate(results, 1):
    print(f"  {i}. {r['title'][:80]}")
    print(f"     Stance: {r['stance']} ({r['stance_confidence']:.2f})")

print("\n" + "="*80 + "\n")