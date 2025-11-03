"""
LLM-based Stance Classifier Agent using Groq API.
Zero-shot classification - no training required.
"""

from typing import List, Dict, Tuple
import json
from loguru import logger

from src.llm.groq_client import GroqClient
from src.llm.prompts import STANCE_CLASSIFICATION_SYSTEM, STANCE_CLASSIFICATION_PROMPT
from src.config import settings



class LLMStanceAgent:
    """LLM-based stance classification using Groq (NOT a BaseTool)."""
    
    def __init__(self):
        """Initialize LLM stance classifier."""
        self.llm_client = GroqClient()
        logger.info("LLM Stance Classifier (Groq) initialized")
    
    def classify_stance_batch(self, claims: List[Dict], timeout_per_batch: int = 120) -> List[Dict]:
        """
        Classify stance with global timeout per batch (no hanging)
        
        Args:
            claims: List of claims with evidence
            timeout_per_batch: Maximum time for entire batch (seconds)
        """
        import time
        
        logger.info("Classifying stance with Groq LLM for all evidence")
        
        start_time = time.time()
        total_classifications = 0
        failed_classifications = 0
        skipped_classifications = 0
        
        for claim_idx, claim in enumerate(claims):
            try:
                # Check global timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_per_batch:
                    logger.warning(
                        f"Batch timeout reached ({elapsed:.1f}s > {timeout_per_batch}s). "
                        f"Skipping remaining {len(claims) - claim_idx} claims."
                    )
                    # Skip remaining claims
                    for remaining_claim in claims[claim_idx:]:
                        for evidence in remaining_claim.get('evidence', []):
                            evidence['stance'] = 'NEUTRAL'
                            evidence['stance_confidence'] = 0.2
                            evidence['llm_reasoning'] = 'Skipped due to batch timeout'
                            skipped_classifications += 1
                    break
                
                claim_text = claim['text']
                evidence_items = claim.get('evidence', [])
                
                if not evidence_items:
                    continue
                
                for evidence in evidence_items:
                    try:
                        evidence_text = f"{evidence.get('title', '')} {evidence.get('snippet', '')}"
                        
                        if not evidence_text.strip():
                            evidence['stance'] = 'NEUTRAL'
                            evidence['stance_confidence'] = 0.33
                            evidence['llm_reasoning'] = 'No text'
                            continue
                        
                        # Classify (with built-in timeout)
                        result = self._classify_pair(claim_text, evidence_text)
                        
                        evidence['stance'] = result['stance']
                        evidence['stance_confidence'] = result['confidence']
                        evidence['llm_reasoning'] = result['reasoning']
                        
                        total_classifications += 1
                        
                    except Exception as e:
                        logger.error(f"Classification error: {e}")
                        evidence['stance'] = 'NEUTRAL'
                        evidence['stance_confidence'] = 0.2
                        evidence['llm_reasoning'] = f'Error: {str(e)[:50]}'
                        failed_classifications += 1
                        
            except Exception as e:
                logger.error(f"Error in claim {claim_idx}: {e}")
        
        logger.info(
            f"Stance classification complete: "
            f"✓{total_classifications} | ✗{failed_classifications} | ⊘{skipped_classifications}"
        )
        stance_dist = {'SUPPORTS': 0, 'REFUTES': 0, 'NEUTRAL': 0}
        for claim in claims:
            for evidence in claim.get('evidence', []):
                stance = evidence.get('stance', 'NEUTRAL')
                if stance in stance_dist:
                    stance_dist[stance] += 1

        logger.info(f"Stance distribution: {stance_dist}")

        if stance_dist['SUPPORTS'] == 0 and stance_dist['REFUTES'] == 0:
            logger.warning("⚠️ WARNING: All evidence classified as NEUTRAL!")
            logger.warning("This may indicate a problem with stance classification.")
        return claims
        
    
    def _classify_pair(self, claim: str, evidence: str) -> Dict:
        """
        Classify stance with smart timeout and circuit breaker (NO nested retries)
        """
        import json
        import re
        import time
        
        max_retries = 2
        max_total_time = 30
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Check circuit breaker
                elapsed = time.time() - start_time
                if elapsed > max_total_time:
                    logger.warning(
                        f"Stance classification timeout ({elapsed:.1f}s > {max_total_time}s). "
                        f"Aborting after {retry_count} retries."
                    )
                    return {
                        'stance': 'NEUTRAL',
                        'confidence': 0.2,
                        'reasoning': f'Timeout after {retry_count} retries'
                    }
                
                # Format prompt - include system prompt in the user prompt instead
                system_instructions = STANCE_CLASSIFICATION_SYSTEM or ""
                full_prompt = f"{system_instructions}\n\n{STANCE_CLASSIFICATION_PROMPT.format(claim=claim, evidence=evidence)}"
                
                # Call Groq LLM (WITHOUT system_prompt parameter)
                try:
                    response = self.llm_client.generate(
                        prompt=full_prompt,  # ← Fixed: combine system + user prompt
                        temperature=0.3,
                        max_tokens=200,
                        timeout=min(15, max_total_time - elapsed)
                    )
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        logger.warning(
                            f"Groq API error (attempt {retry_count}/{max_retries}): {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Groq API failed after {max_retries} attempts")
                        return {
                            'stance': 'NEUTRAL',
                            'confidence': 0.2,
                            'reasoning': 'API error - defaulting to NEUTRAL'
                        }
                
                # Parse response
                response_text = str(response).strip()
                
                if not response_text:
                    logger.warning("Empty response from Groq")
                    return {
                        'stance': 'NEUTRAL',
                        'confidence': 0.3,
                        'reasoning': 'Empty response'
                    }
                
                # Try JSON parsing
                try:
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = response_text[start_idx:end_idx]
                        result = json.loads(json_str)
                    else:
                        result = json.loads(response_text)
                        
                except json.JSONDecodeError:
                    # Regex fallback
                    stance_match = re.search(r'"stance"\s*:\s*"(\w+)"', response_text, re.IGNORECASE)
                    stance = stance_match.group(1).upper() if stance_match else "NEUTRAL"
                    
                    confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response_text)
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                    
                    result = {
                        "stance": stance,
                        "confidence": confidence,
                        "reasoning": "Parsed with regex fallback"
                    }
                
                # Validate
                stance = str(result.get('stance', 'NEUTRAL')).upper()
                if stance not in ['SUPPORTS', 'REFUTES', 'NEUTRAL']:
                    stance = 'NEUTRAL'
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))
                
                return {
                    'stance': stance,
                    'confidence': float(confidence),
                    'reasoning': str(result.get('reasoning', 'No reasoning'))
                }
                
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in _classify_pair: {e}")
                return {
                    'stance': 'NEUTRAL',
                    'confidence': 0.2,
                    'reasoning': 'Error during classification'
                }
        
        # All retries exhausted
        return {
            'stance': 'NEUTRAL',
            'confidence': 0.2,
            'reasoning': f'Failed after {max_retries} attempts'
        }

