"""
Verdict Agent: Uses LLM to generate final verdicts for claims based on evidence
"""

from typing import Dict, List, Optional
from loguru import logger
from src.llm.groq_client import GroqClient
import json

class VerdictAgent:
    """Generate verdicts using LLM reasoning"""
    
    def __init__(self, groq_client: Optional[GroqClient] = None):
        """Initialize verdict agent"""
        self.groq_client = groq_client or GroqClient()
        logger.info("Verdict Agent initialized")
    
    def generate_verdict(self, claim: Dict, evidence: List[Dict]) -> Dict:
        """
        Generate verdict for a claim based on evidence using LLM
        
        Args:
            claim: Claim dictionary with text
            evidence: List of evidence items with source, stance, reasoning
            
        Returns:
            Verdict dictionary with label, confidence, and reasoning
        """
        try:
            # If no evidence, return inconclusive
            if not evidence:
                return {
                    'label': 'INCONCLUSIVE',
                    'confidence': 0.0,
                    'reasoning': 'No evidence found to verify or refute this claim.',
                    'evidence_summary': 'No relevant evidence retrieved.'
                }
            
            # Format evidence for LLM
            evidence_text = self._format_evidence(evidence)
            
            # Generate verdict with LLM
            verdict = self._llm_verdict_generation(claim, evidence)
            
            return verdict
            
        except Exception as e:
            logger.error(f"Verdict generation failed: {e}")
            return {
                'label': 'INCONCLUSIVE',
                'confidence': 0.0,
                'reasoning': 'Error during verdict generation',
                'evidence_summary': ''
            }
    
    def _format_evidence(self, evidence: List[Dict]) -> str:
        """Format evidence for LLM - show all evidence types"""
        if not evidence:
            return "No evidence found."
        
        formatted = "EVIDENCE SOURCES:\n\n"
        
        # Group by stance
        supporting = [e for e in evidence if e.get('stance') == 'SUPPORTS']
        refuting = [e for e in evidence if e.get('stance') == 'REFUTES']
        neutral = [e for e in evidence if e.get('stance') == 'NEUTRAL']
        
        # Supporting evidence
        formatted += "SUPPORTING EVIDENCE:\n"
        if supporting:
            for i, e in enumerate(supporting[:5], 1):
                formatted += f"\n{i}. {e.get('source', 'Unknown')}\n"
                formatted += f"   Title: {e.get('title', 'N/A')}\n"
                formatted += f"   Reasoning: {e.get('llm_reasoning', 'N/A')}\n"
                formatted += f"   Confidence: {e.get('stance_confidence', 0):.0%}\n"
        else:
            formatted += "None found.\n"
        
        # Refuting evidence
        formatted += "\nREFUTING EVIDENCE:\n"
        if refuting:
            for i, e in enumerate(refuting[:5], 1):
                formatted += f"\n{i}. {e.get('source', 'Unknown')}\n"
                formatted += f"   Title: {e.get('title', 'N/A')}\n"
                formatted += f"   Reasoning: {e.get('llm_reasoning', 'N/A')}\n"
                formatted += f"   Confidence: {e.get('stance_confidence', 0):.0%}\n"
        else:
            formatted += "None found.\n"
        
        # Neutral evidence
        if neutral:
            formatted += f"\nNEUTRAL/UNRELATED EVIDENCE:\n"
            formatted += f"Found {len(neutral)} neutral sources from {set(e.get('source') for e in neutral)}\n"
        
        formatted += f"\nTOTAL EVIDENCE: {len(supporting)} supporting, {len(refuting)} refuting, {len(neutral)} neutral"
        
        return formatted

    
    def _llm_verdict_generation(self, claim: Dict, evidence: List[Dict]) -> Dict:
        """Use LLM to generate verdict - with better handling of neutral evidence"""
        
        # Count evidence by stance
        supporting = sum(1 for e in evidence if e.get('stance') == 'SUPPORTS')
        refuting = sum(1 for e in evidence if e.get('stance') == 'REFUTES')
        neutral = sum(1 for e in evidence if e.get('stance') == 'NEUTRAL')
        
        # If mostly neutral, it might mean no evidence against it (which supports it)
        # rather than neutral evidence
        no_refuting = refuting == 0
        some_support = supporting > 0
        
        prompt = f"""You are a fact-checking expert. Analyze the following claim and evaluate it based on the evidence provided.

    CLAIM TO EVALUATE:
    "{claim.get('text', 'Unknown claim')}"

    EVIDENCE BREAKDOWN:
    - Supporting evidence: {supporting} sources
    - Refuting evidence: {refuting} sources  
    - Neutral/unrelated evidence: {neutral} sources

    KEY INSIGHT:
    {"Important: NO refuting evidence was found. When fact-checkers provide no evidence against a claim, it often means the claim aligns with established facts." if no_refuting and neutral > 0 else ""}

    Based on the evidence provided, determine if this claim is:
    - LIKELY_REAL: Strong evidence supports the claim, OR no evidence refutes it
    - LIKELY_FAKE: Strong evidence refutes the claim
    - MIXED: Evidence both supports and refutes the claim
    - INCONCLUSIVE: Insufficient conclusive evidence

    Respond ONLY with valid JSON (no markdown, no extra text):
    {{
        "verdict": "LIKELY_REAL or LIKELY_FAKE or MIXED or INCONCLUSIVE",
        "confidence": 0.0 to 1.0,
        "reasoning": "2-3 sentence explanation of the verdict"
    }}
    """
        
        try:
            response = self.groq_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=400,
                timeout=60
            )
            
            response_text = str(response).strip()
            
            # Clean markdown
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Parse JSON
            import json
            result = json.loads(response_text)
            
            verdict = {
                'label': str(result.get('verdict', 'INCONCLUSIVE')).upper(),
                'confidence': float(result.get('confidence', 0.5)),
                'reasoning': str(result.get('reasoning', 'No reasoning provided')),
                'evidence_summary': self._create_evidence_summary(evidence)
            }
            
            # Validate label
            valid_labels = ['LIKELY_REAL', 'LIKELY_FAKE', 'MIXED', 'INCONCLUSIVE']
            if verdict['label'] not in valid_labels:
                verdict['label'] = 'INCONCLUSIVE'
            
            verdict['confidence'] = max(0.0, min(1.0, verdict['confidence']))
            
            logger.info(f"✓ Verdict generated: {verdict['label']} ({verdict['confidence']:.0%})")
            
            return verdict
            
        except Exception as e:
            logger.error(f"LLM verdict generation error: {e}")
            return self._get_default_verdict(f"Error: {str(e)}")

    def _get_default_verdict(self, error_msg: str) -> Dict:
        """Return default verdict on error"""
        return {
            'label': 'INCONCLUSIVE',
            'confidence': 0.0,
            'reasoning': f'Could not generate verdict: {error_msg}',
            'evidence_summary': ''
        }

    
    def _create_evidence_summary(self, evidence: List[Dict]) -> str:
        """Create summary of evidence"""
        supporting = sum(1 for e in evidence if e.get('stance') == 'SUPPORTS')
        refuting = sum(1 for e in evidence if e.get('stance') == 'REFUTES')
        neutral = sum(1 for e in evidence if e.get('stance') == 'NEUTRAL')
        
        return f"{supporting} supporting, {refuting} refuting, {neutral} neutral sources"
    
    def generate_article_verdict(self, claims: List[Dict]) -> Dict:
        """
        Generate overall article verdict from individual claim verdicts
        
        Args:
            claims: List of claims with verdicts
            
        Returns:
            Overall verdict for the article
        """
        try:
            if not claims:
                return {
                    'label': 'INCONCLUSIVE',
                    'confidence': 0.0,
                    'reasoning': 'No claims analyzed',
                    'score': 0.0
                }
            
            # Filter claims with verdicts
            verdicts = [c.get('verdict', {}) for c in claims if c.get('verdict')]
            
            if not verdicts:
                return {
                    'label': 'INCONCLUSIVE',
                    'confidence': 0.0,
                    'reasoning': 'No verdicts generated',
                    'score': 0.0
                }
            
            # Count verdict types
            real_count = sum(1 for v in verdicts if v.get('label') == 'LIKELY_REAL')
            fake_count = sum(1 for v in verdicts if v.get('label') == 'LIKELY_FAKE')
            mixed_count = sum(1 for v in verdicts if v.get('label') == 'MIXED')
            inconclusive_count = sum(1 for v in verdicts if v.get('label') == 'INCONCLUSIVE')
            
            # Calculate average confidence
            avg_confidence = sum(v.get('confidence', 0) for v in verdicts) / len(verdicts)
            
            # Determine overall verdict
            if fake_count > real_count + mixed_count:
                overall_label = 'LIKELY_FAKE'
                reasoning = f"Majority of claims ({fake_count}/{len(verdicts)}) refuted by evidence"
            elif real_count > fake_count + mixed_count:
                overall_label = 'LIKELY_REAL'
                reasoning = f"Majority of claims ({real_count}/{len(verdicts)}) supported by evidence"
            elif mixed_count >= real_count or mixed_count >= fake_count:
                overall_label = 'MIXED_OR_INCONCLUSIVE'
                reasoning = f"Mixed evidence: {real_count} supported, {fake_count} refuted, {mixed_count} mixed"
            else:
                overall_label = 'INCONCLUSIVE'
                reasoning = f"Insufficient evidence: {inconclusive_count} inconclusive claims"
            
            return {
                'label': overall_label,
                'confidence': avg_confidence,
                'reasoning': reasoning,
                'score': real_count - fake_count,
                'claim_breakdown': {
                    'real': real_count,
                    'fake': fake_count,
                    'mixed': mixed_count,
                    'inconclusive': inconclusive_count
                }
            }
            
        except Exception as e:
            logger.error(f"Article verdict generation failed: {e}")
            return {
                'label': 'INCONCLUSIVE',
                'confidence': 0.0,
                'reasoning': f'Error generating article verdict: {e}',
                'score': 0.0
            }