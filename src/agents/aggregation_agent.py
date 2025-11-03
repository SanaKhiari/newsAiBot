"""
Aggregation Agent: Combines stance classifications into final verdict.
Applies source weighting, time decay, and confidence thresholds.
"""

from typing import List, Dict
from datetime import datetime
import math
from loguru import logger

from src.config import settings
from src.utils import time_decay_weight, get_source_weight


class AggregationAgent:
    """Agent for aggregating evidence into final verdict."""
    
    def __init__(self):
        """Initialize aggregation agent with settings."""
        self.source_weights = settings.source_weights
        self.time_decay_days = settings.time_decay_days
        self.verdict_threshold = settings.verdict_threshold
        logger.info("Aggregation agent initialized")
    
    def time_decay_weight(publish_date, time_decay_days: int = 30) -> float:
        """
        Calculate exponential time decay weight for evidence.
        
        Args:
            publish_date: Publication date (datetime or ISO string)
            time_decay_days: Half-life in days (default 30)
            
        Returns:
            Weight between 0 and 1
        """
        from datetime import datetime, timezone
        from dateutil import parser
        
        if not publish_date:
            return 1.0  # No date = full weight
        
        try:
            # Parse if string
            if isinstance(publish_date, str):
                publish_date = parser.isoparse(publish_date)
            
            # Ensure timezone-aware (use UTC)
            if publish_date.tzinfo is None:
                publish_date = publish_date.replace(tzinfo=timezone.utc)
            
            # Get current time in UTC (ALWAYS timezone-aware)
            now = datetime.now(timezone.utc)
            
            # Calculate days difference
            age_days = (now - publish_date).days
            
            # Exponential decay: weight = 0.5 ^ (age / half_life)
            if age_days < 0:
                age_days = 0  # Handle future dates
            
            weight = 0.5 ** (age_days / time_decay_days)
            return max(0.0, min(1.0, weight))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"Error in time_decay_weight: {e}, returning 1.0")
            return 1.0

    def aggregate_verdict(self, claims: List[Dict], article: Dict) -> Dict:
        """
        Aggregate all evidence into final article-level verdict.
        
        Args:
            claims: List of claims with stance classifications
            article: Original article metadata
            
        Returns:
            Verdict dictionary with scores and explanations
        """
        logger.info("Aggregating evidence into final verdict")
        
        # First, compute per-claim verdicts
        for claim in claims:
            claim_verdict = self._aggregate_claim(claim)
            claim['verdict'] = claim_verdict
        
        # Then aggregate to article level
        article_verdict = self._aggregate_article(claims, article)
        
        logger.info(f"Final verdict: {article_verdict['label']} (confidence: {article_verdict['confidence']:.3f})")
        
        return article_verdict
    
    def _aggregate_claim(self, claim: Dict) -> Dict:
        """Aggregate evidence for a single claim with improved logic."""
        from datetime import datetime, timezone
        
        evidence_items = claim.get('evidence', [])
        
        if not evidence_items:
            return {
                'label': 'INCONCLUSIVE',
                'confidence': 0.0,
                'score': 0.0,
                'supporting_count': 0,
                'refuting_count': 0,
                'neutral_count': 0,
                'reasoning': 'No evidence found'
            }
        
        supports_score = 0.0
        refutes_score = 0.0
        neutral_count = 0
        high_quality_neutral = 0
        
        for evidence in evidence_items:
            stance = evidence.get('stance', 'NEUTRAL')
            confidence = evidence.get('stance_confidence', 0.0)
            
            # Get weights
            source_weight = get_source_weight(
                evidence.get('url', ''),
                self.source_weights
            )
            
            # FIX: Handle datetime properly - ensure both are timezone-aware
            publish_date = evidence.get('publish_date')
            if publish_date:
                try:
                    if isinstance(publish_date, str):
                        # Parse ISO format string
                        from dateutil import parser
                        publish_date = parser.isoparse(publish_date)
                    
                    # Ensure timezone-aware
                    if publish_date.tzinfo is None:
                        publish_date = publish_date.replace(tzinfo=timezone.utc)
                    
                    # Now safe to compare with datetime.now(timezone.utc)
                    time_weight = time_decay_weight(publish_date, self.time_decay_days)
                except Exception as e:
                    logger.warning(f"Error parsing date {publish_date}: {e}, using default time_weight")
                    time_weight = 1.0
            else:
                time_weight = 1.0
            
            # Compute weighted score
            weighted_score = confidence * source_weight * time_weight
            
            if stance == 'SUPPORTS':
                supports_score += weighted_score
            elif stance == 'REFUTES':
                refutes_score += weighted_score
            else:
                neutral_count += 1
                domain = evidence.get('domain', '').lower()
                if any(trusted in domain for trusted in ['pubmed', 'nih', 'gov', 'edu', 'snopes']):
                    high_quality_neutral += 1
        
        # Compute final score
        score = supports_score - refutes_score
        total_weight = supports_score + refutes_score
        
        # Determine verdict
        if total_weight == 0:
            if high_quality_neutral >= 2:
                label = 'UNVERIFIABLE'
                confidence = 0.7
                reasoning = f"High-quality sources ({high_quality_neutral}) found no evidence for or against this claim"
            else:
                label = 'INCONCLUSIVE'
                confidence = 0.3
                reasoning = f"No relevant evidence found ({neutral_count} unrelated sources)"
        
        elif score > self.verdict_threshold * total_weight:
            label = 'LIKELY_REAL'
            confidence = min(score / total_weight, 1.0)
            reasoning = "Evidence supports the claim"
        
        elif score < -self.verdict_threshold * total_weight:
            label = 'LIKELY_FAKE'
            confidence = min(abs(score) / total_weight, 1.0)
            reasoning = "Evidence refutes the claim"
        
        else:
            label = 'INCONCLUSIVE'
            confidence = 0.5
            reasoning = "Mixed or insufficient evidence"
        
        return {
            'label': label,
            'confidence': float(confidence),
            'score': float(score),
            'supporting_count': int(sum(1 for e in evidence_items if e.get('stance') == 'SUPPORTS')),
            'refuting_count': int(sum(1 for e in evidence_items if e.get('stance') == 'REFUTES')),
            'neutral_count': int(neutral_count),
            'reasoning': reasoning
        }

    
    def _aggregate_article(self, claims: List[Dict], article: Dict) -> Dict:
        """Aggregate claim-level verdicts to article level."""
        if not claims:
            return {
                'label': 'INCONCLUSIVE',
                'confidence': 0.0,
                'score': 0.0,
                'claim_count': 0,
                'fake_claims': 0,
                'real_claims': 0,
                'inconclusive_claims': 0,
                'reasoning': 'No verifiable claims detected in article'
            }
        
        real_count = 0
        fake_count = 0
        inconclusive_count = 0
        total_score = 0.0
        
        for claim in claims:
            verdict = claim.get('verdict', {})
            label = verdict.get('label', 'INCONCLUSIVE')
            score = verdict.get('score', 0.0)
            
            if label == 'LIKELY_REAL':
                real_count += 1
                total_score += abs(score)
            elif label == 'LIKELY_FAKE':
                fake_count += 1
                total_score -= abs(score)
            else:
                inconclusive_count += 1
        
        # Article-level verdict
        claim_count = len(claims)
        
        if fake_count > real_count * 1.5:
            label = 'LIKELY_FAKE'
            confidence = fake_count / claim_count
            reasoning = f"{fake_count}/{claim_count} claims appear false or misleading"
        elif real_count > fake_count * 1.5:
            label = 'LIKELY_REAL'
            confidence = real_count / claim_count
            reasoning = f"{real_count}/{claim_count} claims are supported by evidence"
        else:
            label = 'MIXED_OR_INCONCLUSIVE'
            confidence = 0.5
            reasoning = f"Mixed evidence: {real_count} supported, {fake_count} refuted, {inconclusive_count} inconclusive"
        
        return {
            'label': label,
            'confidence': float(confidence),
            'score': float(total_score),
            'claim_count': claim_count,
            'real_claims': real_count,
            'fake_claims': fake_count,
            'inconclusive_claims': inconclusive_count,
            'reasoning': reasoning,
            'article_title': article.get('title', ''),
            'article_url': article.get('url', ''),
            'analyzed_at': datetime.now().isoformat()
        }
