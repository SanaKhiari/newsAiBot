"""
Claim Detection Agent: Extracts factual claims from article sentences.
Uses spaCy NER and pattern matching to identify verifiable statements.
"""

from typing import Dict, List
import spacy
from spacy.tokens import Doc
import re
from loguru import logger

from src.config import settings


class ClaimDetectionAgent:
    """Agent for detecting and extracting factual claims from text."""
    
    def __init__(self):
        """Initialize claim detection with spaCy NER."""
        try:
            self.nlp = spacy.load(settings.claim_detector_model)
            logger.info("Claim detection agent initialized")
        except OSError:
            logger.error(f"spaCy model not found: {settings.claim_detector_model}")
            raise
        
        # Patterns indicating factual claims
        self.claim_indicators = [
            r'\d+',  # Numbers
            r'\d{4}',  # Years
            r'percent|%',  # Percentages
            r'according to',  # Attribution
            r'study shows?|research|report',  # Studies
            r'said|stated|announced|declared',  # Statements
        ]
        
        # Patterns to exclude (opinions, questions)
        self.exclusion_patterns = [
            r'^\s*(I|We|You) (think|believe|feel|hope)',
            r'\?$',  # Questions
            r'^\s*(Maybe|Perhaps|Possibly)',
        ]
    
    def extract_claims(self, article: Dict) -> List[Dict]:
        """
        Extract factual claims from article sentences.
        
        Args:
            article: Article dictionary from IngestAgent
            
        Returns:
            List of claim dictionaries
        """
        logger.info(f"Extracting claims from {len(article['sentences'])} sentences")
        
        claims = []
        for sent_dict in article['sentences']:
            sentence = sent_dict['text']
            
            # Skip if matches exclusion patterns
            if any(re.search(pattern, sentence, re.IGNORECASE) 
                   for pattern in self.exclusion_patterns):
                continue
            
            # Process with spaCy
            doc = self.nlp(sentence)
            
            # Calculate claim score
            score = self._calculate_claim_score(doc, sentence)
            
            # Extract if score above threshold
            if score > 0.3:
                claim = {
                    "text": sentence,
                    "start_char": sent_dict['start_char'],
                    "end_char": sent_dict['end_char'],
                    "sentence_index": sent_dict['index'],
                    "confidence": score,
                    "entities": self._extract_entities(doc),
                    "reasoning": self._explain_extraction(doc, sentence)
                }
                claims.append(claim)
        
        logger.info(f"Extracted {len(claims)} claims from {len(article['sentences'])} sentences")
        
        return claims
    
    def _calculate_claim_score(self, doc: Doc, text: str) -> float:
        """
        Calculate claim worthiness score based on linguistic features.
        
        Args:
            doc: spaCy Doc object
            text: Original text
            
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        
        # Named entities boost score
        if len(doc.ents) > 0:
            score += 0.3
        
        # Numbers/dates boost score
        if any(re.search(pattern, text, re.IGNORECASE) 
               for pattern in self.claim_indicators[:3]):
            score += 0.2
        
        # Attribution phrases boost score
        if any(re.search(pattern, text, re.IGNORECASE) 
               for pattern in self.claim_indicators[3:]):
            score += 0.2
        
        # Proper nouns boost score
        proper_nouns = [token for token in doc if token.pos_ == 'PROPN']
        if len(proper_nouns) > 0:
            score += 0.15
        
        # Length check (too short or too long reduces score)
        words = len(text.split())
        if words < 5:
            score *= 0.5
        elif words > 50:
            score *= 0.7
        
        # Has verb (essential for claims)
        has_verb = any(token.pos_ == 'VERB' for token in doc)
        if not has_verb:
            score *= 0.5
        
        return min(score, 1.0)
    
    def _extract_entities(self, doc: Doc) -> List[Dict]:
        """Extract named entities from document."""
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities
    
    def _explain_extraction(self, doc: Doc, text: str) -> str:
        """Generate human-readable explanation for claim extraction."""
        reasons = []
        
        if len(doc.ents) > 0:
            ent_labels = [ent.label_ for ent in doc.ents]
            reasons.append(f"Contains entities: {', '.join(set(ent_labels))}")
        
        if re.search(r'\d+', text):
            reasons.append("Contains numerical data")
        
        if any(re.search(pattern, text, re.IGNORECASE) 
               for pattern in self.claim_indicators[3:]):
            reasons.append("Contains attribution or evidence markers")
        
        return "; ".join(reasons) if reasons else "General factual statement"
