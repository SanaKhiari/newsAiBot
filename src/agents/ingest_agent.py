"""
Ingest Agent: Parses article from URL or text.
Extracts structured content including title, text, and sentences.
"""

from typing import Dict, List, Optional
from datetime import datetime
from newspaper import Article, ArticleException
import spacy
from loguru import logger

from src.config import settings


class IngestAgent:
    """Agent responsible for ingesting and parsing news articles."""
    
    def __init__(self):
        """Initialize the ingest agent with spaCy model."""
        try:
            self.nlp = spacy.load(settings.claim_detector_model)
            logger.info(f"Loaded spaCy model: {settings.claim_detector_model}")
        except OSError:
            logger.error(f"spaCy model not found: {settings.claim_detector_model}")
            logger.info("Install with: python -m spacy download en_core_web_sm")
            raise
    
    def process_url(self, url: str) -> Dict:
        """
        Fetch and parse article from URL.
        
        Args:
            url: Article URL
            
        Returns:
            Structured article dictionary
        """
        logger.info(f"Processing URL: {url}")
        
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Extract metadata
            title = article.title or "Untitled"
            authors = article.authors or []
            publish_date = article.publish_date
            text = article.text
            top_image = article.top_image
            
            if not text:
                raise ValueError("No text content extracted from article")
            
            # Segment into sentences
            sentences = self._segment_sentences(text)
            
            result = {
                "url": url,
                "title": title,
                "authors": authors,
                "publish_date": publish_date.isoformat() if publish_date else None,
                "text": text,
                "sentences": sentences,
                "top_image": top_image,
                "word_count": len(text.split()),
                "sentence_count": len(sentences),
                "ingested_at": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed article: {title}")
            logger.debug(f"  - Sentences: {len(sentences)}, Words: {result['word_count']}")
            
            return result
            
        except ArticleException as e:
            logger.error(f"Article extraction failed: {e}")
            raise ValueError(f"Failed to extract article from URL: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing URL: {e}", exc_info=True)
            raise
    
    def process_text(self, text: str, title: Optional[str] = None) -> Dict:
        """
        Process raw article text.
        
        Args:
            text: Article text content
            title: Optional article title
            
        Returns:
            Structured article dictionary
        """
        logger.info("Processing raw text input")
        
        if not text or len(text.strip()) < 50:
            raise ValueError("Text too short for analysis (minimum 50 characters)")
        
        sentences = self._segment_sentences(text)
        
        result = {
            "url": None,
            "title": title or "User-provided text",
            "authors": [],
            "publish_date": None,
            "text": text,
            "sentences": sentences,
            "top_image": None,
            "word_count": len(text.split()),
            "sentence_count": len(sentences),
            "ingested_at": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully processed text: {len(sentences)} sentences")
        
        return result
    
    def _segment_sentences(self, text: str) -> List[Dict]:
        """
        Segment text into sentences with character positions.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence dictionaries with text and char positions
        """
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            sentences.append({
                "text": sent.text.strip(),
                "start_char": sent.start_char,
                "end_char": sent.end_char,
                "index": len(sentences)
            })
        
        logger.debug(f"Segmented text into {len(sentences)} sentences")
        return sentences
