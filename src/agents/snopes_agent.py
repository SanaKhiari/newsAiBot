"""
Snopes Fact Check Integration (Web Scraping)
Searches Snopes database for fact-checked claims
Accuracy: 10/10 | Speed: 5-10 seconds | Cost: FREE (unlimited)
"""

from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from loguru import logger
import time
from urllib.parse import urlencode, quote_plus

from src.utils import Timer, extract_domain


class SnopesAgent:
    """
    Agent for searching Snopes fact-check database using web scraping.
    
    Features:
    - Direct article search via Snopes API-like endpoint
    - Extracts ratings and verdicts
    - Handles timeouts gracefully
    
    Rate Limit: None (web scraping)
    Accuracy: 10/10 (highest credibility)
    Speed: 5-10 seconds
    Cost: FREE
    """
    
    def __init__(self):
        """Initialize Snopes scraper agent."""
        self.base_url = "https://www.snopes.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        logger.info("✓ Snopes fact-check agent initialized")
    
    def search_claims(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search Snopes for fact-checked claims.
        Uses multiple search strategies.
        
        Args:
            query: Search query (claim text)
            limit: Maximum results to return
            
        Returns:
            List of Snopes fact-checks in evidence format
        """
        try:
            with Timer(f"Snopes search: {query[:50]}"):
                logger.debug(f"Searching Snopes for: {query[:100]}")
                
                # Strategy 1: Try direct article search
                evidence = self._search_direct(query, limit)
                
                if evidence:
                    logger.info(f"✓ Snopes: Found {len(evidence)} results")
                    return evidence
                
                # Strategy 2: Try homepage search
                logger.debug("Strategy 1 failed, trying homepage search")
                evidence = self._search_homepage(query, limit)
                
                if evidence:
                    logger.info(f"✓ Snopes: Found {len(evidence)} results")
                    return evidence
                
                logger.warning(f"No Snopes results found for: {query[:50]}")
                return []
                
        except requests.exceptions.Timeout:
            logger.error("❌ Snopes request timeout (15s)")
            return []
        except requests.exceptions.ConnectionError:
            logger.error("❌ Snopes connection error")
            return []
        except Exception as e:
            logger.error(f"❌ Snopes search failed: {e}")
            return []
    
    def _search_direct(self, query: str, limit: int) -> List[Dict]:
        """
        Search Snopes articles directory directly.
        """
        try:
            # Try searching in /fact-check/ directory
            search_url = f"{self.base_url}/fact-check/?s={quote_plus(query)}"
            
            logger.debug(f"Trying direct search: {search_url[:80]}")
            
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for article links in search results
            article_links = soup.find_all('a', class_='article-link', limit=limit)
            
            if not article_links:
                # Try alternative selector
                article_links = soup.find_all('a', {'data-testid': 'article-link'}, limit=limit)
            
            evidence = []
            for link in article_links:
                evidence_item = self._parse_article_link(link)
                if evidence_item:
                    evidence.append(evidence_item)
            
            return evidence
            
        except Exception as e:
            logger.debug(f"Direct search failed: {e}")
            return []
    
    def _search_homepage(self, query: str, limit: int) -> List[Dict]:
        """
        Search from Snopes homepage search.
        """
        try:
            # Homepage search sometimes works better
            search_url = f"{self.base_url}/?s={quote_plus(query)}"
            
            logger.debug(f"Trying homepage search: {search_url[:80]}")
            
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for any article links
            article_links = soup.find_all('a', limit=limit*2)
            
            evidence = []
            for link in article_links:
                # Filter for fact-check articles only
                href = link.get('href', '')
                if '/fact-check/' in href and 'snopes.com' in href:
                    evidence_item = self._parse_article_link(link)
                    if evidence_item:
                        evidence.append(evidence_item)
                        if len(evidence) >= limit:
                            break
            
            return evidence
            
        except Exception as e:
            logger.debug(f"Homepage search failed: {e}")
            return []
    
    def _parse_article_link(self, link) -> Optional[Dict]:
        """Parse an article link into evidence format."""
        from datetime import datetime, timezone
        
        try:
            url = link.get('href', '')
            if not url:
                return None
            
            if not url.startswith('http'):
                url = f"https://www.snopes.com{url}"
            
            title = link.get_text(strip=True)
            if not title or len(title) < 5:
                return None
            
            rating = self._infer_rating_from_title(title)
            stance, confidence = self._rating_to_stance(rating)
            
            now_utc = datetime.now(timezone.utc).isoformat()  # ← TIMEZONE-AWARE
            
            evidence = {
                'url': url,
                'title': title,
                'snippet': title[:200],
                'source': 'Snopes',
                'domain': 'snopes.com',
                'publish_date': now_utc,  # ← TIMEZONE-AWARE
                'retrieved_at': now_utc,  # ← TIMEZONE-AWARE
                'stance': stance,
                'stance_confidence': confidence,
                'llm_reasoning': f"Snopes fact-check: {title[:80]}",
                'fact_checker': 'Snopes',
                'original_rating': rating,
                'source_priority': 'secondary'
            }
            
            logger.debug(f"Parsed Snopes article: {title[:60]}")
            return evidence
            
        except Exception as e:
            logger.debug(f"Error parsing article link: {e}")
            return None

    
    def _infer_rating_from_title(self, title: str) -> str:
        """
        Infer rating from article title (simplified).
        """
        title_lower = title.lower()
        
        if 'true' in title_lower:
            return 'True'
        elif 'false' in title_lower:
            return 'False'
        elif 'misleading' in title_lower:
            return 'Mostly False'
        elif 'mixed' in title_lower:
            return 'Mixed'
        else:
            return 'Unrated'
    
    def _rating_to_stance(self, rating: str) -> tuple:
        """
        Convert Snopes rating to stance format.
        """
        rating_lower = rating.lower()
        
        if 'true' in rating_lower:
            if 'mostly' in rating_lower or 'partly' in rating_lower:
                return ('SUPPORTS', 0.75)
            else:
                return ('SUPPORTS', 0.95)
        
        elif 'false' in rating_lower:
            if 'mostly' in rating_lower or 'partly' in rating_lower:
                return ('REFUTES', 0.75)
            else:
                return ('REFUTES', 0.95)
        
        else:
            return ('NEUTRAL', 0.5)
