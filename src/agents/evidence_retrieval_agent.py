"""
Evidence Retrieval Agent: Searches for evidence supporting/refuting claims.
Uses GDELT, RSS feeds, and CommonCrawl - all free sources.
"""

from typing import List, Dict, Optional
import requests
from datetime import datetime, timedelta
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urlencode, quote_plus
import feedparser
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import json
from sentence_transformers import SentenceTransformer, util  # for semantic similarity
from langdetect import detect 
from src.config import settings
from src.utils import extract_domain, compute_hash, Timer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')    # or your selected model


class EvidenceRetrievalAgent:
    

    """Agent for retrieving evidence from web sources."""
    
    def __init__(self):
        """Initialize retrieval agent with RSS feeds and API endpoints."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FakeNewsDetectionBot/1.0 (Educational; +https://github.com/yourproject)'
        })
        
        # Load RSS feeds
        self.rss_feeds = self._load_rss_feeds()
        logger.info(f"Loaded {len(self.rss_feeds)} RSS feeds")
        
        # Cache for avoiding duplicate requests
        self.cache = {}

    def filter_evidence_by_relevance(self, claim_text: str, evidence_list: List[Dict], threshold=0.25) -> List[Dict]:
        "Filters evidence for relevance and language. Returns sorted, relevant items."
        # Step 1: Embed claim
        claim_embedding = embedding_model.encode(claim_text)
        relevant_evidence = []

        for evidence in evidence_list:
            # Filter non-English
            try:
                if detect(evidence.get('title', '') + ' ' + evidence.get('snippet', '')) != 'en':
                    continue
            except Exception:
                continue

            # Semantic similarity
            text_for_embedding = evidence.get('title', '') + ' ' + evidence.get('snippet', '')
            evidence_embedding = embedding_model.encode(text_for_embedding)
            similarity = util.cos_sim(claim_embedding, evidence_embedding).item()
            evidence['relevance_score'] = similarity

            if similarity >= threshold:
                relevant_evidence.append(evidence)

        # Sort by relevance
        relevant_evidence.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_evidence

    def _load_rss_feeds(self) -> List[Dict]:
        """Load RSS feed list."""
        default_feeds = [
            {"name": "BBC World", "url": "http://feeds.bbci.co.uk/news/world/rss.xml", "weight": 0.9},
            {"name": "Reuters World", "url": "https://www.reutersagency.com/feed/", "weight": 1.0},
            {"name": "AP News Top", "url": "https://apnews.com/hub/world-news", "weight": 1.0},
            {"name": "CNN World", "url": "http://rss.cnn.com/rss/edition_world.rss", "weight": 0.8},
            {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml", "weight": 0.9},
            {"name": "The Guardian World", "url": "https://www.theguardian.com/world/rss", "weight": 0.9},
            {"name": "New York Times World", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", "weight": 0.9},
            {"name": "Fox News Latest", "url": "http://feeds.foxnews.com/foxnews/latest", "weight": 0.8},
            {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "weight": 0.8},
            {"name": "Deutsche Welle", "url": "https://rss.dw.com/rdf/rss-en-all", "weight": 0.8}
        ]
        return default_feeds
    
    def retrieve_evidence(self, claims: List[Dict], max_per_claim: int = None) -> List[Dict]:
        """
        Retrieve evidence for all claims.
        
        Args:
            claims: List of claim dictionaries
            max_per_claim: Maximum evidence items per claim
            
        Returns:
            Claims with added 'evidence' field
        """
        max_per_claim = max_per_claim or settings.max_evidence_per_claim
        logger.info(f"Retrieving evidence for {len(claims)} claims")
        
        for claim in claims:
            with Timer(f"Evidence retrieval for claim: {claim['text'][:50]}..."):
                try:
                    evidence = self._retrieve_for_claim(claim, max_per_claim)
                    claim['evidence'] = evidence
                except Exception as e:
                    logger.error(f"Evidence retrieval failed for claim: {e}")
                    claim['evidence'] = []
        
        return claims
    def _search_pubmed(self, query: str, limit: int = 3) -> List[Dict]:
        """Search PubMed for medical research articles."""
        try:
            from datetime import datetime, timezone
            
            logger.debug(f"Searching PubMed: {query[:100]}")
            
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': limit,
                'retmode': 'json'
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"PubMed search returned status {response.status_code}")
                return []
            
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                logger.debug(f"No PubMed results found for: {query[:50]}")
                return []
            
            logger.info(f"✓ PubMed: Found {len(pmids)} research articles")
            
            # Build evidence items from PMIDs
            evidence = []
            now = datetime.now(timezone.utc).isoformat()
            
            for pmid in pmids[:limit]:
                evidence.append({
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    'title': f"PubMed Research Article {pmid}",
                    'snippet': f"Medical research on {query}",
                    'source': 'PubMed',
                    'domain': 'pubmed.ncbi.nlm.nih.gov',
                    'publish_date': now,
                    'retrieved_at': now,
                    'stance': 'NEUTRAL',
                    'stance_confidence': 0.5,
                    'llm_reasoning': f"PubMed research article related to: {query[:60]}",
                    'source_priority': 'tertiary'
                })
            
            return evidence
            
        except Exception as e:
            logger.error(f"❌ PubMed search failed: {e}")
            return []

    def _retrieve_for_claim(self, claim: Dict, max_items: int) -> List[Dict]:
        """
        Retrieve evidence for a single claim with prioritized sources.
        
        Priority order:
        1. Google Fact Check (highest accuracy)
        2. Snopes (highest credibility)
        3. PubMed (scientific claims)
        4. GDELT (fallback)
        
        Args:
            claim: Claim dictionary
            max_items: Maximum evidence items to retrieve
            
        Returns:
            List of evidence items
        """
        evidence_items = []
        
        # Generate search queries from claim
        queries = self._generate_queries(claim)
        logger.debug(f"Generated {len(queries)} search queries for claim")
        
        for query_index, query in enumerate(queries, 1):
            try:
                logger.debug(f"\n--- Query {query_index}/{len(queries)}: {query[:80]} ---")
                
                # PRIORITY 1: GOOGLE FACT CHECK
                if settings.google_fact_check_enabled:
                    try:
                        from src.agents.google_fact_check_agent import GoogleFactCheckAgent
                        gfc_agent = GoogleFactCheckAgent()
                        gfc_results = gfc_agent.search_claims(query, limit=3)
                        if gfc_results:
                            evidence_items.extend(gfc_results)
                            logger.info(f"✓ Google FC: Found {len(gfc_results)} results")
                            
                            # If we have good Google FC results, use them
                            if len(evidence_items) >= max_items * 0.6:
                                logger.info("Good results from Google FC, moving forward")
                                break
                    except Exception as e:
                        logger.error(f"Google FC error: {e}")
                
                # PRIORITY 2: SNOPES
                if settings.snopes_enabled and len(evidence_items) < max_items * 0.5:
                    try:
                        from src.agents.snopes_agent import SnopesAgent
                        snopes_agent = SnopesAgent()
                        snopes_results = snopes_agent.search_claims(query, limit=2)
                        if snopes_results:
                            evidence_items.extend(snopes_results)
                            logger.info(f"✓ Snopes: Found {len(snopes_results)} results")
                    except Exception as e:
                        logger.error(f"Snopes error: {e}")
                
                # PRIORITY 3: PUBMED (for medical/health claims)
                if any(word in query.lower() for word in ['cancer', 'disease', 'health', 'medical', 'study', 'risk']):
                    if len(evidence_items) < max_items * 0.4:
                        try:
                            pubmed_results = self._search_pubmed(query, limit=2)
                            if pubmed_results:
                                evidence_items.extend(pubmed_results)
                                logger.info(f"✓ PubMed: Found {len(pubmed_results)} results")
                        except Exception as e:
                            logger.error(f"PubMed error: {e}")
                
                # PRIORITY 4: GDELT (fallback)
                if settings.gdelt_enabled and len(evidence_items) < max_items * 0.3:
                    try:
                        gdelt_results = self._search_gdelt(query, limit=2)
                        if gdelt_results:
                            evidence_items.extend(gdelt_results)
                            logger.info(f"✓ GDELT: Found {len(gdelt_results)} results (fallback)")
                    except Exception as e:
                        logger.error(f"GDELT error: {e}")
                
                # Check if we have enough evidence
                if len(evidence_items) >= max_items:
                    logger.info(f"Sufficient evidence found ({len(evidence_items)}/{max_items}), stopping search")
                    break
                
                # Rate limiting delay
                time.sleep(settings.request_delay)
                
            except Exception as e:
                logger.error(f"Error with query '{query}': {e}")
                continue
        
        # DEDUPLICATE BY URL
        logger.debug(f"\nDeduplicating {len(evidence_items)} evidence items")
        seen_urls = set()
        unique_evidence = []
        for item in evidence_items:
            url = item.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_evidence.append(item)
        
        # Log breakdown
        google_fc_count = sum(1 for e in unique_evidence if 'Google FC' in e.get('source', ''))
        snopes_count = sum(1 for e in unique_evidence if e.get('source') == 'Snopes')
        pubmed_count = sum(1 for e in unique_evidence if e.get('source') == 'PubMed')
        gdelt_count = sum(1 for e in unique_evidence if e.get('source') == 'GDELT')
        
        logger.info(f"\nEvidence breakdown:")
        logger.info(f"  - Google Fact Check: {google_fc_count}")
        logger.info(f"  - Snopes: {snopes_count}")
        logger.info(f"  - PubMed: {pubmed_count}")
        logger.info(f"  - GDELT: {gdelt_count}")
        logger.info(f"  - Total unique: {len(unique_evidence)}")
        
        claim_text = claim.get('text', '')
        relevant_evidence = self.filter_evidence_by_relevance(claim_text, unique_evidence, threshold=0.25)
        if not relevant_evidence:  # Fallback: If nothing left, just return the top evidence by default
            logger.warning("No relevant evidence after filtering, returning unfiltered top items")
            return unique_evidence[:max_items]
        logger.info(f"✓ Filtered: {len(relevant_evidence)} relevant evidence items")
        return relevant_evidence[:max_items]
    
    def _generate_queries(self, claim: Dict) -> List[str]:
        """
        Generate focused, relevant search queries from claim.
        Creates multiple query variations for better results.
        
        Args:
            claim: Claim dictionary
            
        Returns:
            List of search queries, ordered by relevance
        """
        queries = []
        
        # Original claim text (without period)
        text = claim['text'].strip().rstrip('.')
        queries.append(text)
        
        # Extract entities and create entity-based queries
        if claim.get('entities'):
            # Method 1: Get all entity texts
            main_entities = [
                ent['text'] for ent in claim['entities']
                if ent['label'] in ['PRODUCT', 'ORG', 'PERSON', 'GPE', 'NORP', 'PERCENT']
            ]
            if main_entities:
                entity_query = " ".join(main_entities[:3])
                if entity_query not in queries:
                    queries.append(entity_query)
        
        # Method 2: Extract key terms (remove filler words)
        import re
        text_lower = text.lower()
        
        # Remove common filler phrases
        text_clean = re.sub(r'(scientists|researchers) (have )?(discovered|found|shows?|suggests?)', '', text_lower)
        text_clean = re.sub(r'(a|the) (study|research|report) (shows?|suggests?|found) that', '', text_clean)
        text_clean = re.sub(r'according to', '', text_clean)
        
        # Extract important words (nouns, verbs)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 'be', 
                    'have', 'has', 'had', 'do', 'does', 'did', 'that', 'this', 'by', 'of'}
        
        words = [w for w in text_clean.split() if w not in stopwords and len(w) > 2]
        
        if len(words) >= 2:
            # Create query from key words
            key_query = " ".join(words[:5])  # Top 5 key words
            if key_query not in queries and len(key_query) > 5:
                queries.append(key_query)
        
        # Method 3: Simplified version (first few words)
        first_words = " ".join(text.split()[:4])
        if first_words not in queries and len(first_words) > 5:
            queries.append(first_words)
        
        logger.debug(f"Generated {len(queries)} queries from claim:")
        for i, q in enumerate(queries, 1):
            logger.debug(f"  {i}. {q[:80]}")
        
        return queries[:4]  # Return top 4 queries

    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _search_gdelt(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search GDELT for articles.
        GDELT DOC API is free and searches last 3 months of news.
        """
        try:
            # Prepare request parameters
            params = {
                'query': query,
                'mode': 'artlist',
                'maxrecords': limit,
                'format': 'json',
                'sort': 'datedesc'
            }
            
            logger.debug(f"Searching GDELT with query: {query[:100]}")
            
            # Make request with timeout
            response = self.session.get(
                settings.gdelt_api_url,
                params=params,
                timeout=settings.evidence_timeout
            )
            
            # ========== HANDLE RATE LIMITING (429) ==========
            if response.status_code == 429:
                logger.warning(f"GDELT rate limited (429). Retrying with exponential backoff...")
                # The @retry decorator will handle the retry with exponential wait
                raise requests.exceptions.ConnectionError("GDELT rate limited: 429")
            
            # ========== HANDLE OTHER HTTP ERRORS ==========
            if response.status_code == 404:
                logger.warning(f"GDELT returned 404: Query not found")
                return []
            
            if response.status_code == 401:
                logger.error(f"GDELT authentication error (401)")
                return []
            
            if response.status_code == 500:
                logger.warning(f"GDELT server error (500), retrying...")
                raise requests.exceptions.ConnectionError("GDELT server error: 500")
            
            if response.status_code != 200:
                logger.warning(f"GDELT API returned unexpected status: {response.status_code}")
                return []
            
            # ========== PARSE RESPONSE ==========
            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError as e:
                logger.error(f"GDELT returned invalid JSON: {e}")
                logger.debug(f"Response content: {response.text[:200]}")
                return []
            
            # ========== EXTRACT ARTICLES ==========
            articles = data.get('articles', [])
            
            if not articles:
                logger.debug(f"GDELT returned no articles for query: {query[:50]}")
                return []
            
            logger.debug(f"GDELT found {len(articles)} articles, extracting top {limit}")
            
            # ========== BUILD EVIDENCE LIST ==========
            evidence = []
            for idx, article in enumerate(articles[:limit]):
                try:
                    article_url = article.get('url', '')
                    article_title = article.get('title', '')
                    
                    # Skip articles with missing critical data
                    if not article_url or not article_title:
                        logger.debug(f"Skipping article {idx}: missing URL or title")
                        continue
                    
                    evidence_item = {
                        'url': article_url,
                        'title': article_title,
                        'snippet': article.get('seendate', '')[:200],
                        'source': 'GDELT',
                        'domain': extract_domain(article_url),
                        'publish_date': article.get('seendate'),
                        'retrieved_at': datetime.now().isoformat()
                    }
                    
                    evidence.append(evidence_item)
                    
                except Exception as e:
                    logger.warning(f"Error parsing article {idx}: {e}")
                    continue
            
            # ========== LOG RESULTS ==========
            logger.debug(f"GDELT returned {len(evidence)} valid results for query: {query[:50]}")
            
            if len(evidence) > 0:
                logger.info(f"✓ GDELT: Found {len(evidence)} articles")
            else:
                logger.warning(f"✗ GDELT: No valid articles extracted")
            
            return evidence
            
        # ========== ERROR HANDLING ==========
        except requests.exceptions.Timeout:
            logger.error(f"GDELT request timeout ({settings.evidence_timeout}s)")
            return []
        
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"GDELT connection error (will retry): {e}")
            # Re-raise to trigger @retry decorator
            raise
        
        except requests.exceptions.RequestException as e:
            logger.error(f"GDELT request failed: {e}")
            return []
        
        except Exception as e:
            logger.error(f"GDELT search failed with unexpected error: {e}", exc_info=True)
            return []
    
    def _search_rss_feeds(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search RSS feeds with STRICT keyword matching.
        Only returns articles highly relevant to the query.
        """
        evidence = []
        query_lower = query.lower()
        
        # Extract meaningful keywords (remove stopwords)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been', 'be',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'must', 'can', 'that', 'this', 'these', 'those'}
        
        keywords = [w for w in query_lower.split() if w not in stopwords and len(w) > 2]
        
        # If query is empty after stopword removal, use original
        if not keywords:
            keywords = query_lower.split()
        
        # Require more keywords to match for longer queries
        required_matches = max(2, int(len(keywords) * 0.6))  # At least 60% of keywords
        
        logger.debug(f"RSS search keywords: {keywords}, required matches: {required_matches}")
        
        for feed_info in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_info['url'])
                
                for entry in feed.entries[:20]:  # Check recent entries
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()
                    
                    # Combined text for matching
                    text = f"{title} {summary}"
                    
                    # Count keyword matches
                    matches = sum(1 for kw in keywords if kw in text)
                    
                    # Calculate relevance score
                    relevance_score = matches / len(keywords) if keywords else 0
                    
                    # STRICTER: Require 60% keyword match AND title must contain at least 1 keyword
                    title_has_keyword = any(kw in title for kw in keywords)
                    
                    if matches >= required_matches and title_has_keyword and relevance_score >= 0.6:
                        evidence.append({
                            'url': entry.get('link', ''),
                            'title': entry.get('title', ''),
                            'snippet': entry.get('summary', '')[:200],
                            'source': feed_info['name'],
                            'domain': extract_domain(entry.get('link', '')),
                            'publish_date': entry.get('published', None),
                            'retrieved_at': datetime.now().isoformat(),
                            'relevance_score': relevance_score  # For debugging
                        })
                        
                        logger.debug(f"RSS match: {entry.get('title', '')[:60]} (score: {relevance_score:.2f})")
                    
                    if len(evidence) >= limit:
                        break
                
                if len(evidence) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"RSS feed {feed_info['name']} failed: {e}")
                continue
        
        logger.debug(f"RSS feeds returned {len(evidence)} results for query: {query[:50]}")
        return evidence
