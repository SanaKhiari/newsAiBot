"""
Orchestration Nodes: Implementation of each workflow step.
Each node executes an agent and updates the state.
"""

from datetime import datetime
from loguru import logger

from src.orchestration.state_graph import FakeNewsState
from src.agents.ingest_agent import IngestAgent
from src.agents.claim_detection_agent import ClaimDetectionAgent
from src.agents.claim_embedding_agent import ClaimEmbeddingAgent
from src.agents.evidence_retrieval_agent import EvidenceRetrievalAgent
from src.agents.aggregation_agent import AggregationAgent
from src.agents.explainability_agent import MultiMethodExplainabilityAgent
from src.utils import Timer


# ========== SINGLETON AGENT INSTANCES ==========
# Initialize agents once and reuse them
_agents = {
    'ingest': None,
    'claim_detector': None,
    'claim_embedder': None,
    'evidence_retriever': None,
    'aggregator': None,
    'explainability': None,
}


def _get_agents():
    """
    Lazy initialization of agents.
    Agents are created only once and reused.
    """
    global _agents
    
    if _agents['ingest'] is None:
        logger.info("Initializing agents (first run)...")
        _agents['ingest'] = IngestAgent()
        _agents['claim_detector'] = ClaimDetectionAgent()
        _agents['claim_embedder'] = ClaimEmbeddingAgent()
        _agents['evidence_retriever'] = EvidenceRetrievalAgent()
        _agents['aggregator'] = AggregationAgent()
        _agents['explainability'] = MultiMethodExplainabilityAgent(
            embedding_model=_agents['claim_embedder'].model
        )
        logger.info("All agents initialized")
    
    return _agents


# ========== NODE 1: INGEST NODE ==========
def ingest_node(state: FakeNewsState) -> FakeNewsState:
    """
    Node 1: Ingest article from URL or text.
    
    Input: URL or raw text
    Output: Parsed article with sentences
    """
    logger.info("=" * 60)
    logger.info("NODE 1: INGEST - Parsing article...")
    logger.info("=" * 60)
    
    state['current_step'] = 'ingest'
    
    try:
        agents = _get_agents()
        ingest_agent = agents['ingest']
        
        with Timer("Article ingestion"):
            if state.get('url'):
                article = ingest_agent.process_url(state['url'])
            elif state.get('text'):
                article = ingest_agent.process_text(state['text'], state.get('title'))
            else:
                raise ValueError("Must provide either url or text")
        
        state['article'] = article
        logger.info(f"✓ Article ingested: {article['title']}")
        logger.info(f"  Sentences: {article['sentence_count']}, Words: {article['word_count']}")
        
    except Exception as e:
        logger.error(f"✗ Ingest failed: {e}", exc_info=True)
        state['error'] = f"Ingest error: {str(e)}"
    
    return state


# ========== NODE 2: CLAIM DETECTION NODE ==========
def claim_detection_node(state: FakeNewsState) -> FakeNewsState:
    """
    Node 2: Extract factual claims from sentences.
    
    Input: Parsed article
    Output: List of extracted claims
    """
    logger.info("=" * 60)
    logger.info("NODE 2: CLAIM DETECTION - Extracting claims...")
    logger.info("=" * 60)
    
    state['current_step'] = 'claim_detection'
    
    try:
        agents = _get_agents()
        claim_detector = agents['claim_detector']
        
        with Timer("Claim detection"):
            claims = claim_detector.extract_claims(state['article'])
        
        state['claims'] = claims
        logger.info(f"✓ Claims extracted: {len(claims)} claims found")
        
        if len(claims) > 0:
            logger.info("Sample claims:")
            for i, claim in enumerate(claims[:3], 1):
                logger.info(f"  {i}. {claim['text'][:80]}... (confidence: {claim['confidence']:.2f})")
        
    except Exception as e:
        logger.error(f"✗ Claim detection failed: {e}", exc_info=True)
        state['error'] = f"Claim detection error: {str(e)}"
    
    return state


# ========== NODE 3: EMBEDDING NODE ==========
def embedding_node(state: FakeNewsState) -> FakeNewsState:
    """
    Node 3: Generate semantic embeddings for claims.
    
    Input: Claims
    Output: Claims with embedding vectors
    """
    logger.info("=" * 60)
    logger.info("NODE 3: EMBEDDING - Generating claim embeddings...")
    logger.info("=" * 60)
    
    state['current_step'] = 'embedding'
    
    try:
        agents = _get_agents()
        claim_embedder = agents['claim_embedder']
        
        with Timer("Embedding generation"):
            claims = claim_embedder.embed_claims(state['claims'])
        
        state['claims'] = claims
        state['embeddings_complete'] = True
        
        logger.info(f"✓ Embeddings generated for {len(claims)} claims")
        logger.debug(f"  Embedding dimension: {claim_embedder.embedding_dim}")
        
    except Exception as e:
        logger.error(f"✗ Embedding failed: {e}", exc_info=True)
        state['error'] = f"Embedding error: {str(e)}"
    
    return state


# ========== NODE 4: EVIDENCE RETRIEVAL NODE ==========
def evidence_retrieval_node(state: FakeNewsState) -> FakeNewsState:
    """
    Node 4: Search for evidence supporting/refuting claims.
    
    Input: Claims
    Output: Claims with evidence items
    """
    logger.info("=" * 60)
    logger.info("NODE 4: EVIDENCE RETRIEVAL - Searching for evidence...")
    logger.info("=" * 60)
    
    state['current_step'] = 'evidence_retrieval'
    
    try:
        agents = _get_agents()
        evidence_retriever = agents['evidence_retriever']
        
        with Timer("Evidence retrieval"):
            claims = evidence_retriever.retrieve_evidence(state['claims'])
        
        state['claims'] = claims
        
        # Count total evidence
        total_evidence = sum(len(c.get('evidence', [])) for c in claims)
        state['evidence_count'] = total_evidence
        
        logger.info(f"✓ Evidence retrieved: {total_evidence} items across {len(claims)} claims")
        
        # Show summary
        for claim in claims[:3]:
            evidence_count = len(claim.get('evidence', []))
            logger.info(f"  - {claim['text'][:60]}...: {evidence_count} evidence items")
        
    except Exception as e:
        logger.error(f"✗ Evidence retrieval failed: {e}", exc_info=True)
        state['error'] = f"Evidence retrieval error: {str(e)}"
    
    return state


# ========== NODE 5: STANCE CLASSIFICATION NODE ==========
def stance_classification_node(state):
    """NODE 5: STANCE CLASSIFICATION - with proper error handling"""
    logger.info("=" * 60)
    logger.info("NODE 5: STANCE CLASSIFICATION")
    logger.info("=" * 60)
    
    try:
        from src.agents.llm_stance_agent import LLMStanceAgent  # ← NOT BaseTool
        
        stance_agent = LLMStanceAgent()  # ← Direct instantiation
        claims = state.get('claims', [])
        
        logger.info(f"Classifying stance for evidence items")
        
        # Call without extra parameters
        claims = stance_agent.classify_stance_batch(claims)
        
        state['claims'] = claims
        
        logger.info("✓ Stance classification complete")
        logger.info("=" * 60)
        
        return state
        
    except AttributeError as e:
        if "llm_client" in str(e):
            logger.error(f"✗ LLMStanceAgent initialization error: {e}")
            logger.error("Make sure LLMStanceAgent is NOT inheriting from BaseTool")
        else:
            logger.error(f"✗ Attribute error: {e}")
        
        # Fallback: don't mark as NEUTRAL, mark as unable to classify
        for claim in state.get('claims', []):
            for evidence in claim.get('evidence', []):
                evidence['stance'] = 'NEUTRAL'
                evidence['stance_confidence'] = 0.1  # Very low confidence
                evidence['llm_reasoning'] = 'Classification error - fallback'
        
        return state
        
    except Exception as e:
        logger.error(f"✗ Stance classification failed: {e}", exc_info=True)
        
        # Fallback
        for claim in state.get('claims', []):
            for evidence in claim.get('evidence', []):
                evidence['stance'] = 'NEUTRAL'
                evidence['stance_confidence'] = 0.1
                evidence['llm_reasoning'] = f'Error: {e}'
        
        return state


# ========== NODE 6: AGGREGATION NODE ==========
def aggregation_node(state):
    """NODE 6: AGGREGATION - Generate verdicts using LLM"""
    logger.info("=" * 60)
    logger.info("NODE 6: AGGREGATION - Generating verdicts...")
    logger.info("=" * 60)
    
    try:
        from src.agents.verdict_agent import VerdictAgent
        
        verdict_agent = VerdictAgent()
        claims = state.get('claims', [])
        
        # Generate verdict for each claim
        for claim in claims:
            evidence = claim.get('evidence', [])
            
            try:
                claim_verdict = verdict_agent.generate_verdict(claim, claim.get('evidence', []))
                #claim_verdict = verdict_agent.generate_verdict(claim, evidence)
                claim['verdict'] = claim_verdict
                
                logger.info(f"Claim: {claim['text'][:60]}...")
                logger.info(f"  Verdict: {claim_verdict['label']}")
                logger.info(f"  Confidence: {claim_verdict['confidence']:.0%}")
            except Exception as e:
                logger.error(f"Failed to generate verdict for claim: {e}")
                claim['verdict'] = {
                    'label': 'INCONCLUSIVE',
                    'confidence': 0.0,
                    'reasoning': f'Error generating verdict: {e}',
                    'evidence_summary': ''
                }
        
        # Generate overall article verdict
        article_verdict = verdict_agent.generate_article_verdict(claims)
        state['verdict'] = article_verdict
        
        logger.info(f"✓ Article Verdict: {article_verdict['label']}")
        logger.info(f"  Confidence: {article_verdict['confidence']:.0%}")
        logger.info("=" * 60)
        
        return state
        
    except Exception as e:
        logger.error(f"✗ Aggregation failed: {e}", exc_info=True)
        state['error_handler'] = f"Aggregation error: {e}"
        return state


# ========== NODE 7: EXPLAINABILITY NODE ==========
def explainability_node(state):
    """NODE 7: EXPLAINABILITY - Generate explanations with correct signature"""
    logger.info("=" * 60)
    logger.info("NODE 7: EXPLAINABILITY - Generating explanations...")
    logger.info("=" * 60)
    
    try:
        from src.agents.explainability_agent import MultiMethodExplainabilityAgent
        
        explainability_agent = MultiMethodExplainabilityAgent()
        claims = state.get('claims', [])
        
        # FIX: Pass ONLY claims argument
        # The agent.generate_explanations(claims) expects self + claims = 2 arguments
        explanations = explainability_agent.generate_explanations(claims)
        
        state['explanations'] = explanations
        
        logger.info("✓ Explanations generated")
        logger.info("=" * 60)
        
        return state
        
    except Exception as e:
        logger.error(f"✗ Explainability failed: {e}", exc_info=True)
        
        # Fallback explanation
        state['explanations'] = {
            'llm_explanation': 'Analysis complete. Could not generate detailed explanations.',
            'llm_summary': f'Article verdict: {state.get("verdict", {}).get("label", "UNKNOWN")}',
            'evidence_ranking': [],
            'counterfactual_examples': []
        }
        
        return state


# ========== ERROR NODE ==========
def error_node(state: FakeNewsState) -> FakeNewsState:
    """
    Error Node: Handle errors gracefully.
    
    If any node encounters an error, this node logs it properly.
    """
    logger.error("=" * 60)
    logger.error(f"ERROR in step: {state.get('current_step')}")
    logger.error(f"Message: {state.get('error')}")
    logger.error("=" * 60)
    
    state['completed_at'] = datetime.now().isoformat()
    return state
