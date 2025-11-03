"""Agents Module - Core processing agents for fake news detection."""

from src.agents.ingest_agent import IngestAgent
from src.agents.claim_detection_agent import ClaimDetectionAgent
from src.agents.claim_embedding_agent import ClaimEmbeddingAgent
from src.agents.evidence_retrieval_agent import EvidenceRetrievalAgent
from src.agents.aggregation_agent import AggregationAgent
from src.agents.explainability_agent import MultiMethodExplainabilityAgent
from src.agents.google_fact_check_agent import GoogleFactCheckAgent
from src.agents.snopes_agent import SnopesAgent
from src.agents.llm_stance_agent import LLMStanceAgent

__all__ = [
    "IngestAgent",
    "ClaimDetectionAgent",
    "ClaimEmbeddingAgent",
    "EvidenceRetrievalAgent",
    "AggregationAgent",
    "MultiMethodExplainabilityAgent",
    "GoogleFactCheckAgent",
    "SnopesAgent",
    "LLMStanceAgent",
]
