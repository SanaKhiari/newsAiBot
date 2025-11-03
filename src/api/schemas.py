"""
API Request/Response schemas using Pydantic.
These validate and document all API data.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime


# ========== REQUEST SCHEMAS ==========

class AnalyzeRequest(BaseModel):
    """
    Request model for article analysis endpoint.
    
    User sends one of these to the API.
    Pydantic validates the data automatically.
    """
    
    url: Optional[HttpUrl] = Field(
        None,
        description="Article URL to analyze",
        example="https://www.bbc.com/news/world"
    )
    text: Optional[str] = Field(
        None,
        description="Raw article text to analyze",
        example="Apple released the iPhone 15 with improved battery life."
    )
    title: Optional[str] = Field(
        None,
        description="Article title (optional)",
        example="iPhone 15 Released"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "text": "The stock market rose 2% today.",
                "title": "Market Report"
            }
        }


# ========== RESPONSE SCHEMAS ==========

class ClaimData(BaseModel):
    """Single claim extracted from article."""
    
    text: str = Field(description="Claim text")
    confidence: float = Field(description="Extraction confidence (0-1)")
    sentence_index: int = Field(description="Index in article")
    entities: List[Dict[str, Any]] = Field(description="Named entities found")


class EvidenceData(BaseModel):
    """Single piece of evidence for a claim."""
    
    url: str = Field(description="Evidence source URL")
    title: str = Field(description="Evidence title")
    source: str = Field(description="Source name (BBC, Reuters, etc)")
    stance: str = Field(description="SUPPORTS, REFUTES, or NEUTRAL")
    stance_confidence: float = Field(description="Confidence in stance (0-1)")
    llm_reasoning: str = Field(description="Why LLM assigned this stance")


class VerdictData(BaseModel):
    """Final verdict on article."""
    
    label: str = Field(
        description="LIKELY_REAL, LIKELY_FAKE, or MIXED_OR_INCONCLUSIVE"
    )
    confidence: float = Field(description="Confidence in verdict (0-1)")
    claim_count: int = Field(description="Total claims analyzed")
    real_claims: int = Field(description="Claims supported by evidence")
    fake_claims: int = Field(description="Claims contradicted by evidence")
    inconclusive_claims: int = Field(description="Inconclusive claims")
    reasoning: str = Field(description="Human-readable explanation")
    article_title: str = Field(description="Article being analyzed")
    analyzed_at: str = Field(description="When analysis completed")


class ConfidenceBreakdown(BaseModel):
    """Breakdown of confidence factors."""
    
    overall_confidence: float = Field(description="Overall confidence score")
    claim_count: int = Field(description="Number of claims analyzed")
    evidence_quality: str = Field(description="High/Medium/Low")
    source_reliability: str = Field(description="High/Medium/Low/Unknown")
    llm_consistency: str = Field(description="Consistency assessment")


class EvidenceRankingItem(BaseModel):
    """Most impactful evidence item."""
    
    claim_text: str = Field(description="Related claim")
    evidence_title: str = Field(description="Evidence title")
    evidence_url: str = Field(description="Evidence URL")
    stance: str = Field(description="SUPPORTS or REFUTES")
    confidence: float = Field(description="Stance confidence")
    llm_reasoning: str = Field(description="Reasoning from LLM")
    impact_score: float = Field(description="Impact on verdict")


class ExplanationsData(BaseModel):
    """Explanations using multiple methods."""
    
    llm_explanation: Optional[str] = Field(
        None,
        description="Natural language explanation from LLM"
    )
    llm_summary: Optional[str] = Field(
        None,
        description="Brief summary from LLM"
    )
    evidence_ranking: List[EvidenceRankingItem] = Field(
        description="Top evidence items ranked by impact"
    )
    confidence_breakdown: ConfidenceBreakdown = Field(
        description="Breakdown of confidence factors"
    )
    counterfactual_examples: List[Dict[str, Any]] = Field(
        description="What-if scenarios that would change verdict"
    )
    html_report: Optional[str] = Field(
        None,
        description="HTML report with visualizations"
    )


class AnalyzeResponse(BaseModel):
    """
    Complete response from analysis endpoint.
    
    Contains all results: verdict, claims, evidence, explanations.
    """
    
    status: str = Field(description="'success' or 'error'")
    
    # Analysis results
    verdict: VerdictData = Field(description="Final verdict")
    claims: List[Dict[str, Any]] = Field(description="Extracted claims")
    explanations: ExplanationsData = Field(description="Multi-method explanations")
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        description="Timing and execution info"
    )


class ErrorResponse(BaseModel):
    """Error response."""
    
    status: str = Field(default="error", description="Status")
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(None, description="Error details")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service status")
    ollama_status: str = Field(description="LLM availability")
    timestamp: str = Field(description="Check timestamp")
