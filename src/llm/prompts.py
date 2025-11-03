"""LLM Prompt templates."""



"""LLM Prompt templates for fake news detection."""

STANCE_CLASSIFICATION_SYSTEM = """
You are an expert fact-checker specializing in identifying conspiracy theories and viral misinformation.

Your core task: Given a claim and a piece of evidence, classify the relationship as SUPPORTS, REFUTES, or NEUTRAL.

IMPORTANT: Be on high alert for the following CONSPIRACY THEORY RED FLAGS:
- Anonymous or unverified sources
- Claims of global plots, "shadowy" organizations, or mass coordinated cover-ups
- Allegations that evidence is "suppressed" or there is a secret agenda
- Scientific or technical impossibilities (e.g., microchips in vaccines, mind control via radio waves)
- Appeals to emotion, fear, or distrust of all experts
- "Suppressed cures" or commercial incentives (selling a product/kit)
- Absence of confirmation from reputable scientific bodies or major fact-checkers

GUIDELINES:
- SUPPORTS: Only if clear and credible evidence from trusted, independent sources is found.
- REFUTES: Use if evidence or scientific consensus clearly contradicts the claim OR if the claim matches conspiracy patterns without credible support.
- NEUTRAL: If evidence is unrelated OR claim is vague, ambiguous, or untestable.

WARNING: Do not treat absence of refuting evidence as SUPPORTS, especially for extraordinary or outlandish claims. If signs of conspiracy/hoax are present and no strong support is found, prefer REFUTES or NEUTRAL.

Return your answer in this exact JSON schema:
{
  "stance": "SUPPORTS" | "REFUTES" | "NEUTRAL",
  "confidence": 0.0 to 1.0,
  "reasoning": "Detailed rationale including any conspiracy red flags and source quality."
}
"""

STANCE_CLASSIFICATION_PROMPT = """Claim: {claim}

Evidence: {evidence}

Does the claim show signs of being a conspiracy theory or viral hoax? Look for:
- Anonymous or unverifiable sources
- Allegations of mass coordinated coverup
- Technical/scientific impossibility
- Links to online sales or "kits" to counter the claim
- Phrases like "they don't want you to know"

If any of the above are present and NO credible supporting evidence exists, lean towards REFUTES or at least NEUTRAL with low confidence, but never SUPPORTS.
Analysis instructions:
1. Is this evidence from a fact-checking source (Google Fact Check, Snopes)?
   - If YES: Trust their verdict more, they've already done verification
   - If NO: Do your own analysis

2. Does the evidence address the claim's core topic?
3. Are there matching or similar findings?
4. What is the verdict from the fact-checker?

Be LESS STRICT than with raw news. Fact-checked sources are verified.

Return JSON response."""

EVIDENCE_QUALITY_ASSESSMENT = """Assess the quality of this evidence for the given claim:

Claim: {claim}
Evidence: {evidence}
Source: {source}
Fact-Checker Rating: {rating}

Quality factors:
1. Source credibility: {source_credibility}
2. How recent: {recency}
3. How relevant: {relevance}
4. Direct match: {direct_match}

Provide confidence level: 0.0 (low quality) to 1.0 (high quality)"""


EXPLANATION_SYSTEM = """You are an expert explainer helping users understand fake news detection results.

Your task is to provide clear, concise explanations of why an article was classified as real or fake.

Focus on:
- Key evidence that influenced the verdict
- Source credibility assessment
- Logical reasoning
- Potential biases or limitations

Be objective and educational."""

EXPLANATION_PROMPT = """Article Title: {title}

Final Verdict: {verdict}
Confidence: {confidence}

Claims Analyzed: {claim_count}
- Claims supported by evidence: {supported_claims}
- Claims refuted by evidence: {refuted_claims}
- Inconclusive claims: {inconclusive_claims}

Top Evidence:
{top_evidence}

Generate a comprehensive explanation (200-300 words) that:
1. Summarizes why this verdict was reached
2. Highlights the most decisive evidence
3. Explains any uncertainties or limitations
4. Provides context for the confidence score

Explanation:"""

COUNTERFACTUAL_PROMPT = """Given this fake news detection analysis:

Claim: {claim}
Current Verdict: {verdict}
Key Evidence: {key_evidence}

Generate 2-3 counterfactual scenarios that would change the verdict. For each scenario, explain:
1. What evidence would need to be different
2. How that would change the final verdict
3. The likelihood of such evidence existing

Format as a JSON array:
[
  {{
    "scenario": "description",
    "required_change": "what evidence needs to change",
    "impact": "how verdict would change",
    "likelihood": "high/medium/low"
  }}
]"""

SUMMARIZATION_PROMPT = """Summarize this fake news detection analysis in 2-3 sentences for a general audience:

Article: {title}
Verdict: {verdict}
Confidence: {confidence}
Key Finding: {key_finding}

Summary:"""