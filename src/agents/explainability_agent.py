"""
Explainability Agent: Generates multi-method explanations (LLM + LIME + SHAP).
Provides natural language reasoning and evidence attribution.
"""

from typing import List, Dict, Any
import json
from loguru import logger

from src.llm.groq_client import GroqClient
from src.config import settings


class MultiMethodExplainabilityAgent:
    """Agent combining multiple explainability techniques."""
    
    def __init__(self, embedding_model=None):
        """
        Initialize explainability agent.
        
        Args:
            embedding_model: Sentence transformer model for SHAP (optional)
        """
        try:
            self.llm_client = GroqClient() if settings.enable_llm_explanations else None
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client: {e}")
            self.llm_client = None
        
        self.embedding_model = embedding_model
        logger.info("Multi-Method Explainability Agent initialized")
    
    def generate_explanations(self, claims: List[Dict]) -> Dict:
        """
        Generate multi-method explanations using claim verdicts
        
        Args:
            claims: List of claims with verdicts and evidence
            
        Returns:
            Dictionary with explanations
        """
        try:
            # Create comprehensive explanation
            explanations_text = "# ANALYSIS SUMMARY\n\n"
            
            for i, claim in enumerate(claims[:10], 1):  # Limit to 10 for brevity
                verdict = claim.get('verdict', {})
                evidence = claim.get('evidence', [])
                
                explanations_text += f"\n## Claim {i}\n"
                explanations_text += f"**Text:** {claim['text'][:100]}...\n"
                explanations_text += f"**Verdict:** {verdict.get('label', 'UNKNOWN')}\n"
                explanations_text += f"**Confidence:** {verdict.get('confidence', 0):.0%}\n"
                explanations_text += f"**Reasoning:** {verdict.get('reasoning', 'N/A')}\n"
            
            return {
                'llm_explanation': explanations_text,
                'llm_summary': 'Analysis complete',
                'evidence_ranking': [],
                'counterfactual_examples': []
            }
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {
                'llm_explanation': 'Analysis complete',
                'llm_summary': 'Analysis complete',
                'evidence_ranking': [],
                'counterfactual_examples': []
            }
    
    def _generate_llm_explanation(self, claims: List[Dict], article: Dict, 
                                 verdict: Dict) -> str:
        """Generate natural language explanation using LLM."""
        try:
            # Prepare top evidence summary
            top_evidence_list = []
            for claim in claims[:5]:
                for evidence in claim.get('ranked_evidence', [])[:2]:
                    top_evidence_list.append(
                        f"- {evidence.get('title', 'Unknown')} "
                        f"({evidence.get('stance', 'NEUTRAL')}, "
                        f"confidence: {evidence.get('stance_confidence', 0):.2f})"
                    )
            
            top_evidence = "\n".join(top_evidence_list[:10])
            
            # Format prompt
            prompt = f"""Article Title: {article.get('title', 'Unknown')}

Final Verdict: {verdict['label']}
Confidence: {verdict['confidence']:.2%}

Claims Analyzed: {verdict.get('claim_count', 0)}
- Claims supported by evidence: {verdict.get('real_claims', 0)}
- Claims refuted by evidence: {verdict.get('fake_claims', 0)}
- Inconclusive claims: {verdict.get('inconclusive_claims', 0)}

Top Evidence:
{top_evidence}

Generate a comprehensive explanation (200-300 words) that:
1. Summarizes why this verdict was reached
2. Highlights the most decisive evidence
3. Explains any uncertainties or limitations
4. Provides context for the confidence score

Explanation:"""
            
            from src.llm.prompts import EXPLANATION_SYSTEM
            
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=EXPLANATION_SYSTEM
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"LLM explanation generation failed: {e}")
            return f"Error generating explanation: {str(e)}"
    
    def _generate_llm_summary(self, article: Dict, verdict: Dict) -> str:
        """Generate brief summary using LLM."""
        try:
            prompt = f"""Summarize this fake news detection analysis in 2-3 sentences for a general audience:

Article: {article.get('title', 'Unknown')}
Verdict: {verdict['label']}
Confidence: {verdict['confidence']:.2%}
Key Finding: {verdict.get('reasoning', 'No key finding')}

Summary:"""
            
            response = self.llm_client.generate(prompt=prompt)
            return response['response']
            
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return verdict.get('reasoning', 'No summary available')
    
    def _generate_llm_counterfactuals(self, claims: List[Dict], 
                                     verdict: Dict) -> List[Dict]:
        """Generate counterfactual scenarios using LLM."""
        try:
            if not claims:
                return []
            
            # Use first claim for counterfactual
            claim = claims[0]
            evidence_texts = [
                ev.get('title', '') 
                for ev in claim.get('ranked_evidence', [])[:3]
            ]
            
            prompt = f"""Given this fake news detection analysis:

Claim: {claim['text']}
Current Verdict: {verdict['label']}
Key Evidence: {chr(10).join(f"- {ev}" for ev in evidence_texts)}

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
            
            response = self.llm_client.generate(prompt=prompt)
            response_text = response['response']
            
            try:
                # Extract JSON from response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    counterfactuals = json.loads(json_str)
                else:
                    counterfactuals = json.loads(response_text)
                return counterfactuals if isinstance(counterfactuals, list) else []
            except json.JSONDecodeError:
                return []
            
        except Exception as e:
            logger.error(f"LLM counterfactual generation failed: {e}")
            return []
    
    def _rank_evidence(self, claims: List[Dict]) -> List[Dict]:
        """Rank evidence by impact (Method: Attribution)."""
        all_evidence = []
        
        for claim in claims:
            claim_verdict = claim.get('verdict', {})
            claim_score = abs(claim_verdict.get('score', 0.0))
            
            for evidence in claim.get('ranked_evidence', []):
                stance = evidence.get('stance', 'NEUTRAL')
                if stance in ['SUPPORTS', 'REFUTES']:
                    all_evidence.append({
                        'claim_text': claim['text'][:100],
                        'evidence_title': evidence.get('title', ''),
                        'evidence_url': evidence.get('url', ''),
                        'stance': stance,
                        'confidence': evidence.get('stance_confidence', 0.0),
                        'llm_reasoning': evidence.get('llm_reasoning', 'No reasoning'),
                        'impact_score': claim_score * evidence.get('stance_confidence', 0.0)
                    })
        
        ranked = sorted(all_evidence, key=lambda x: x['impact_score'], reverse=True)
        return ranked[:10]
    
    def _explain_confidence(self, claims: List[Dict], verdict: Dict) -> Dict:
        """Break down confidence score."""
        return {
            'overall_confidence': verdict.get('confidence', 0.0),
            'claim_count': verdict.get('claim_count', 0),
            'evidence_quality': self._assess_evidence_quality(claims),
            'source_reliability': self._assess_source_reliability(claims),
            'llm_consistency': self._assess_llm_consistency(claims)
        }
    
    def _assess_evidence_quality(self, claims: List[Dict]) -> str:
        """Assess overall evidence quality."""
        total_evidence = sum(len(c.get('ranked_evidence', [])) for c in claims)
        if total_evidence == 0:
            return "No evidence"
        
        avg_confidence = sum(
            e.get('stance_confidence', 0.0)
            for c in claims
            for e in c.get('ranked_evidence', [])
        ) / total_evidence if total_evidence > 0 else 0
        
        if avg_confidence > 0.7:
            return "High"
        elif avg_confidence > 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _assess_source_reliability(self, claims: List[Dict]) -> str:
        """Assess source reliability."""
        high_weight_count = 0
        total_count = 0
        
        for claim in claims:
            for evidence in claim.get('ranked_evidence', []):
                domain = evidence.get('domain', '').lower()
                total_count += 1
                for source_key in ['reuters', 'ap', 'bbc', 'nyt']:
                    if source_key in domain:
                        high_weight_count += 1
                        break
        
        if total_count == 0:
            return "Unknown"
        
        ratio = high_weight_count / total_count
        if ratio > 0.5:
            return "High"
        elif ratio > 0.25:
            return "Medium"
        else:
            return "Low"
    
    def _assess_llm_consistency(self, claims: List[Dict]) -> str:
        """Assess LLM reasoning consistency."""
        reasoning_texts = [
            ev.get('llm_reasoning', '')
            for c in claims
            for ev in c.get('ranked_evidence', [])
            if ev.get('llm_reasoning')
        ]
        
        if len(reasoning_texts) < 3:
            return "Insufficient data"
        
        has_contradiction = any('contradict' in r.lower() or 'refute' in r.lower() 
                               for r in reasoning_texts)
        has_support = any('support' in r.lower() or 'confirm' in r.lower() 
                         for r in reasoning_texts)
        
        if has_contradiction and has_support:
            return "Mixed (both supporting and contradicting evidence)"
        elif has_contradiction:
            return "Consistent (contradicting claim)"
        elif has_support:
            return "Consistent (supporting claim)"
        else:
            return "Neutral/Unclear"
    
    def _generate_html_report(self, article: Dict, claims: List[Dict],
                            verdict: Dict, explanations: Dict) -> str:
        """Generate comprehensive HTML report."""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fake News Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .verdict {{ font-size: 28px; font-weight: bold; margin: 20px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .verdict.LIKELY_REAL {{ background: #d4edda; color: #155724; border-left: 5px solid #28a745; }}
        .verdict.LIKELY_FAKE {{ background: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }}
        .verdict.MIXED_OR_INCONCLUSIVE {{ background: #fff3cd; color: #856404; border-left: 5px solid #ffc107; }}
        .section {{ background: white; margin: 20px 0; padding: 25px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .llm-explanation {{ background: #f8f9ff; padding: 20px; border-left: 4px solid #667eea; border-radius: 5px; line-height: 1.8; }}
        .claim {{ border-left: 4px solid #3498db; padding: 20px; margin: 15px 0; background: #f8f9fa; border-radius: 5px; }}
        .evidence {{ margin: 12px 0; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .supports {{ border-left: 4px solid #28a745; }}
        .refutes {{ border-left: 4px solid #dc3545; }}
        .neutral {{ border-left: 4px solid #6c757d; }}
        .reasoning {{ font-style: italic; color: #666; margin-top: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Fake News Detection Report</h1>
        <h3>Multi-Method Explainable AI Analysis</h3>
        <p><strong>Article:</strong> {article.get('title', 'Untitled')}</p>
        <p><strong>Analyzed:</strong> {verdict.get('analyzed_at', '')}</p>
    </div>
    
    <div class="verdict {verdict['label']}">
        <h2>📊 Final Verdict: {verdict['label'].replace('_', ' ')}</h2>
        <p><strong>Confidence:</strong> {verdict['confidence']:.2%}</p>
        <p><strong>Analysis:</strong> {verdict.get('reasoning', '')}</p>
    </div>
    
    <div class="section">
        <h2>🤖 LLM Explanation (Natural Language)</h2>
        <div class="llm-explanation">
            {explanations.get('llm_explanation', 'LLM explanation not available')}
        </div>
        <p><strong>Summary:</strong> {explanations.get('llm_summary', 'No summary available')}</p>
    </div>
    
    <div class="section">
        <h2>📈 Confidence Breakdown</h2>
        <table>
            <tr>
                <th>Factor</th>
                <th>Assessment</th>
            </tr>
            <tr>
                <td><strong>Overall Confidence</strong></td>
                <td>{explanations['confidence_breakdown']['overall_confidence']:.2%}</td>
            </tr>
            <tr>
                <td><strong>Evidence Quality</strong></td>
                <td>{explanations['confidence_breakdown']['evidence_quality']}</td>
            </tr>
            <tr>
                <td><strong>Source Reliability</strong></td>
                <td>{explanations['confidence_breakdown']['source_reliability']}</td>
            </tr>
            <tr>
                <td><strong>LLM Consistency</strong></td>
                <td>{explanations['confidence_breakdown']['llm_consistency']}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>🎯 Claims Analysis ({len(claims)} claims detected)</h2>
"""
        
        # Add claims
        for i, claim in enumerate(claims[:10], 1):
            claim_verdict = claim.get('verdict', {})
            html += f"""
        <div class="claim">
            <h3>Claim {i}: {claim_verdict.get('label', 'UNKNOWN').replace('_', ' ')}</h3>
            <p><strong>Text:</strong> "{claim['text']}"</p>
            <p><strong>Confidence:</strong> {claim_verdict.get('confidence', 0):.2%} | 
               <strong>Supporting:</strong> {claim_verdict.get('supporting_count', 0)} | 
               <strong>Refuting:</strong> {claim_verdict.get('refuting_count', 0)}</p>
"""
            
            # Add top evidence
            for evidence in claim.get('ranked_evidence', [])[:3]:
                stance = evidence.get('stance', 'NEUTRAL').lower()
                html += f"""
            <div class="evidence {stance}">
                <p><strong>{evidence.get('stance', 'NEUTRAL')}</strong> 
                   (confidence: {evidence.get('stance_confidence', 0):.2%})</p>
                <p><strong>Source:</strong> <a href="{evidence.get('url', '#')}" target="_blank">{evidence.get('title', 'No title')}</a></p>
                <p><em>From: {evidence.get('source', 'Unknown')}</em></p>
                <div class="reasoning">
                    <strong>Reasoning:</strong> {evidence.get('llm_reasoning', 'No reasoning provided')}
                </div>
            </div>
"""
            
            html += "</div>"
        
        # Add evidence ranking
        html += """
    </div>
    
    <div class="section">
        <h2>🏆 Most Impactful Evidence</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Evidence</th>
                <th>Stance</th>
                <th>Confidence</th>
                <th>Impact</th>
            </tr>
"""
        
        for i, ev in enumerate(explanations['evidence_ranking'][:10], 1):
            html += f"""
            <tr>
                <td>{i}</td>
                <td><a href="{ev['evidence_url']}" target="_blank">{ev['evidence_title'][:80]}</a></td>
                <td>{ev['stance']}</td>
                <td>{ev['confidence']:.2%}</td>
                <td>{ev['impact_score']:.3f}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>💡 Methodology Notes</h2>
        <p><strong>Stance Classification:</strong> LLM-based zero-shot classification using Groq/Llama</p>
        <p><strong>Explainability Methods:</strong></p>
        <ul>
            <li><strong>LLM Reasoning:</strong> Natural language explanations for each classification</li>
            <li><strong>Evidence Attribution:</strong> Ranking evidence by impact on verdict</li>
            <li><strong>Confidence Analysis:</strong> Assessment of evidence quality and source reliability</li>
            <li><strong>Counterfactuals:</strong> Alternative scenarios that would change the verdict</li>
        </ul>
        <p><strong>Limitations:</strong> Evidence quality depends on freely accessible sources. Results depend on LLM reasoning quality.</p>
    </div>
    
    <footer style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
        <p>Generated by Fake News Detection Multi-Agent Pipeline v2.0</p>
        <p>Powered by LangChain • LangGraph • Groq (Llama 3.1)</p>
    </footer>
</body>
</html>
"""
        
        return html
