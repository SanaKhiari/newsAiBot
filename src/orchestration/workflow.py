"""
Workflow Builder: Constructs and executes the LangGraph.
This is where all nodes are connected into a complete workflow.
"""
from typing import Dict
from datetime import datetime
from loguru import logger
from langgraph.graph import StateGraph, END

from src.orchestration.state_graph import FakeNewsState
from src.orchestration.nodes import (
    ingest_node,
    claim_detection_node,
    embedding_node,
    evidence_retrieval_node,
    stance_classification_node,
    aggregation_node,
    explainability_node,
    error_node,
)

# ========== CONDITIONAL ROUTING FUNCTIONS ==========

def should_continue_after_ingest(state: FakeNewsState) -> str:
    """
    Route after ingest: continue if article is valid.
    If ingest failed, go to error node.
    If ingest succeeded, go to claim detection.
    """
    if state.get('error'):
        return "error_handler"
    if state.get('article'):
        return "claim_detection"
    return "error_handler"

def should_continue_after_claims(state: FakeNewsState) -> str:
    """
    Route after claim detection: continue if claims found.
    If no claims found, verdict will be inconclusive.
    """
    if state.get('error'):
        return "error_handler"
    if state.get('claims') and len(state['claims']) > 0:
        return "embedding"

    # No claims found - create inconclusive verdict
    logger.warning("No claims detected in article")
    state['verdict'] = {
        'label': 'INCONCLUSIVE',
        'confidence': 0.0,
        'claim_count': 0,
        'real_claims': 0,
        'fake_claims': 0,
        'inconclusive_claims': 0,
        'reasoning': 'No verifiable claims detected in article'
    }

    return "explainability"

def should_continue_after_evidence(state: FakeNewsState) -> str:
    """Route after evidence retrieval."""
    if state.get('error'):
        return "error_handler"
    return "stance_classification"

# ========== WORKFLOW BUILDER ==========

def build_workflow() -> object:
    """
    Build the LangGraph workflow.
    1. Creates a new StateGraph
    2. Adds all nodes
    3. Connects nodes with edges
    4. Compiles the graph
    Returns:
        Compiled LangGraph workflow
    """
    logger.info("=" * 60)
    logger.info("BUILDING LANGGRAPH WORKFLOW")
    logger.info("=" * 60)

    # Create graph
    workflow = StateGraph(FakeNewsState)

    # ========== ADD NODES ==========
    logger.info("Adding nodes...")

    workflow.add_node("ingest", ingest_node)
    workflow.add_node("claim_detection", claim_detection_node)
    workflow.add_node("embedding", embedding_node)
    workflow.add_node("evidence_retrieval", evidence_retrieval_node)
    workflow.add_node("stance_classification", stance_classification_node)
    workflow.add_node("aggregation", aggregation_node)
    workflow.add_node("explainability", explainability_node)
    workflow.add_node("error_handler", error_node)

    logger.info("✓ 8 nodes added")

    # ========== SET ENTRY POINT ==========
    logger.info("Setting entry point...")
    workflow.set_entry_point("ingest")
    logger.info("✓ Entry point: ingest")

    # ========== ADD CONDITIONAL EDGES ==========
    logger.info("Adding conditional edges...")

    workflow.add_conditional_edges(
        "ingest",
        should_continue_after_ingest,
        {
            "claim_detection": "claim_detection",
            "error_handler": "error_handler"
        }
    )
    logger.info("  ✓ ingest → [claim_detection | error_handler]")

    workflow.add_conditional_edges(
        "claim_detection",
        should_continue_after_claims,
        {
            "embedding": "embedding",
            "explainability": "explainability",
            "error_handler": "error_handler"
        }
    )
    logger.info("  ✓ claim_detection → [embedding | explainability | error_handler]")

    workflow.add_conditional_edges(
        "evidence_retrieval",
        should_continue_after_evidence,
        {
            "stance_classification": "stance_classification",
            "error_handler": "error_handler"
        }
    )
    logger.info("  ✓ evidence_retrieval → [stance_classification | error_handler]")

    # ========== ADD LINEAR EDGES ==========
    logger.info("Adding linear edges...")

    workflow.add_edge("embedding", "evidence_retrieval")
    logger.info("  ✓ embedding → evidence_retrieval")

    workflow.add_edge("stance_classification", "aggregation")
    logger.info("  ✓ stance_classification → aggregation")

    workflow.add_edge("aggregation", "explainability")
    logger.info("  ✓ aggregation → explainability")

    workflow.add_edge("explainability", END)
    logger.info("  ✓ explainability → END")

    workflow.add_edge("error_handler", END)
    logger.info("  ✓ error_handler → END")

    # ========== COMPILE GRAPH ==========
    logger.info("Compiling workflow...")
    app = workflow.compile()

    logger.info("=" * 60)
    logger.info("✓ WORKFLOW COMPILED SUCCESSFULLY")
    logger.info("=" * 60)

    return app

# ========== WORKFLOW EXECUTOR ==========

def run_workflow(url: str = None, text: str = None, title: str = None) -> Dict:
    """
    Run the complete workflow.
    1. Creates initial state
    2. Builds workflow
    3. Executes workflow
    4. Returns final state
    Returns:
        Final state dictionary with all results
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING FAKE NEWS DETECTION WORKFLOW")
    logger.info("=" * 80)
    # Validate input
    if not url and not text:
        logger.error("Must provide either url or text")
        return {
            "error_handler": "Must provide either url or text",
            "current_step": "start"
        }

    # Create initial state
    initial_state = FakeNewsState.create_initial(url=url, text=text, title=title)

    logger.info(f"Initial state created")
    logger.info(f"  URL: {url if url else 'None'}")
    logger.info(f"  Text length: {len(text) if text else 0} chars")
    logger.info(f"  Title: {title if title else 'None'}")

    try:
        app = build_workflow()
        logger.info("\nExecuting workflow...")
        final_state = app.invoke(initial_state)

        logger.info("\n" + "=" * 80)
        logger.info("WORKFLOW COMPLETED")
        logger.info("=" * 80)

        # Defensive: Ensure verdict/explanations present
        verdict = final_state.get('verdict')
        if not verdict or not isinstance(verdict, dict):
            logger.warning("No verdict produced by workflow, setting default INCONCLUSIVE verdict")
            final_state['verdict'] = {
                'label': 'INCONCLUSIVE',
                'confidence': 0.0,
                'score': 0.0,
                'supporting_count': 0,
                'refuting_count': 0,
                'neutral_count': 0,
                'reasoning': 'Workflow did not produce a verdict, possibly due to internal error.'
            }
        explanations = final_state.get('explanations')
        if not explanations or not isinstance(explanations, dict):
            logger.warning("No explanations produced, defaulting to empty dict.")
            final_state['explanations'] = {}

        if final_state.get('error_handler'):
            logger.error(f"Workflow finished with error_handler: {final_state['error_handler']}")
        else:
            logger.info(f"Final verdict: {final_state.get('verdict', {}).get('label', 'UNKNOWN')}")
        return final_state

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        return {
            "error_handler": str(e),
            "current_step": initial_state.get('current_step', 'unknown'),
            "started_at": initial_state.get('started_at'),
            "completed_at": datetime.now().isoformat()
        }

def visualize_workflow():
    """
    Print a text representation of the workflow.
    Useful for understanding the pipeline structure.
    """
    logger.info("""
    
    WORKFLOW STRUCTURE:
    
    START
      ↓
    [INGEST] - Parse article from URL or text
      ↓
    [CLAIM DETECTION] - Extract factual claims
      ↓ (if no claims)
      → [EXPLAINABILITY] (skip to final)
      ↓ (if claims found)
    [EMBEDDING] - Generate semantic embeddings
      ↓
    [EVIDENCE RETRIEVAL] - Search for supporting/refuting evidence
      ↓
    [STANCE CLASSIFICATION] - LLM classifies evidence stance
      ↓
    [AGGREGATION] - Combine results into verdict
      ↓
    [EXPLAINABILITY] - Generate multi-method explanations
      ↓
    END (or error_handler → END)
    
    """)
