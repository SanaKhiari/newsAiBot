"""
FastAPI Server: Main REST API application.
Endpoints for analyzing articles and health checks.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
)
from src.orchestration.workflow import run_workflow
from src.config import settings
from src.utils import save_json, sanitize_filename


# ========== LOGGING SETUP ==========

# Remove default logger
logger.remove()

# Add stderr logger
logger.add(
    sys.stderr,
    level=settings.log_level.upper(),
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Add file logger
log_file = Path(settings.log_dir) / "api_{time}.log"
logger.add(
    log_file,
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

logger.info("API Logger initialized")


# ========== FASTAPI APP INITIALIZATION ==========

app = FastAPI(
    title="Fake News Detection API",
    description="Multi-Agent LLM-Powered Fake News Detection System",
    version="2.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# ========== CORS MIDDLEWARE ==========
# Allows requests from any origin (for development)
# In production, restrict to specific domains

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

logger.info("CORS middleware configured")


# ========== STARTUP & SHUTDOWN ==========

@app.on_event("startup")
async def startup_event():
    """
    Run when API starts.
    Create directories, initialize resources.
    """
    logger.info("\n" + "="*80)
    logger.info("FAKE NEWS DETECTION API STARTING")
    logger.info("="*80)
    
    # Create required directories
    directories = [
        settings.output_dir,
        settings.log_dir,
        f"{settings.output_dir}/reports",
        settings.cache_dir,
        settings.model_dir,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✓ Directories created")
    logger.info(f"  - Output: {settings.output_dir}")
    logger.info(f"  - Logs: {settings.log_dir}")
    logger.info(f"  - Reports: {settings.output_dir}/reports")
    
    # Verify LLM is available
    try:
        from src.llm.groq_client import GroqClient
        client = GroqClient()
        logger.info(f"✓ Groq LLM initialized: {settings.groq_model}")
    except Exception as e:
        logger.warning(f"✗ Groq LLM not available: {e}")
        logger.warning("  API will run but LLM explanations will be disabled")
    
    logger.info("✓ API startup complete")
    logger.info("="*80 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Run when API shuts down."""
    logger.info("\n" + "="*80)
    logger.info("FAKE NEWS DETECTION API SHUTTING DOWN")
    logger.info("="*80)
    logger.info("✓ Cleanup complete")


# ========== ROOT ENDPOINT ==========

@app.get("/", tags=["Info"])
async def root():
    """
    Root endpoint with API info.
    
    Returns:
        API metadata and available endpoints
    """
    return {
        "message": "Fake News Detection API v2.0 - LLM Powered",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "LangChain multi-agent framework",
            "LangGraph workflow orchestration",
            "LLM-based stance classification (Groq/Llama 3.1)",
            "Multi-method XAI (LLM + LIME + SHAP)",
            "Real-time evidence retrieval (GDELT, RSS, CommonCrawl)",
            "HTML reports with detailed analysis"
        ],
        "endpoints": {
            "analyze": "/api/analyze (POST)",
            "health": "/health (GET)",
            "documentation": "/docs (GET)",
            "report": "/api/report/{filename} (GET)"
        },
        "llm_provider": settings.llm_provider,
        "llm_model": settings.groq_model
    }


# ========== HEALTH CHECK ENDPOINT ==========

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check():
    """
    Health check endpoint.
    
    Verifies API and LLM are operational.
    
    Returns:
        Health status including Groq LLM status
    """
    logger.debug("Health check requested")
    
    # Check Groq connection
    groq_status = "healthy"
    try:
        from src.llm.groq_client import GroqClient
        client = GroqClient()
        # Try a quick API call
        response = client.client.models.list()
        if response:
            groq_status = "healthy"
    except Exception as e:
        logger.warning(f"Groq health check failed: {e}")
        groq_status = "unavailable"
    
    return {
        "status": "healthy",
        "ollama_status": groq_status,
        "timestamp": datetime.now().isoformat()
    }


# ========== MAIN ANALYSIS ENDPOINT ==========

@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    responses={
        200: {"description": "Analysis successful"},
        400: {"description": "Invalid request"},
        500: {"description": "Server error"}
    },
    tags=["Analysis"],
    summary="Analyze Article for Fake News"
)
async def analyze_article(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Analyze an article for fake news.
    
    This is the main endpoint. It:
    1. Validates input
    2. Runs LangGraph workflow
    3. Generates explanations
    4. Returns results
    
    Args:
        request: Article (URL or text)
        background_tasks: Tasks to run after response
    
    Returns:
        Analysis results with verdict and evidence
        
    Example:
        ```
        POST /api/analyze
        {
            "text": "Apple announced iPhone 15...",
            "title": "iPhone 15 Release"
        }
        ```
    """
    
    logger.info("\n" + "="*80)
    logger.info("NEW ANALYSIS REQUEST")
    logger.info("="*80)
    
    # ========== VALIDATION ==========
    
    if not request.url and not request.text:
        logger.error("Invalid request: Must provide either url or text")
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'url' or 'text' field"
        )
    
    if request.text and len(request.text.strip()) < 50:
        logger.error("Text too short: Must be at least 50 characters")
        raise HTTPException(
            status_code=400,
            detail="Text must be at least 50 characters"
        )
    
    logger.info(f"✓ Request validated")
    logger.info(f"  Input: {'URL' if request.url else 'Text'}")
    logger.info(f"  Title: {request.title or 'None'}")
    
    try:
        # ========== EXECUTE WORKFLOW ==========
        
        logger.info("\n→ Starting workflow execution...")
        
        url_str = str(request.url) if request.url else None
        
        final_state = run_workflow(
            url=url_str,
            text=request.text,
            title=request.title
        )
        
        # ========== ERROR HANDLING ==========
        
        if final_state.get('error'):
            logger.error(f"Workflow error: {final_state['error']}")
            raise HTTPException(
                status_code=500,
                detail=final_state['error']
            )
        
        # ========== PREPARE RESPONSE ==========
        
        logger.info("\n→ Preparing response...")
        
        # Extract key data
        verdict = final_state.get('verdict', {})
        claims = final_state.get('claims', [])
        explanations = final_state.get('explanations', {})
        
        response = {
            'status': 'success',
            'verdict': verdict,
            'claims': claims,
            'explanations': explanations,
            'metadata': {
                'started_at': final_state.get('started_at'),
                'completed_at': final_state.get('completed_at'),
                'evidence_count': final_state.get('evidence_count', 0),
                'claim_count': len(claims),
                'analysis_duration': _calculate_duration(
                    final_state.get('started_at'),
                    final_state.get('completed_at')
                )
            }
        }
        
        # ========== BACKGROUND TASKS ==========
        
        # Save report in background (don't block response)
        if explanations.get('html_report'):
            background_tasks.add_task(
                _save_report,
                response,
                final_state.get('article', {})
            )
        
        # Save JSON result in background
        background_tasks.add_task(
            _save_json_result,
            response
        )
        
        logger.info(f"\n✓ Analysis completed successfully")
        logger.info(f"  Verdict: {verdict.get('label')}")
        logger.info(f"  Confidence: {verdict.get('confidence', 0):.2%}")
        logger.info(f"  Claims: {len(claims)}")
        logger.info("="*80 + "\n")
        
        # ========== REMOVE EMBEDDINGS FROM RESPONSE ==========
        
        # Embeddings are large and not needed in response
        response = _remove_embeddings(response)
        
        return JSONResponse(content=response)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)  # ← FIX: Use direct string, not logger attribute
        logger.error("="*80)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# ========== REPORT DOWNLOAD ENDPOINT ==========

@app.get(
    "/api/report/{filename}",
    response_class=HTMLResponse,
    tags=["Reports"],
    summary="Download HTML Report"
)
async def get_report(filename: str):
    """
    Download HTML report for a previous analysis.
    
    Args:
        filename: Report filename
        
    Returns:
        HTML report
        
    Example:
        GET /api/report/my-analysis-2025-10-28.html
    """
    logger.info(f"Report requested: {filename}")
    
    # Sanitize filename to prevent directory traversal
    filename = sanitize_filename(filename)
    
    report_path = Path(settings.output_dir) / "reports" / filename
    
    if not report_path.exists():
        logger.warning(f"Report not found: {filename}")
        raise HTTPException(
            status_code=404,
            detail=f"Report '{filename}' not found"
        )
    
    logger.info(f"✓ Report found: {filename}")
    
    html_content = report_path.read_text(encoding='utf-8')
    return HTMLResponse(content=html_content)


# ========== HELPER FUNCTIONS ==========

def _calculate_duration(start_iso: str, end_iso: str) -> str:
    """Calculate human-readable duration."""
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        duration = (end - start).total_seconds()
        
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"
    except:
        return "unknown"


def _save_report(result: dict, article: dict):
    """
    Save HTML report to disk in background.
    
    Args:
        result: Analysis result
        article: Article metadata
    """
    try:
        html = result['explanations'].get('html_report')
        if not html:
            return
        
        title = article.get('title', 'analysis')
        filename = sanitize_filename(title) + '.html'
        
        report_dir = Path(settings.output_dir) / "reports"
        report_path = report_dir / filename
        
        report_path.write_text(html, encoding='utf-8')
        
        logger.info(f"✓ Report saved: {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")


def _save_json_result(result: dict):
    """
    Save analysis result as JSON in background.
    
    Args:
        result: Analysis result
    """
    try:
        verdict = result['verdict']
        title = verdict.get('article_title', 'analysis')
        filename = sanitize_filename(title) + '.json'
        
        result_path = Path(settings.output_dir) / "results" / filename
        
        save_json(result, result_path)
        
        logger.debug(f"✓ Result JSON saved: {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save result JSON: {e}")


def _remove_embeddings(result: dict) -> dict:
    """
    Remove embedding vectors from response (they're large).
    
    Args:
        result: Analysis result
        
    Returns:
        Result without embeddings
    """
    if 'claims' in result:
        for claim in result['claims']:
            claim.pop('embedding', None)
    
    return result


# ========== ERROR HANDLERS ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP Exception: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.detail,
            "detail": str(exc.detail)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ========== MAIN ENTRY POINT ==========

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Fake News Detection API Server")
    
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True if settings.environment == "development" else False,
        log_level=settings.log_level,
        workers=1
    )
