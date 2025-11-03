"""
Flask Application for Fake News Detection System
Uses the orchestrated LangGraph workflow with Groq LLM
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.orchestration.workflow import run_workflow
from src.config import settings

# ========== FLASK INITIALIZATION ==========

app = Flask(__name__, template_folder='templates')
app.config['JSON_SORT_KEYS'] = False
CORS(app)

# ========== LOGGING SETUP ==========

logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level.upper(),
    format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

log_file = Path(settings.log_dir) / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(
    log_file,
    rotation="1 day",
    retention="7 days",
    level="INFO"
)

logger.info("Flask application initialized")

# ========== STARTUP ==========

def startup():
    """Initialize app on startup"""
    logger.info("\n" + "="*80)
    logger.info("FAKE NEWS DETECTION SYSTEM - DEPLOYMENT")
    logger.info("="*80)
    
    # Create required directories
    for directory in [settings.output_dir, settings.log_dir, f"{settings.output_dir}/reports", 
                      settings.cache_dir, settings.model_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Directories created")
    logger.info("✓ Flask app ready")
    logger.info("="*80 + "\n")

startup()

# ========== ROUTES ==========

@app.route('/')
def index():
    """Serve main UI"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "service": "Fake News Detective",
        "version": "2.0.0"
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    logger.info("\n" + "="*80)
    logger.info("NEW ANALYSIS REQUEST")
    logger.info("="*80)
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No JSON data"}), 400
        
        url = data.get('url')
        text = data.get('text')
        title = data.get('title') or 'Untitled'
        
        if not url and not text:
            return jsonify({"status": "error", "error": "Provide url or text"}), 400
        
        if text and len(text.strip()) < 50:
            return jsonify({"status": "error", "error": "Text must be 50+ chars"}), 400
        
        logger.info(f"✓ Request validated: {('URL' if url else 'Text')} | Title: {title}")
        logger.info("→ Starting workflow...")
        
        final_state = run_workflow(url=url, text=text, title=title)
        
        if final_state.get('error_handler'):
            logger.error(f"Workflow error: {final_state['error_handler']}")
            return jsonify({"status": "error", "error": final_state['error_handler']}), 500
        
        verdict = final_state.get('verdict', {})
        claims = final_state.get('claims', [])
        explanations = final_state.get('explanations', {})
        
        # Remove embeddings (too large)
        for claim in claims:
            claim.pop('embedding', None)
        
        response = {
            'status': 'success',
            'verdict': verdict,
            'claims': claims,
            'explanations': explanations,
            'metadata': {
                'started_at': final_state.get('started_at'),
                'completed_at': final_state.get('completed_at'),
                'analysis_duration': final_state.get('completed_at') and final_state.get('started_at') and 
                    f"{(datetime.fromisoformat(final_state['completed_at']) - datetime.fromisoformat(final_state['started_at'])).total_seconds():.1f}s" or 'N/A',
                'evidence_count': final_state.get('evidence_count', 0),
                'claim_count': len(claims)
            }
        }
        
        logger.info(f"✓ Analysis complete: {verdict.get('label')} ({verdict.get('confidence', 0):.0%})")
        logger.info("="*80 + "\n")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"✗ Analysis failed: {e}", exc_info=True)
        logger.error("="*80 + "\n")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Not found"}), 404

@app.errorhandler(500)
def error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"status": "error", "error": "Server error"}), 500

# ========== MAIN ==========

if __name__ == '__main__':
    app.run(
        host=settings.api_host,
        port=settings.api_port,
        debug=(settings.environment == 'development'),
        threaded=True
    )
