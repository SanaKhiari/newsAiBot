"""
Groq API client for LLM inference.
Uses free Groq API with generous rate limits.
"""

"""
Groq LLM Client without conflicting retry mechanisms
"""

import os
import time
from typing import Optional
from loguru import logger
from groq import Groq

class GroqClient:
    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: Optional[str] = None):
        """Initialize Groq client"""
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        
        logger.info(f"Groq client initialized with model: {model}")
        self._check_connection()
    
    def _check_connection(self):
        """Check if Groq API is accessible"""
        try:
            response = self.client.models.list()
            models = len(response.data)
            logger.info(f"✓ Connected to Groq API")
        except Exception as e:
            logger.warning(f"Could not verify Groq connection: {e}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 300,
        timeout: int = 30
    ) -> str:
        """
        Generate text using Groq (NO retry decorator - handled by caller)
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            timeout: Request timeout in seconds
            
        Returns:
            Generated text or raises Exception
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
            
            if response and response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content.strip()
            else:
                raise ValueError("Empty response from Groq API")
                
        except Exception as e:
            # Don't retry here - let caller handle it
            raise



# Groq Models Available (Free Tier)
GROQ_MODELS = {
    "llama-3.1-8b-instant": "Fast, good for most tasks",
    "llama-3.1-70b-versatile": "More powerful, slower",
    "llama-3.2-11b-vision-preview": "Can process images",
    "mixtral-8x7b-32768": "Mixture of experts, very good",
}
