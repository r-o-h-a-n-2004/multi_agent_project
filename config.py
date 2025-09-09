import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # DuckDuckGo doesn't need API keys!
    # Model configuration (you might still want OpenAI for the LLM)
    MODEL_NAME = "gpt-4o-mini"  # or use a local model
    MODEL_TEMPERATURE = 0.1
    
    # Search configuration
    MAX_SEARCH_RESULTS = 5