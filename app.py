"""
NeuroRAG - ICD-10 Mental Health Diagnostic Assistant
Deployed on Hugging Face Spaces

This is the main entry point for Hugging Face deployment.
"""

import os
import sys

# Set environment variables for Hugging Face
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/huggingface'

# Import the Flask app from run_server.py
from run_server import app

if __name__ == "__main__":
    # Hugging Face Spaces uses port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    
    print(f"🚀 Starting NeuroRAG on port {port}...")
    print(f"📊 Loading FAISS index and embeddings...")
    
    # Run the Flask app
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False  # Disable debug mode in production
    )
