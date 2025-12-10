# config.py
import os
import streamlit as st
from dotenv import load_dotenv

# Load local .env file
load_dotenv(dotenv_path=".env", override=True)

# Constants
QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "legal_precedents"

# Paths
VOTING_PIPELINE_PATH = "Case Cateogarization/voting_pipeline.pkl"
LABEL_ENCODER_PATH = "Case Cateogarization/label_encoder.pkl"
PRIORITY_PIPELINE_PATH = "Case Prioritization/stacking_pipeline.pkl"
PRIORITY_ENCODER_PATH = "Case Prioritization/label_encoder.pkl"

def get_groq_api_key():
    """Robust API Key Retrieval"""
    # 1. Streamlit Secrets (Cloud)
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, AttributeError):
        pass
    
    # 2. Local Environment
    if os.getenv("GROQ_API_KEY"):
        return os.getenv("GROQ_API_KEY")
    
    return None

def get_qdrant_api_key():
    # Similar logic for Qdrant
    if "QDRANT_API_KEY" in st.secrets:
        return st.secrets["QDRANT_API_KEY"]
    return os.getenv("QDRANT_API_KEY")
