import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import streamlit as st
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

# --- IMPORTS ---
# Using standard LangChain imports to ensure compatibility
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

# Page config
st.set_page_config(
    page_title="Legal AI Toolkit", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM LEGAL UI THEME ---
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background-color: #F4F6F9;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A; /* Deep Navy */
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stRadio div {
        color: #E2E8F0 !important; /* Off-white text */
    }

    /* Typography */
    h1, h2, h3 {
        color: #1E3A8A; /* Royal Blue */
        font-family: 'Merriweather', serif; /* Legal/Formal font */
        font-weight: 700;
    }
    p, li, .stMarkdown {
        color: #334155;
        line-height: 1.6;
    }

    /* Custom Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #172554 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.4);
        background: linear-gradient(135deg, #2563EB 0%, #1E3A8A 100%);
    }

    /* Input Fields */
    .stTextArea textarea {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        color: #1E293B;
    }
    .stTextArea textarea:focus {
        border-color: #1E3A8A;
        box-shadow: 0 0 0 2px rgba(30, 58, 138, 0.1);
    }

    /* Success/Info Messages */
    .stSuccess {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        color: #166534;
    }
    
    /* Custom Cards for Home Page */
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #1E3A8A;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: #D97706; /* Gold */
    }
    .feature-title {
        font-weight: bold;
        font-size: 1.25rem;
        color: #0F172A;
        margin-bottom: 0.5rem;
        font-family: 'Merriweather', serif;
    }
    .feature-desc {
        color: #64748B;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ensure NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Helper Functions
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = text.split()
    sw = set(stopwords.words('english'))
    lemm = WordNetLemmatizer()
    return " ".join([lemm.lemmatize(tok) for tok in tokens if tok not in sw])

@st.cache_resource
def load_pickle(path: str):
    if not os.path.exists(path):
        # Fail silently or with log, let the UI handle the missing state
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scales-of-justice.png", width=80)
    st.title("Legal AI Toolkit")
    st.markdown("---")
    mode = st.radio(
        "Navigation",
        ("Home", "Case Classification", "Case Prioritization", "Legal Precedent Search"),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("###### System Status: 🟢 Online")

# Home Screen
if mode == "Home":
    st.title("AI-Powered Legal Intelligence")
    st.markdown("#### Streamlining Case Management & Research")
    st.write("Welcome to the Legal AI Toolkit. Select a module from the sidebar to begin.")
    
    st.markdown("---")
    
    # Grid Layout for Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📂</div>
            <div class="feature-title">Case Classification</div>
            <div class="feature-desc">
                Automatically categorize legal documents into Civil, Criminal, or Constitutional domains using machine learning.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Case Prioritization</div>
            <div class="feature-desc">
                Predict the urgency level (High, Medium, Low) of incoming cases to optimize workflow and resource allocation.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Precedent Search</div>
            <div class="feature-desc">
                Leverage RAG technology to retrieve relevant past judgments and legal precedents from the Qdrant Cloud vector database.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Case Classification
if mode == "Case Classification":
    st.title("📂 Case Classification")
    st.markdown("Paste the case description below to identify its legal domain.")
    
    pipeline_path = "Case Classification/voting_pipeline.pkl"
    label_path = "Case Classification/label_encoder.pkl"

    with st.spinner("Initializing models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Case Description / Brief:", height=250, placeholder="Enter the legal facts here...")

    if st.button("Analyze Category"):
        if not text_input.strip():
            st.warning("⚠️ Please provide case details.")
        elif pipeline is None:
            st.error("❌ Model files not found. Please check directory structure.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            
            st.divider()
            st.markdown(f"### Result")
            st.success(f"**Predicted Category:** {pred_label}")
            st.info("This classification is based on textual patterns in the case description.")

# Case Prioritization
if mode == "Case Prioritization":
    st.title("⚡ Case Prioritization")
    st.markdown("Determine the urgency of a case to prioritize docket management.")
    
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"

    with st.spinner("Initializing models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Case Description / Brief:", height=250, placeholder="Enter the legal facts here...")

    if st.button("Assess Priority"):
        if not text_input.strip():
            st.warning("⚠️ Please provide case details.")
        elif pipeline is None:
            st.error("❌ Model files not found. Please check directory structure.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            
            # Custom color logic for priorities
            priority_color = "red" if "High" in pred_label else "orange" if "Medium" in pred_label else "green"
            
            st.divider()
            st.markdown(f"### Result")
            st.markdown(f"**Predicted Priority:** <span style='color:{priority_color}; font-size:1.5rem; font-weight:bold'>{pred_label}</span>", unsafe_allow_html=True)

# Legal Precedent Search (RAG)
if mode == "Legal Precedent Search":
    st.title("🔍 Legal Precedent Search")
    st.markdown("Ask natural language questions to retrieve relevant case law and precedents.")

    # --- Qdrant Configuration ---
    QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
    # ⚠️ YOUR API KEY
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.j5Kv9gmGOtLHLL4RGMJpeqzdVJSrbmsFLlNdbtvmtYs"
    COLLECTION_NAME = "legal_precedents"

    @st.cache_resource
    def load_rag_chain():
        try:
            # 1. Setup Embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # 2. Connect to Qdrant Cloud
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            # 3. Create Vector Store
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings, 
            )

            # 4. Setup LLM
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("api_key")
            if not api_key:
                return {"error": "Missing Groq API Key in .env"}
            
            llm = ChatGroq(model_name="llama3-8b-8192", api_key=api_key, temperature=0.2)

            # 5. Prompt
            prompt = ChatPromptTemplate.from_template(
                """
                You are a senior legal research assistant. 
                Use the retrieved context below to answer the user's legal question.
                If the context is insufficient, state that clearly.
                Provide citations or references to the context where possible.

                Context:
                {context}

                Question:
                {input}
                """
            )

            # 6. Chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            return {"rag_chain": rag_chain}

        except Exception as e:
            return {"error": str(e)}

    with st.spinner("Connecting to Secure Legal Database..."):
        rag_resources = load_rag_chain()

    if "error" in rag_resources:
        st.error(f"❌ Connection Error: {rag_resources['error']}")
        st.stop()

    rag_chain = rag_resources["rag_chain"]
    
    # Layout
    col_input, col_action = st.columns([4, 1])
    
    query = st.text_area("Legal Query:", height=100, placeholder="E.g., What are the precedents regarding breach of contract in construction?")
    
    if st.button("Search Database"):
        if not query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("Searching precedents and generating response..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response.get("answer")
                    
                    st.markdown("### 📝 Legal Summary")
                    st.markdown(f"""
                    <div style="background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #1E3A8A; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📚 Source Documents")
                    for i, doc in enumerate(response["context"]):
                        content = getattr(doc, 'page_content', "N/A")
                        with st.expander(f"Reference {i+1} (Click to expand)"):
                            st.info(content)
                            st.caption("Source: Qdrant Cloud Database")
                            
                except Exception as e:
                    st.error(f"An error occurred during search: {e}")
