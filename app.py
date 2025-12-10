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
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
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

# --- SAFE & CLEAN UI THEME (FIXED) ---
st.markdown(
    """
    <style>
    /* 1. Force Light Mode Backgrounds to prevent Dark Mode conflicts */
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] {
        background-color: #F0F2F6 !important; /* Light Grey Sidebar */
        border-right: 1px solid #D1D5DB;
    }
    
    /* 2. Force Text Colors to Black/Dark Grey */
    h1, h2, h3, h4, h5, h6, p, li, span, div, label {
        color: #111827 !important; /* Almost Black */
        font-family: 'Helvetica', sans-serif;
    }
    
    /* 3. High Visibility Buttons */
    div.stButton > button {
        background-color: #1E3A8A !important; /* Navy Blue */
        color: #FFFFFF !important; /* White Text */
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8 !important; /* Brighter Blue */
        border: 2px solid #000000 !important;
    }

    /* 4. Fix Input Fields Text Color */
    .stTextArea textarea, .stTextInput input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #9CA3AF !important;
    }

    /* 5. Custom Card Styling */
    .feature-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #000000 !important;
    }
    .result-box {
        background-color: #F0FDF4; /* Light Green */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #16A34A;
        margin-top: 20px;
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
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

# Sidebar Content
with st.sidebar:
    st.title("⚖️ Legal AI Toolkit")
    st.markdown("---")
    mode = st.radio(
        "Select Module:",
        ("Home", "Case Classification", "Case Prioritization", "Legal Precedent Search")
    )
    st.markdown("---")
    st.info("System Online ✅")

# Home Screen
if mode == "Home":
    st.title("Legal Intelligence Dashboard")
    st.markdown("Select a tool from the sidebar to begin.")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📂 Classification</h3>
            <p>Categorize cases into Civil, Criminal, or Constitutional domains.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Prioritization</h3>
            <p>Assess case urgency (High/Medium/Low) for docket management.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Precedents</h3>
            <p>Search case law using AI-powered semantic search.</p>
        </div>
        """, unsafe_allow_html=True)

# Case Classification
if mode == "Case Classification":
    st.title("📂 Case Classification")
    
    pipeline_path = "Case Classification/voting_pipeline.pkl"
    label_path = "Case Classification/label_encoder.pkl"

    with st.spinner("Loading models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    st.markdown("**Enter Case Description:**")
    text_input = st.text_area("brief", height=200, label_visibility="collapsed", placeholder="Type here...")

    if st.button("Predict Category"):
        if not text_input.strip():
            st.warning("Please enter text.")
        elif pipeline is None:
            st.error("Model files not found.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            
            st.markdown(f"""
            <div class="result-box" style="background-color: #EFF6FF; border-color: #2563EB;">
                <h3>Predicted Category: {pred_label}</h3>
            </div>
            """, unsafe_allow_html=True)

# Case Prioritization
if mode == "Case Prioritization":
    st.title("⚡ Case Prioritization")
    
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"

    with st.spinner("Loading models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    st.markdown("**Enter Case Description:**")
    text_input = st.text_area("brief", height=200, label_visibility="collapsed", placeholder="Type here...")

    if st.button("Predict Priority"):
        if not text_input.strip():
            st.warning("Please enter text.")
        elif pipeline is None:
            st.error("Model files not found.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            
            colors = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
            color = colors.get(pred_label, "#3B82F6")
            
            st.markdown(f"""
            <div class="result-box" style="background-color: white; border-left: 10px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: {color} !important;">{pred_label} Priority</h2>
            </div>
            """, unsafe_allow_html=True)

# Legal Precedent Search (RAG)
if mode == "Legal Precedent Search":
    st.title("🔍 Legal Precedent Search")
    
    # Qdrant Config
    QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.j5Kv9gmGOtLHLL4RGMJpeqzdVJSrbmsFLlNdbtvmtYs"
    COLLECTION_NAME = "legal_precedents"

    @st.cache_resource
    def load_rag_chain():
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings, 
            )

            api_key = os.getenv("GROQ_API_KEY") or os.getenv("api_key")
            if not api_key: return {"error": "Missing Groq API Key in .env"}
            
            llm = ChatGroq(model_name="llama3-8b-8192", api_key=api_key, temperature=0.1)

            prompt = ChatPromptTemplate.from_template(
                """
                You are a legal assistant. Answer based strictly on the context below.
                Structure the answer as:
                1. Summary
                2. Key Precedents
                3. Conclusion

                Context:
                {context}

                Question:
                {input}
                """
            )

            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            rag_chain = create_retrieval_chain(retriever, document_chain)
            return {"rag_chain": rag_chain}

        except Exception as e:
            return {"error": str(e)}

    with st.spinner("Connecting to Database..."):
        rag_resources = load_rag_chain()

    if "error" in rag_resources:
        st.error(rag_resources["error"])
        st.stop()

    rag_chain = rag_resources["rag_chain"]
    
    st.markdown("**Legal Query:**")
    query = st.text_area("query", height=100, label_visibility="collapsed", placeholder="Enter your legal question here...")
    
    if st.button("Search Precedents"):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response.get("answer")
                    
                    st.markdown("### 📝 Legal Memo")
                    st.markdown(f"""
                    <div style="background-color: #F8FAFC; padding: 25px; border-radius: 5px; border: 1px solid #E2E8F0; color: #000;">
                        {answer.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📚 References")
                    for i, doc in enumerate(response["context"]):
                        content = getattr(doc, 'page_content', "N/A")
                        with st.expander(f"Reference Source {i+1}"):
                            st.write(content)
                            
                except Exception as e:
                    st.error(f"Error: {e}")
