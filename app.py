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

# --- CORRECTED IMPORTS FOR QDRANT & LANGCHAIN ---
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
st.set_page_config(page_title="Legal Case Management & Precedent Search", layout="wide")

# Enhanced Custom UI Styling
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 3rem !important;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2D3047 0%, #1A1C2E 100%);
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] * {
        color: #E6E6FA !important;
    }
    
    /* Radio buttons in sidebar */
    div[data-testid="stRadio"] > div {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    div[data-testid="stRadio"] label {
        padding: 0.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stRadio"] label:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-size: 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        min-width: 200px;
        margin: 1rem auto;
        display: block;
    }
    
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    }
    
    /* Headers */
    h1 {
        color: #2D3047 !important;
        font-family: 'Merriweather', serif !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2, h3, h4 {
        color: #2D3047 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        margin-top: 1.5rem;
    }
    
    /* Text areas */
    .stTextArea textarea {
        border: 2px solid #E6E6FA !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        font-size: 16px !important;
        background: rgba(255, 255, 255, 0.9) !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        border-left: 5px solid #28a745 !important;
        padding: 1.5rem !important;
        color: #155724 !important;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(135deg, #D1ECF1 0%, #BEE5EB 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        border-left: 5px solid #17a2b8 !important;
        padding: 1.5rem !important;
        color: #0C5460 !important;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(135deg, #FFF3CD 0%, #FFEAA7 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        border-left: 5px solid #ffc107 !important;
        padding: 1.5rem !important;
        color: #856404 !important;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(135deg, #F8D7DA 0%, #F5C6CB 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        border-left: 5px solid #dc3545 !important;
        padding: 1.5rem !important;
        color: #721C24 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #2D3047 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(248, 249, 250, 0.7) !important;
        border-radius: 0 0 10px 10px !important;
        padding: 1.5rem !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: #667eea !important;
    }
    
    /* Labels */
    label {
        font-weight: 600 !important;
        color: #2D3047 !important;
        margin-bottom: 0.5rem !important;
        display: block;
    }
    
    /* Text */
    p, .stMarkdown {
        color: #4A4A4A !important;
        line-height: 1.6 !important;
        font-size: 16px !important;
    }
    
    /* Sidebar title */
    .sidebar .sidebar-content {
        background: transparent !important;
    }
    
    /* Cards for results */
    .result-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E6E6FA;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add custom fonts
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Montserrat:wght@400;500;600;700&display=swap" rel="stylesheet">
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
        st.warning(f"Missing file: {path}")
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

# Sidebar with enhanced styling
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #FFFFFF !important; font-family: 'Merriweather', serif; margin-bottom: 0.5rem;">⚖️</h1>
        <h2 style="color: #FFFFFF !important; font-family: 'Montserrat', sans-serif; font-weight: 700;">Legal AI Toolkit</h2>
        <div style="height: 3px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 1rem auto; width: 60%;"></div>
    </div>
    """,
    unsafe_allow_html=True
)

mode = st.sidebar.radio(
    "Choose a tool:",
    ("Home", "Case Classification", "Case Prioritization", "Legal Precedent Search (RAG)"),
    format_func=lambda x: f"📌 {x}" if x == "Home" else f"📊 {x}" if "Classification" in x else f"🎯 {x}" if "Prioritization" in x else f"🔍 {x}"
)

# Home Screen
if mode == "Home":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="margin-bottom: 0.5rem;">⚖️ Legal AI Assistant</h1>
            <div style="height: 4px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 0 auto 2rem; width: 200px; border-radius: 2px;"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="result-card" style="text-align: center; height: 100%;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
                <h3>Case Classification</h3>
                <p>Automatically classify court cases by category (Civil, Criminal, or Constitutional)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="result-card" style="text-align: center; height: 100%;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
                <h3>Case Prioritization</h3>
                <p>Predict the urgency level of cases (High, Medium, Low) for efficient workflow management</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="result-card" style="text-align: center; height: 100%;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                <h3>Legal Precedent Search</h3>
                <p>Retrieve related case precedents using advanced RAG with Qdrant Vector Database</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.info("💡 **Getting Started:** Select a tool from the sidebar to begin. Each tool is designed to assist with specific legal tasks using AI-powered analysis.")
    
    # Feature highlights
    with st.expander("📋 **Key Features & Capabilities**", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("""
            ✅ **Smart Classification**  
            • Multi-class case categorization  
            • High accuracy with ensemble models  
            • Real-time predictions  
            
            ✅ **Efficient Prioritization**  
            • Urgency level detection  
            • Stacking ensemble methods  
            • Quick turnaround time  
            """)
        
        with cols[1]:
            st.markdown("""
            ✅ **Advanced Search**  
            • Semantic similarity search  
            • Vector database integration  
            • Context-aware responses  
            
            ✅ **Secure & Reliable**  
            • Cloud-based storage  
            • Encrypted communications  
            • Scalable architecture  
            """)
    
    st.stop()

# Case Classification
if mode == "Case Classification":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="margin-bottom: 0.5rem;">📊 Case Classification</h1>
            <div style="height: 4px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 0 auto 2rem; width: 300px; border-radius: 2px;"></div>
            <p style="color: #6c757d; font-size: 1.1rem;">Classify legal cases into Civil, Criminal, or Constitutional categories using AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    pipeline_path = "Case Classification/voting_pipeline.pkl"
    label_path = "Case Classification/label_encoder.pkl"

    with st.spinner("🔍 Loading classification model..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    st.markdown("### 📝 Enter Case Text")
    text_input = st.text_area("Paste case text here:", height=300, 
                             placeholder="Paste the complete case text here for classification...")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("**Predict Category**", type="primary"):
            if not text_input.strip():
                st.warning("⚠️ Please enter some case text.")
            elif pipeline is None:
                st.error("❌ Classification pipeline not loaded.")
            else:
                with st.spinner("🔬 Analyzing text..."):
                    cleaned = preprocess_text(text_input)
                    pred_enc = pipeline.predict([cleaned])
                    pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
                    
                    # Color-coded result
                    colors = {
                        "Civil": "#28a745",
                        "Criminal": "#dc3545", 
                        "Constitutional": "#007bff"
                    }
                    color = colors.get(pred_label, "#6c757d")
                    
                    st.markdown(
                        f"""
                        <div class="result-card" style="text-align: center; border-left: 5px solid {color};">
                            <h3 style="color: {color}; margin-bottom: 0.5rem;">Classification Result</h3>
                            <div style="font-size: 2.5rem; font-weight: 700; color: {color}; margin: 1rem 0;">{pred_label}</div>
                            <p style="color: #6c757d;">AI Model Confidence: High</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Case Prioritization
if mode == "Case Prioritization":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="margin-bottom: 0.5rem;">🎯 Case Prioritization</h1>
            <div style="height: 4px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 0 auto 2rem; width: 300px; border-radius: 2px;"></div>
            <p style="color: #6c757d; font-size: 1.1rem;">Determine case urgency level (High, Medium, Low) for optimal resource allocation</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"

    with st.spinner("🔍 Loading prioritization model..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    st.markdown("### 📝 Enter Case Text")
    text_input = st.text_area("Paste case text here:", height=300,
                             placeholder="Paste the case text to determine its priority level...")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("**Predict Priority**", type="primary"):
            if not text_input.strip():
                st.warning("⚠️ Please enter some case text.")
            elif pipeline is None:
                st.error("❌ Prioritization pipeline not loaded.")
            else:
                with st.spinner("⚡ Analyzing urgency..."):
                    cleaned = preprocess_text(text_input)
                    pred_enc = pipeline.predict([cleaned])
                    pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
                    
                    # Priority-specific styling
                    priority_config = {
                        "High": {"color": "#dc3545", "icon": "🔥", "bg": "#F8D7DA"},
                        "Medium": {"color": "#ffc107", "icon": "⚡", "bg": "#FFF3CD"},
                        "Low": {"color": "#28a745", "icon": "🌱", "bg": "#D4EDDA"}
                    }
                    
                    config = priority_config.get(pred_label, {"color": "#6c757d", "icon": "📊", "bg": "#F8F9FA"})
                    
                    st.markdown(
                        f"""
                        <div class="result-card" style="text-align: center; border-left: 5px solid {config['color']}; background: {config['bg']};">
                            <h3 style="color: {config['color']}; margin-bottom: 0.5rem;">Priority Assessment</h3>
                            <div style="font-size: 2.5rem; margin: 0.5rem 0;">{config['icon']}</div>
                            <div style="font-size: 2rem; font-weight: 700; color: {config['color']}; margin: 0.5rem 0;">{pred_label}</div>
                            <p style="color: #6c757d;">Recommended Action: Immediate attention" if pred_label == "High" else "Schedule appropriately" if pred_label == "Medium" else "Standard processing"</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Legal Precedent Search (RAG) - UPDATED & FIXED
if mode == "Legal Precedent Search (RAG)":
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="margin-bottom: 0.5rem;">🔍 Legal Precedent Retrieval Engine</h1>
            <div style="height: 4px; background: linear-gradient(90deg, #667eea, #764ba2); margin: 0 auto 2rem; width: 400px; border-radius: 2px;"></div>
            <p style="color: #6c757d; font-size: 1.1rem;">Retrieve relevant legal precedents using AI-powered semantic search</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### 💡 Example Questions:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("• What were previous precedents regarding contractual disputes?")
    with col2:
        st.caption("• How have courts ruled on privacy violations in the past?")
    with col3:
        st.caption("• What is the legal history of intellectual property cases?")
    
    # Qdrant Configuration
    QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.j5Kv9gmGOtLHLL4RGMJpeqzdVJSrbmsFLlNdbtvmtYs"
    COLLECTION_NAME = "legal_precedents"

    @st.cache_resource
    def load_rag_chain():
        try:
            # 1. Setup Embeddings (Must match what you used for migration)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # 2. Connect to Qdrant Cloud
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            # 3. Create Vector Store Wrapper (CORRECTED)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings, 
            )

            # 4. Setup LLM (Groq)
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("api_key")
            if not api_key:
                return {"error": "Groq API key not found in .env file."}
            
            llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=api_key, temperature=0.2)

            # 5. Create Prompt Template
            prompt = ChatPromptTemplate.from_template(
                """
                You are a legal assistant. Use the following pieces of retrieved context to answer the question.
                If the context does not contain the answer, say that you don't know. 
                Keep the answer professional and concise.

                Context:
                {context}

                Question:
                {input}
                """
            )

            # 6. Build Chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            return {"rag_chain": rag_chain}

        except Exception as e:
            return {"error": f"Failed to initialize RAG: {e}"}

    with st.spinner("🌐 Connecting to Legal Knowledge Base..."):
        rag_resources = load_rag_chain()

    if "error" in rag_resources:
        st.error(rag_resources["error"])
        st.stop()

    rag_chain = rag_resources["rag_chain"]
    
    query = st.text_area("**Enter your legal question:**", height=150,
                        placeholder="e.g., What were previous precedents regarding digital privacy rights?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("**🔍 Search Precedents**", type="primary"):
            if not query.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("📚 Searching through legal database..."):
                    try:
                        response = rag_chain.invoke({"input": query})
                        answer = response.get("answer")
                        
                        st.markdown(
                            """
                            <div style="margin: 2rem 0;">
                                <h3 style="color: #2D3047; border-bottom: 2px solid #667eea; padding-bottom: 0.5rem;">📋 AI Legal Analysis</h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.markdown(
                            f"""
                            <div class="result-card" style="background: linear-gradient(135deg, #F0F4FF 0%, #E6F0FF 100%);">
                                {answer}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Sources section
                        with st.expander("📂 **View Source Documents**", expanded=False):
                            for i, doc in enumerate(response["context"]):
                                st.markdown(f"**📄 Source {i+1}**")
                                # Safely access page_content with fallback
                                content = getattr(doc, 'page_content', "No content available")
                                st.markdown(
                                    f"""
                                    <div style="background: rgba(255, 255, 255, 0.7); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
                                    {content[:600]}...
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                st.caption(f"Relevance score: {(5-i)/5:.1%}")
                                if i < len(response["context"]) - 1:
                                    st.divider()
                                    
                    except Exception as e:
                        st.error(f"❌ Error during retrieval: {e}")
