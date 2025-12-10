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

# --- MODERN LEGAL UI THEME ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Playfair+Display:wght@600;700&display=swap');

    /* Global Settings */
    .stApp {
        background-color: #F8FAFC; /* Slate 50 */
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F172A; /* Slate 900 */
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] h1, p, label {
        color: #E2E8F0 !important;
    }

    /* Typography */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #1E293B; /* Slate 800 */
        font-weight: 700;
    }
    
    /* Custom Buttons */
    div.stButton > button {
        background: linear-gradient(to right, #2563EB, #1D4ED8); /* Blue 600-700 */
        color: white !important;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        transform: translateY(-1px);
    }

    /* Input Areas */
    .stTextArea textarea {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        background-color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        color: #334155;
    }
    .stTextArea textarea:focus {
        border-color: #2563EB;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1);
    }

    /* Result Cards */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Feature Cards (Home) */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #F1F5F9;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
        height: 100%;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        border-color: #BFDBFE;
    }
    .icon-box {
        width: 48px;
        height: 48px;
        background-color: #EFF6FF;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        margin-bottom: 1rem;
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

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/law.png", width=70) # Updated Icon
    st.markdown("### Legal AI Toolkit")
    st.caption("v2.0 | Enterprise Edition")
    st.markdown("---")
    mode = st.radio(
        "Module Selection",
        ("Home", "Case Classification", "Case Prioritization", "Legal Precedent Search"),
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.info("💡 **Tip:** Be specific in your queries for better citation accuracy.")

# Home Screen
if mode == "Home":
    st.title("Intelligent Legal Assistant")
    st.markdown("<p style='font-size: 1.1rem; color: #64748B;'>Streamline your legal workflow with AI-powered analysis and research tools.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="icon-box">📂</div>
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Automated Classification</h3>
            <p style="font-size: 0.9rem; color: #64748B;">
                Instantly categorize legal documents into Civil, Criminal, or Constitutional domains using advanced NLP.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="icon-box">⚡</div>
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Smart Prioritization</h3>
            <p style="font-size: 0.9rem; color: #64748B;">
                AI-driven urgency assessment (High, Medium, Low) to optimize case docket management.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="icon-box">⚖️</div>
            <h3 style="font-size: 1.1rem; margin-bottom: 0.5rem;">Precedent Discovery</h3>
            <p style="font-size: 0.9rem; color: #64748B;">
                RAG-powered search engine retrieving relevant case law from your secure Qdrant knowledge base.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Case Classification
if mode == "Case Classification":
    st.title("📂 Case Classification")
    st.markdown("Input the case summary below to determine its legal jurisdiction.")
    
    pipeline_path = "Case Classification/voting_pipeline.pkl"
    label_path = "Case Classification/label_encoder.pkl"

    with st.spinner("Loading AI Models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Case Brief", height=200, placeholder="Example: The plaintiff filed a suit for specific performance of contract regarding property #123...")

    if st.button("Classify Case"):
        if not text_input.strip():
            st.warning("⚠️ Please provide a case description.")
        elif pipeline is None:
            st.error("❌ Model files missing.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            
            st.markdown("---")
            st.markdown(f"""
            <div class="result-card" style="border-left-color: #2563EB;">
                <h4 style="margin:0; color: #64748B; font-size: 0.9rem;">PREDICTED CATEGORY</h4>
                <h2 style="margin:5px 0 0 0; color: #1E293B;">{pred_label}</h2>
            </div>
            """, unsafe_allow_html=True)

# Case Prioritization
if mode == "Case Prioritization":
    st.title("⚡ Case Prioritization")
    st.markdown("Assess case urgency to optimize resource allocation.")
    
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"

    with st.spinner("Loading AI Models..."):
        pipeline = load_pickle(pipeline_path)
        label_encoder = load_pickle(label_path)

    text_input = st.text_area("Case Brief", height=200, placeholder="Enter case details to assess urgency...")

    if st.button("Assess Priority"):
        if not text_input.strip():
            st.warning("⚠️ Please provide a case description.")
        elif pipeline is None:
            st.error("❌ Model files missing.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            
            color_map = {
                "High": "#DC2626",   # Red
                "Medium": "#D97706", # Amber
                "Low": "#059669"     # Emerald
            }
            color = color_map.get(pred_label, "#2563EB")
            
            st.markdown("---")
            st.markdown(f"""
            <div class="result-card" style="border-left-color: {color};">
                <h4 style="margin:0; color: #64748B; font-size: 0.9rem;">RECOMMENDED PRIORITY</h4>
                <h2 style="margin:5px 0 0 0; color: {color};">{pred_label} Priority</h2>
            </div>
            """, unsafe_allow_html=True)

# Legal Precedent Search (RAG)
if mode == "Legal Precedent Search":
    st.title("🔍 Precedent Research")
    st.markdown("Retrieve relevant case law and generate legal memos using AI.")

    # Configuration
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
            
            # Using Llama 3 for better reasoning
            llm = ChatGroq(model_name="llama3-8b-8192", api_key=api_key, temperature=0.1)

            # --- IMPROVED PROMPT TEMPLATE ---
            # Forces the LLM to act like a lawyer and structure the output
            prompt = ChatPromptTemplate.from_template(
                """
                You are an expert legal research assistant. Your task is to answer the user's legal question based EXCLUSIVELY on the provided context.
                
                Guidelines:
                1. Structure your answer as a formal Legal Memorandum.
                2. Start with a direct answer (Holding/Conclusion).
                3. Cite specific precedents or facts from the context to support your answer.
                4. If the context does not contain the answer, state "I cannot find relevant precedents in the database."
                5. Do not hallucinate or make up laws.

                Context:
                {context}

                Question:
                {input}
                
                Output Format:
                **Summary of Findings:** [Direct Answer]
                
                **Relevant Precedents & Analysis:**
                * [Point 1 based on context]
                * [Point 2 based on context]
                
                **Conclusion:** [Final thought]
                """
            )

            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            return {"rag_chain": rag_chain}

        except Exception as e:
            return {"error": str(e)}

    with st.spinner("Connecting to Legal Knowledge Base..."):
        rag_resources = load_rag_chain()

    if "error" in rag_resources:
        st.error(f"❌ Connection Error: {rag_resources['error']}")
        st.stop()

    rag_chain = rag_resources["rag_chain"]
    
    query = st.text_area("Legal Query", height=120, placeholder="E.g., What is the precedent for granting bail in non-bailable offenses regarding medical grounds?")
    
    col_submit, col_clear = st.columns([1, 5])
    with col_submit:
        search = st.button("Search Database")

    if search:
        if not query.strip():
            st.warning("⚠️ Please enter a query.")
        else:
            with st.spinner("Analyzing precedents and drafting memo..."):
                try:
                    response = rag_chain.invoke({"input": query})
                    answer = response.get("answer")
                    
                    st.markdown("### 📝 Legal Memorandum")
                    # Display answer in a nice paper-like container
                    st.markdown(f"""
                    <div style="background-color: white; padding: 30px; border-radius: 5px; border: 1px solid #E2E8F0; font-family: 'Times New Roman', serif; color: #000;">
                        {answer.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>### 📚 Source Citations", unsafe_allow_html=True)
                    for i, doc in enumerate(response["context"]):
                        content = getattr(doc, 'page_content', "N/A")
                        # Clean up content preview
                        preview = content[:300].replace("\n", " ") + "..."
                        
                        with st.expander(f"Citation {i+1} (Relevance Score: High)"):
                            st.markdown(f"**Extract:** *{preview}*")
                            st.info("Full Context Available in Database")
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
