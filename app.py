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

# --- CORRECTED IMPORTS FOR QDRANT & LANGCHAIN ---
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- NEW IMPORTS FOR RERANKING ---
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Page config
st.set_page_config(page_title="Legal Case Management & Precedent Search", layout="wide")

# Custom UI Styling
st.markdown("""
<style>
.stApp { background-color: #F8F9FA; }
section[data-testid="stSidebar"] { background-color: #2C2C2C; color: white; }
section[data-testid="stSidebar"] * { color: white !important; }

div.stButton > button {
    background-color: #FFFFFF;
    border-radius: 10px;
    border: 2px solid #C9A227;
    color: #1A2B4C;
    font-weight: 700;
}

div.stButton > button[kind="primary"] {
    background-color: #DC2626 !important;
    color: #FFFFFF !important;
    border: none !important;
}

h1, h2, h3, h4 { color: #1A2B4C; font-family: 'Georgia', serif; }
.stMarkdown, p, label { color: #000000 !important; font-family: 'Georgia', serif; }
.stChatMessage {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

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

# Sidebar
st.sidebar.title("⚖️ Legal AI Toolkit")
mode = st.sidebar.radio(
    "Choose a tool:",
    ("Home", "Case Classification", "Case Prioritization", "Legal Assistant (Chat)")
)

# Sidebar bottom
st.sidebar.markdown("<br>" * 10, unsafe_allow_html=True)
if st.sidebar.button("🗑️ Clear Chat History", type="primary", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

# Home
if mode == "Home":
    st.title("AI Powered Legal Case Management & Precedent Search")
    st.markdown("""
        ### What this project does
        - **Case Classification**: Automatically classify court cases by category (Civil, Criminal, or Constitutional)
        - **Case Prioritization**: Predict the urgency level of cases (High, Medium, Low)
        - **Legal Assistant (Chat)**: Chat with your legal database using memory and re-ranking.
    """)
    st.info("Select a tool from the sidebar to get started.")
    st.stop()

# Case Classification
if mode == "Case Classification":
    st.title("⚖️ Case Classification")
    pipeline_path = "Case Classification/voting_pipeline.pkl"
    label_path = "Case Classification/label_encoder.pkl"
    pipeline = load_pickle(pipeline_path)
    label_encoder = load_pickle(label_path)
    text_input = st.text_area("Paste case text here:", height=300)
    if st.button("Predict Category"):
        if not text_input.strip():
            st.warning("Please enter some case text.")
        elif pipeline is None:
            st.error("Pipeline not loaded.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            st.success(f"Predicted Case Category: **{pred_label}**")

# Case Prioritization
if mode == "Case Prioritization":
    st.title("⚖️ Case Prioritization")
    pipeline_path = "Case Prioritization/stacking_pipeline.pkl"
    label_path = "Case Prioritization/label_encoder.pkl"
    pipeline = load_pickle(pipeline_path)
    label_encoder = load_pickle(label_path)
    text_input = st.text_area("Paste case text here:", height=300)
    if st.button("Predict Priority"):
        if not text_input.strip():
            st.warning("Please enter some case text.")
        elif pipeline is None:
            st.error("Pipeline not loaded.")
        else:
            cleaned = preprocess_text(text_input)
            pred_enc = pipeline.predict([cleaned])
            pred_label = label_encoder.inverse_transform(pred_enc)[0] if label_encoder else str(pred_enc[0])
            st.success(f"Predicted Case Priority: **{pred_label}**")

# Legal Assistant
if mode == "Legal Assistant (Chat)":
    st.title("💬 Legal Research Assistant")
    st.markdown("Ask questions about precedents. The AI remembers context from this conversation.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Qdrant and Groq keys from Streamlit secrets
    QDRANT_URL = "https://2191fd84-3737-4604-ac35-435135b72cf3.us-east4-0.gcp.cloud.qdrant.io"
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    COLLECTION_NAME = "legal_precedents"
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    @st.cache_resource
    def load_rag_chain():
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

            # Reranking retriever
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
            model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            compressor = CrossEncoderReranker(model=model, top_n=5)
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

            llm = ChatGroq(model_name="openai/gpt-oss-20b", api_key=GROQ_API_KEY, temperature=0.1)

            # Memory Step 1: Contextualize Question
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question which might reference context in the chat history, "
                "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, contextualize_q_prompt)

            # Memory Step 2: Answer Question
            qa_system_prompt = """
            You are a Senior Legal Research Assistant. Analyze the provided legal documents and answer the user's question with high precision.

            STRICT INSTRUCTIONS:
            1. Answer ONLY using provided Context.
            2. Structure: Executive Summary, Relevant Precedents, Conclusion.
            3. If answer not in context, say so.

            Context:
            {context}
            """
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            return {"rag_chain": rag_chain}

        except Exception as e:
            return {"error": f"Failed to initialize RAG: {e}"}

    with st.spinner("Connecting to Legal Knowledge Base..."):
        rag_resources = load_rag_chain()

    if "error" in rag_resources:
        st.error(rag_resources["error"])
        st.stop()

    rag_chain = rag_resources["rag_chain"]

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Handle user input
    user_query = st.chat_input("Ask a follow-up question...")
    if user_query:
        st.chat_message("user").write(user_query)
        with st.spinner("Researching..."):
            try:
                response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
                answer = response['answer']
                with st.chat_message("assistant"):
                    st.write(answer)
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(response.get("context", [])):
                            st.markdown(f"**Source {i+1}:**")
                            content = getattr(doc, 'page_content', "No content available")
                            st.caption(content[:300] + "...")
                            st.divider()
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                st.error(f"Error: {e}")


