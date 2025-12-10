# app.py
import streamlit as st
import config
import utils
import rag_engine
from langchain_core.messages import HumanMessage, AIMessage

# Page Setup
st.set_page_config(page_title="Legal AI Toolkit", layout="wide", page_icon="⚖️")
utils.inject_custom_css()

# Sidebar
st.sidebar.title("⚖️ Legal AI Toolkit")
mode = st.sidebar.radio(
    "Choose a tool:",
    ("Home", "Case Classification", "Case Prioritization", "Legal Assistant (Chat)")
)

st.sidebar.markdown("<br>" * 10, unsafe_allow_html=True)
if st.sidebar.button("🗑️ Clear Chat History", type="primary", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

# --- MODULES ---

if mode == "Home":
    st.title("AI Powered Legal Case Management")
    st.markdown("### Streamlining Case Management & Research")
    st.info("Select a tool from the sidebar to get started.")

    # Use Columns with Custom CSS Cards for visibility
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📂 Classification</h3>
            <p>Automatically categorize court cases into Civil, Criminal, or Constitutional domains.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Prioritization</h3>
            <p>Predict the urgency level (High, Medium, Low) of incoming legal cases.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 Legal Assistant</h3>
            <p>Chat with your legal database using AI memory and precision re-ranking.</p>
        </div>
        """, unsafe_allow_html=True)

elif mode == "Case Classification":
    st.title("⚖️ Case Classification")
    with st.spinner("Loading models..."):
        pipeline = utils.load_pickle(config.VOTING_PIPELINE_PATH)
        encoder = utils.load_pickle(config.LABEL_ENCODER_PATH)
    
    text_input = st.text_area("Paste case text:", height=300)
    if st.button("Predict"):
        if pipeline and text_input.strip():
            clean_text = utils.preprocess_text(text_input)
            pred = pipeline.predict([clean_text])
            label = encoder.inverse_transform(pred)[0]
            st.success(f"Category: **{label}**")
        else:
            st.error("Model missing or empty text.")

elif mode == "Case Prioritization":
    st.title("⚡ Case Prioritization")
    with st.spinner("Loading models..."):
        pipeline = utils.load_pickle(config.PRIORITY_PIPELINE_PATH)
        encoder = utils.load_pickle(config.PRIORITY_ENCODER_PATH)
    
    text_input = st.text_area("Paste case text:", height=300)
    if st.button("Predict"):
        if pipeline and text_input.strip():
            clean_text = utils.preprocess_text(text_input)
            pred = pipeline.predict([clean_text])
            label = encoder.inverse_transform(pred)[0]
            st.success(f"Priority: **{label}**")

elif mode == "Legal Assistant (Chat)":
    st.title("💬 Legal Research Assistant")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load the Brain
    with st.spinner("Initializing AI..."):
        chain_data = rag_engine.build_rag_chain()
    
    if "error" in chain_data:
        st.error(chain_data["error"])
        st.stop()
        
    chain = chain_data["rag_chain"]

    # Chat Interface
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)

    if prompt := st.chat_input("Ask a legal question..."):
        st.chat_message("user").write(prompt)
        with st.spinner("Analyzing..."):
            response = chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            answer = response['answer']
            
            with st.chat_message("assistant"):
                st.write(answer)
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Source {i+1}**")
                        st.caption(doc.page_content[:300] + "...")
                        
            st.session_state.chat_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=answer)
            ])

