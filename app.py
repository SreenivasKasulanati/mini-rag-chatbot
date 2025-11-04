import html
import sys, subprocess
import os
import streamlit as st
from rag import answer, clear_cache

st.set_page_config(
    page_title="RAG Chatbot - AI-Powered Document Assistant", 
    page_icon="💬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("")
    
    k = st.slider("📊 Top-K Chunks", 1, 10, 4, help="Number of document chunks to retrieve")
    
    provider = st.selectbox(
        "🤖 LLM Provider", ["auto", "openai", "stub"],
        help="'auto' uses OpenAI if configured, else an offline stub/extractive answer."
    )
    
    st.markdown("---")
    st.markdown("### 🔧 Data Management")
    
    if st.button("🔄 Rebuild Index", help="Run ingest.py to rebuild from data/", use_container_width=True):
        with st.spinner("Rebuilding index…"):
            # Run ingest.py from the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.run(
                [sys.executable, "ingest.py"], 
                capture_output=True, 
                text=True,
                cwd=script_dir  # Ensure we run from the correct directory
            )
        
        if result.returncode == 0:
            # Clear the cached index so it gets reloaded
            clear_cache()
            st.success("✅ Vector store rebuilt successfully.")
            st.info("🔄 Cache cleared - new index will be loaded on next query.")
            if result.stdout:
                st.code(result.stdout, language="bash")
        else:
            st.error("❌ Failed to rebuild index. See logs:")
            st.code(result.stderr or result.stdout or "(no output)", language="bash")
    
    st.info("💡 **Tip:** After adding files to `data/` folder, click Rebuild Index to update the knowledge base.")
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This chatbot uses **RAG (Retrieval-Augmented Generation)** to answer questions based on your documents.
    
    - Upload docs to `data/` folder
    - Supports PDF, TXT, MD files
    - Powered by OpenAI & sentence transformers
    """)


# ---------------- State ----------------
if "chat" not in st.session_state:
    # list of tuples: ("user", str) or ("assistant", dict)
    st.session_state.chat = []

# ---------------- Header ----------------
# Custom CSS for professional styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Remove default Streamlit padding and backgrounds */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Ensure no white backgrounds in main container */
    .main .block-container {
        background: transparent;
    }
    
    /* Remove any default element backgrounds */
    .element-container {
        background: transparent;
    }
    
    /* Custom compact header with creative elements */
    .main-header {
        position: relative;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
        overflow: hidden;
    }
    
    /* Animated gradient background */
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .main-header h1 {
        position: relative;
        z-index: 1;
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        position: relative;
        z-index: 1;
        color: rgba(255, 255, 255, 0.95);
        font-size: 0.95rem;
        margin: 0.4rem 0 0 0;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.6rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Improve chat message styling */
    .stChatMessage {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Hide Streamlit info boxes */
    .stAlert {
        display: none;
    }
</style>

<div class="main-header">
    <h1>✨ Mini RAG Chatbot</h1>
    <p>Intelligent document-based assistance powered by AI</p>
    <div class="status-badge">🟢 Online & Ready</div>
</div>
""", unsafe_allow_html=True)

# Note: Info box removed for cleaner interface


# ---------------- Chat History (scrollable) ----------------
chat_wrapper_start = """
<div style="
  max-height: 560px;
  overflow-y: auto;
  padding: 20px 0;
  border: none;
  background: transparent;
">
"""
st.markdown(chat_wrapper_start, unsafe_allow_html=True)

if not st.session_state.chat:
    # Creative empty state placeholder
    st.markdown(
        '''
        <div style="
            text-align: center;
            padding: 3rem 2rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.8;">💬</div>
            <div style="
                font-size: 1.1rem;
                font-weight: 600;
                color: #a5b4fc;
                margin-bottom: 0.5rem;
            ">Ready to assist you</div>
            <div style="
                font-size: 0.9rem;
                color: #9ca3af;
                font-weight: 400;
            ">Ask me anything about your documents</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
else:
    # Render oldest -> newest so newest sits closest to input
    for role, content in st.session_state.chat:
        if role == "user":
            text = html.escape(str(content))
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; align-items:flex-start; gap:8px; margin:4px 2px;">
                <div style="
                    max-width: 70%;
                    border-radius: 14px;
                    padding: 12px 16px;
                    line-height: 1.5;
                    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                    color:#fff; 
                    font-size: 15px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    box-shadow: 0 3px 12px rgba(99, 102, 241, 0.35);
                ">{text}</div>
                <div title='You' style="
                    width:36px; height:36px; border-radius:50%;
                    display:grid; place-items:center;
                    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                    color:#fff;
                    font-size:16px;
                    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
                ">👤</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            ans = html.escape(str(content.get("answer","")))
            srcs = content.get("sources", [])
            if srcs:
               display = ["OpenAI (general)" if s == "openai-fallback" else s for s in srcs]
               src_html = " · ".join(html.escape(s) for s in display)
               src_html = f'<div style="margin-top:4px; font-size:11px; color:#777;">Sources: {src_html}</div>'
            else:
                src_html = ""
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-start; align-items:flex-start; gap:8px; margin:4px 2px;">
                <div title='Assistant' style="
                    width:36px; height:36px; border-radius:50%;
                    display:grid; place-items:center;
                    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%);
                    color:#fff;
                    font-size:16px;
                    box-shadow: 0 2px 8px rgba(240, 147, 251, 0.25);
                ">🤖</div>
                <div style="
                    max-width: 70%;
                    border-radius: 14px;
                    padding: 12px 16px;
                    line-height: 1.6;
                    background:#ffffff;
                    color:#1a1a1a;
                    font-size: 15px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    border:1px solid #e5e7eb;
                    box-shadow: 0 2px 8px rgba(0,0,0,.06);
                ">{ans}{src_html}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)  # end wrapper

# ---------------- Input (auto-clears) ----------------
user_msg = st.chat_input("Type your question and press Enter…")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    try:
        with st.spinner("Thinking…"):
            # Build conversation history from last 10 messages (for context)
            conversation_history = []
            history_limit = 10
            recent_chat = st.session_state.chat[-history_limit-1:-1]  # Exclude current message
            
            for role, content in recent_chat:
                if role == "user":
                    conversation_history.append({"role": "user", "content": str(content)})
                elif role == "assistant":
                    # Extract just the answer text from the dict
                    answer_text = content.get("answer", "") if isinstance(content, dict) else str(content)
                    conversation_history.append({"role": "assistant", "content": answer_text})
            
            # Call answer with conversation history
            resp = answer(user_msg, k=k, provider=provider, conversation_history=conversation_history)
        st.session_state.chat.append(("assistant", resp))
    except Exception as e:
        st.error(f"Error calling answer(): {e}")
    # 🔧 Important: refresh immediately so the new messages render now
    st.rerun()


