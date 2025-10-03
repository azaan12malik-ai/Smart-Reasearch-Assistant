import ddgs
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="✨ Smart Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================
# CUSTOM CSS – Gradient Bubbles + Animations
# ======================
st.markdown(
    """
    <style>
    /* Scrollable container */
    .chat-container {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #f0f4f8 0%, #e8f0fe 100%);
        max-height: 70vh;
        overflow-y: auto;
        margin-bottom: 20px;
        box-shadow: 0 0 12px rgba(0,0,0,0.1);
    }
    /* Floating input bar */
    .stChatInputContainer {
        position: sticky !important;
        bottom: 0;
        background: #ffffffaa;
        backdrop-filter: blur(6px);
        border-radius: 20px;
        padding: 8px;
        margin-top: 10px;
    }
    /* User bubble */
    .user-bubble {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: black;
        padding: 14px 20px;
        border-radius: 20px 20px 0 20px;
        margin: 8px 0;
        max-width: 80%;
        word-wrap: break-word;
        float: right;
        clear: both;
        font-size: 16px;
        animation: fadeIn 0.4s ease;
    }
    /* Assistant bubble */
    .assistant-bubble {
        background: linear-gradient(135deg, #fddb92 0%, #d1fdff 100%);
        color: #333;
        padding: 14px 20px;
        border-radius: 20px 20px 20px 0;
        margin: 8px 0;
        max-width: 80%;
        word-wrap: break-word;
        float: left;
        clear: both;
        font-size: 16px;
        animation: fadeIn 0.4s ease;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(8px);}
        to {opacity: 1; transform: translateY(0);}
    }
    /* Typing dots animation */
    .typing {
        display: inline-block;
        padding-left: 10px;
    }
    .typing span {
        display: inline-block;
        width: 6px;
        height: 6px;
        background: #999;
        border-radius: 50%;
        margin: 0 2px;
        animation: blink 1.4s infinite;
    }
    .typing span:nth-child(2) {animation-delay: 0.2s;}
    .typing span:nth-child(3) {animation-delay: 0.4s;}
    @keyframes blink {
        0%, 80%, 100% {opacity: 0;}
        40% {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================
# SIDEBAR
# ======================
st.sidebar.markdown("## ⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Enter your **Groq API Key**:", type="password")
temperature = st.sidebar.slider("🔥 Creativity", 0.0, 1.0, 0.3, 0.1)
st.sidebar.markdown("---")
st.sidebar.info("💡 *Tip:* I can search **Web**, **Arxiv**, and **Wikipedia** for real-time answers!")

# ======================
# PAGE HEADER
# ======================
st.markdown(
    "<h1 style='text-align:center;'>🤖 Smart Research Assistant</h1>"
    "<p style='text-align:center;'>Ask anything! I’ll search the Web, Arxiv & Wikipedia for you ✨</p>",
    unsafe_allow_html=True
)

# ======================
# Initialize Tools
# ======================
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
search = DuckDuckGoSearchRun(name="🌐 Web Search")

# ======================
# Chat Session
# ======================
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi there! What topic should I explore for you today?"}
    ]

# ======================
# Display Chat
# ======================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ======================
# User Input
# ======================
if prompt := st.chat_input("💬 Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-bubble">🧑‍💻 {prompt}</div>', unsafe_allow_html=True)

    if not api_key:
        st.warning("⚠️ Please enter your **Groq API Key** in the sidebar.")
    else:
        try:
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="gemma2-9b-it",
                temperature=temperature
            )
            tools = [search, arxiv, wiki]
            agent = initialize_agent(
                tools, llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True
            )

            with st.spinner("🤔 Searching..."):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(f'<div class="assistant-bubble">🤖 {response}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")


