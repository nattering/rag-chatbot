import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
import tempfile, os

st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    background-color: #080b14 !important;
    color: #e8eaf6;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2440;
    padding-top: 2rem;
}

[data-testid="stSidebar"] * { color: #c7cde8 !important; }

.sidebar-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    padding: 0 1rem;
}

.sidebar-tagline {
    font-size: 0.78rem;
    color: #4b5280 !important;
    padding: 0 1rem;
    margin-bottom: 1.5rem;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid #1e2440;
    margin: 1.2rem 0;
}

.doc-stats {
    background: linear-gradient(135deg, #12182e, #0f1526);
    border: 1px solid #2a3060;
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 1rem;
}

.doc-stats .stat-label {
    font-size: 0.7rem;
    color: #4b5280 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.doc-stats .stat-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #a78bfa !important;
    margin-bottom: 0.5rem;
}

/* ── Inputs ── */
input[type="password"], input[type="text"], textarea {
    background-color: #0f1526 !important;
    border: 1px solid #2a3060 !important;
    border-radius: 10px !important;
    color: #e8eaf6 !important;
    font-family: 'Inter', sans-serif !important;
}

input:focus, textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
}

[data-testid="stFileUploader"] {
    background: #0f1526;
    border: 1px dashed #2a3060;
    border-radius: 12px;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #7c3aed;
}

/* ── Main area ── */
.main-header {
    text-align: center;
    padding: 3rem 0 1rem 0;
}

.main-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.main-subtitle {
    color: #4b5280;
    font-size: 1rem;
    font-weight: 400;
}

/* ── Chat area ── */
.chat-wrapper {
    max-width: 780px;
    margin: 0 auto;
    padding: 1rem 0;
}

.glow-container {
    position: relative;
}

.glow-container::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 60%;
    height: 40%;
    background: radial-gradient(ellipse, rgba(124, 58, 237, 0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stChatMessage"] {
    background: #0d1120 !important;
    border: 1px solid #1e2440 !important;
    border-radius: 16px !important;
    margin-bottom: 0.75rem;
    padding: 0.25rem 0.5rem;
    position: relative;
    z-index: 1;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-color: #2a3060 !important;
    background: #0f1526 !important;
}

[data-testid="stChatInput"] {
    max-width: 780px;
    margin: 0 auto;
}

[data-testid="stChatInput"] textarea {
    background: #0d1120 !important;
    border: 1px solid #2a3060 !important;
    border-radius: 14px !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1rem !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.2) !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #2a3060;
}

.empty-state .icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    display: block;
}

.empty-state h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    color: #3b4270;
    margin-bottom: 0.5rem;
}

.empty-state p {
    font-size: 0.85rem;
    color: #2a3060;
}

/* ── Source expander ── */
.stExpander {
    background: #0a0e1a !important;
    border: 1px solid #1e2440 !important;
    border-radius: 10px !important;
    margin-top: 0.3rem;
}

/* ── Ready badge ── */
.ready-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-size: 0.78rem;
    color: #a78bfa;
    margin: 0.5rem 1rem;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #7c3aed !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080b14; }
::-webkit-scrollbar-thumb { background: #2a3060; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI-powered document chat</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("**API Key**")
    api_key = st.text_input("", type="password", placeholder="sk-...", label_visibility="collapsed")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("**Document**")
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if "doc_stats" in st.session_state:
        stats = st.session_state.doc_stats
        st.markdown(f"""
        <div class="doc-stats">
            <div class="stat-label">File</div>
            <div class="stat-value">{stats['name']}</div>
            <div class="stat-label">Pages</div>
            <div class="stat-value">{stats['pages']}</div>
            <div class="stat-label">Chunks indexed</div>
            <div class="stat-value">{stats['chunks']}</div>
        </div>
        <div class="ready-badge">✦ Ready to answer questions</div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.pop("chain", None)
        st.session_state.pop("doc_stats", None)
        st.rerun()

# ── Main ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">Ask your document anything.</div>
    <div class="main-subtitle">Upload a PDF, get instant answers powered by GPT-4o.</div>
</div>
""", unsafe_allow_html=True)

# ── Build chain when PDF + key are ready ──────────────────────
if uploaded_file and api_key:
    file_id = uploaded_file.name + str(uploaded_file.size)

    if st.session_state.get("loaded_file") != file_id:
        os.environ["OPENAI_API_KEY"] = api_key

        with st.spinner("Indexing your document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name

            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            llm = ChatOpenAI(model="gpt-4o", temperature=0)

            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm,
                vectorstore.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )
            st.session_state.history = []
            st.session_state.loaded_file = file_id
            st.session_state.doc_stats = {
                "name": uploaded_file.name,
                "pages": len(pages),
                "chunks": len(chunks)
            }
            st.rerun()

# ── Chat ───────────────────────────────────────────────────────
st.markdown('<div class="chat-wrapper glow-container">', unsafe_allow_html=True)

if "chain" not in st.session_state:
    st.markdown("""
    <div class="empty-state">
        <span class="icon">📄</span>
        <h3>No document loaded yet</h3>
        <p>Add your OpenAI API key and upload a PDF<br>in the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    question = st.chat_input("Ask something about your document...")

    if question:
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            result = st.session_state.chain({
                "question": question,
                "chat_history": st.session_state.history
            })
            answer = result["answer"]
            sources = result.get("source_documents", [])

        st.chat_message("assistant").write(answer)

        if sources:
            pages_cited = sorted(set(
                doc.metadata.get("page", 0) + 1
                for doc in sources
            ))
            with st.expander(f"📎 Sources — pages {', '.join(map(str, pages_cited))}"):
                for i, doc in enumerate(sources):
                    page = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Page {page}**")
                    st.caption(doc.page_content.strip())
                    if i < len(sources) - 1:
                        st.markdown("---")

        st.session_state.history.append(("user", question))
        st.session_state.history.append(("assistant", answer))

st.markdown('</div>', unsafe_allow_html=True)

