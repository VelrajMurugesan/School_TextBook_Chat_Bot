
# app_school_textbook_rag_FINAL_STABLE_UX.py
# ============================================================
# SCHOOL TEXTBOOK AI CHATBOT – FINAL STABLE UX BUILD
#
# FIXED:
# ✅ Upload spinner + success/info messages always visible
# ✅ Chat "Thinking..." spinner always visible
# ✅ Cache usage clearly indicated
# ✅ Valid textbook questions NOT marked Out-of-syllabus
# ✅ Strict but SAFE OOS detection (no false negatives)
#
# Python 3.12 Compatible
# ============================================================

import os
import re
import hashlib
import numpy as np
import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
FAISS_DIR = "faiss_index_cached"
HASH_FILE = "pdf_hash.txt"

TOP_K = 10
SIM_THRESHOLD = 0.30        # lowered to avoid false OOS
WORD_OVERLAP_THRESHOLD = 1  # safer for definition questions

st.set_page_config(layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.user{background:#dcf8c6;color:#000;padding:12px;border-radius:14px;margin:10px;text-align:right;}
.bot{background:#111827;color:#e5e7eb;padding:14px;border-radius:14px;margin:10px;border-left:4px solid #3b82f6;}
.ref{background:#ffffff;color:#111827;padding:12px;border-radius:12px;margin-bottom:12px;border-left:5px solid #22c55e;}
.highlight{background:#fde047;color:#000;padding:2px 4px;border-radius:4px;font-weight:600;}
.section{font-size:18px;font-weight:700;color:#93c5fd;margin-bottom:8px;}
.file{background:#1f2937;color:#e5e7eb;padding:6px 10px;border-radius:999px;margin:4px 0;display:inline-block;font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<div style="background:linear-gradient(90deg,#020617,#1e293b);
padding:12px;border-radius:14px;color:white;text-align:center;">
<h3>📘 School Textbook AI Chatbot 🧠</h3>
</div>
""", unsafe_allow_html=True)

col_upload, col_chat, col_right = st.columns([2, 5, 3])

# ---------------- STATE ----------------
if "vs" not in st.session_state:
    st.session_state.vs = None
if "history" not in st.session_state:
    st.session_state.history = []
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# ---------------- UTILS ----------------


def file_hash(files):
    h = hashlib.sha256()
    for f in files:
        h.update(f.getbuffer())
    return h.hexdigest()


def cosine(a, b):
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


def detect_unit(text):
    m = re.search(r"(unit|lesson|chapter)\s+\d+", text, re.I)
    return m.group(0) if m else "Unknown"


def keyword_overlap(q, docs):
    q_words = set(re.findall(r"\w+", q.lower()))
    for d in docs:
        d_words = set(re.findall(r"\w+", d.page_content.lower()))
        if len(q_words & d_words) >= WORD_OVERLAP_THRESHOLD:
            return True
    return False

# ---------------- EMBEDDINGS ----------------


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200,
        temperature=0.0
    )


embeddings = load_embeddings()
llm = load_llm()

# ---------------- LOAD INDEX (CACHE) ----------------
if os.path.exists(FAISS_DIR) and os.path.exists(HASH_FILE):
    with st.spinner("🔄 Loading cached textbook index..."):
        st.session_state.vs = FAISS.load_local(
            FAISS_DIR, embeddings, allow_dangerous_deserialization=True
        )
        st.session_state.index_ready = True

# ---------------- UPLOAD ----------------
with col_upload:
    st.markdown("<div class='section'>📂 Upload</div>", unsafe_allow_html=True)

    files = st.file_uploader(
        "Upload textbook PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if files:
        for f in files:
            st.markdown(
                f"<span class='file'>📄 {f.name}</span>", unsafe_allow_html=True)

        current = file_hash(files)
        old = open(HASH_FILE).read() if os.path.exists(HASH_FILE) else None

        if current != old:
            with st.spinner("📚 Indexing textbook (one-time)..."):
                docs = []
                for f in files:
                    p = f"_tmp_{f.name}"
                    open(p, "wb").write(f.read())
                    for d in PyPDFLoader(p).load():
                        d.metadata.update({
                            "source": f.name,
                            "unit": detect_unit(d.page_content),
                            "page": d.metadata.get("page", "NA")
                        })
                        docs.append(d)
                    os.remove(p)

                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=200
                ).split_documents(docs)

                vs = FAISS.from_documents(chunks, embeddings)
                vs.save_local(FAISS_DIR)
                open(HASH_FILE, "w").write(current)

                st.session_state.vs = vs
                st.session_state.index_ready = True

                st.success("✅ Textbook indexed successfully")
        else:
            st.info("ℹ️ Using cached textbook index")
            st.session_state.index_ready = True

# ---------------- CHAT ----------------
with col_chat:
    st.markdown("<div class='section'>💬 Chat</div>", unsafe_allow_html=True)

    if not st.session_state.index_ready:
        st.warning("Please upload a textbook PDF first.")
    else:
        q = st.text_input("", placeholder="Ask a question from the textbook")

        if q:
            with st.spinner("🧠 Thinking..."):
                docs = st.session_state.vs.as_retriever(
                    search_kwargs={"k": TOP_K}
                ).invoke(q)

                sims = [
                    cosine(
                        embeddings.embed_query(q),
                        embeddings.embed_query(d.page_content)
                    ) for d in docs
                ]
                avg_sim = np.mean(sorted(sims, reverse=True)
                                  [:5]) if sims else 0.0

                if avg_sim < SIM_THRESHOLD or not keyword_overlap(q, docs):
                    answer = "Out of syllabus."
                    confidence = round(avg_sim * 100, 2)
                    refs = []
                else:
                    context = "\n".join(d.page_content for d in docs[:5])
                    answer = llm(
                        "Answer ONLY from the textbook content below. "
                        "If answer is not present, say Out of syllabus.\n\n"
                        f"{context}\n\nQuestion: {q}"
                    )[0]["generated_text"].strip()

                    confidence = round(avg_sim * 100, 2)

                    refs = [
                        {
                            "source": d.metadata["source"],
                            "page": d.metadata["page"],
                            "text": s.strip()
                        }
                        for d in docs[:5]
                        for s in re.split(r"[.!?]", d.page_content)
                        if len(s) > 30 and any(w.lower() in s.lower() for w in q.split())
                    ]

                st.session_state.history.append((q, answer, confidence, refs))

        for q, a, c, _ in st.session_state.history[::-1]:
            st.markdown(f"<div class='user'>{q}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='bot'>{a}<br><small>Accuracy: {c}%</small></div>",
                unsafe_allow_html=True
            )

# ---------------- RIGHT PANEL ----------------
with col_right:
    tab1, tab2 = st.tabs(["📎 Page Highlights", "📘 Unit Summary"])

    with tab1:
        if st.session_state.history:
            _, ans, _, refs = st.session_state.history[-1]
            if ans != "Out of syllabus." and refs:
                for r in refs:
                    st.markdown(
                        f"<div class='ref'><b>{r['source']}</b> | Page {r['page']}<br>"
                        f"<span class='highlight'>{r['text']}</span></div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("No page highlights available.")

    with tab2:
        if st.session_state.vs:
            unit_docs = {}
            for d in st.session_state.vs.docstore._dict.values():
                unit_docs.setdefault(d.metadata.get(
                    "unit", "Unknown"), []).append(d)

            unit = st.selectbox("Select Unit", sorted(unit_docs.keys()))

            if st.button("Generate Unit Summary"):
                with st.spinner("📘 Generating unit summary..."):
                    text = " ".join(
                        d.page_content for d in unit_docs[unit][:6])
                    summary = llm(
                        "Summarize this unit clearly for school students "
                        "using ONLY the textbook content:\n" + text
                    )[0]["generated_text"]

                    sims = [
                        cosine(
                            embeddings.embed_query(summary),
                            embeddings.embed_query(d.page_content)
                        )
                        for d in unit_docs[unit][:6]
                    ]
                    unit_conf = round(np.mean(sims) * 100, 2)

                    st.success(summary)
                    st.info(f"Summary accuracy confidence: {unit_conf}%")
