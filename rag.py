# rag.py
import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ---------- OpenAI (optional) ----------
try:
    from openai import OpenAI  # openai>=1.x
except Exception:
    OpenAI = None

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_openai = OpenAI() if (OpenAI and OPENAI_KEY) else None

# ---------- Vectorstore paths ----------
VECTOR_DIR = Path("vectorstore")
INDEX_PATH = VECTOR_DIR / "faiss.index"       # produced by ingest.py
META_PATH  = VECTOR_DIR / "metadata.json"     # produced by ingest.py
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # must match ingest.py

# ---------- Retrieval / gating thresholds ----------
DEFAULT_TOP_K = 4
# cosine in [-1,1]; set conservative floor so gibberish won't pass
COS_MIN = 0.30
# minimal lexical overlap (after stopword removal) to accept weak sims
OVERLAP_MIN = 1  # at least 1 shared keyword when cosine is weak

# ---------- Lazy singletons ----------
_INDEX: Optional[faiss.Index] = None
_METADATA: Optional[List[Dict[str, Any]]] = None
_EMBED: Optional[SentenceTransformer] = None


def clear_cache():
    """Clear cached vectorstore to force reload after rebuild."""
    global _INDEX, _METADATA
    _INDEX = None
    _METADATA = None


# ----------------- utils -----------------
_STOPWORDS = {
    "a","an","the","is","am","are","was","were","be","been","being",
    "and","or","of","to","in","on","for","from","by","with","as",
    "at","that","this","these","those","it","its","you","your","yours",
    "we","our","ours","they","their","theirs","i","me","my","mine"
}

def _tokenize(s: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z0-9$%]+", s.lower()) if w not in _STOPWORDS]

def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = u.astype("float32"); v = v.astype("float32")
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return -1.0
    return float(np.dot(u, v) / (nu * nv))


def _load_vectorstore():
    global _INDEX, _METADATA
    if _INDEX is None or _METADATA is None:
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "Vector store not found. Run `python ingest.py` to build it."
            )
        _INDEX = faiss.read_index(str(INDEX_PATH))
        _METADATA = json.loads(META_PATH.read_text(encoding="utf-8"))


def _load_embedder():
    global _EMBED
    if _EMBED is None:
        _EMBED = SentenceTransformer(EMBED_MODEL_NAME)


# ----------------- retrieval -----------------
def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: {text, source, score, cos}
      - score: raw FAISS score
      - cos:   true cosine similarity in [-1, 1] (computed from reconstructed vectors)
    """
    _load_vectorstore()
    _load_embedder()

    q_vec = _EMBED.encode([query])[0].astype("float32")[None, :]
    scores, ids = _INDEX.search(q_vec, k)

    out: List[Dict[str, Any]] = []
    for s, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        meta = _METADATA[int(idx)]
        # reconstruct vector for consistent cosine (works for flat/IVF indexes)
        try:
            v = _INDEX.reconstruct(int(idx))
            cos = _cosine(q_vec[0], v)
        except Exception:
            # fallback: try to interpret FAISS score as inner product (best-effort)
            cos = float(s)
        out.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", "unknown"),
            "score": float(s),
            "cos": float(cos),
        })
    return out


# ----------------- prompting -----------------
def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
    return (
        "You are a helpful assistant that answers strictly using the provided context.\n"
        "If the answer is not present, reply exactly: 'I don’t have this information from the documents provided.'\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"
    )


def _llm_openai_chat(messages, max_tokens=300, temperature=0.2) -> Optional[str]:
    if not _openai:
        return None
    try:
        resp = _openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI error:", e)
        return None


def _llm_openai_prompt(prompt: str, max_tokens=300) -> Optional[str]:
    return _llm_openai_chat(
        [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )


def _extractive_stub(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "I don’t have this information from the documents provided."
    parts = []
    for c in chunks[:2]:
        src = c.get("source", "?")
        txt = (c.get("text","") or "").strip()
        if len(txt) > 600:
            txt = txt[:600] + "…"
        parts.append(f"- {src}\n{txt}")
    return "Here’s what I found in your docs:\n\n" + "\n\n".join(parts) + "\n\n(Enable OpenAI for synthesized answers.)"



# ----------------- small talk detection -----------------
def _is_small_talk(query: str) -> bool:
    """Detect if query is casual conversation vs. information request."""
    q_lower = query.lower().strip()
    
    # Greetings and casual phrases
    casual_patterns = [
        r'^(hi|hello|hey|yo|sup|howdy|greetings)[\s!?.]*$',
        r'^(good\s+(morning|afternoon|evening|day|night))[\s!?.]*$',
        r'^(how\s+(are|r)\s+you|how\s+(are|r)\s+ya|how\'s\s+it\s+going)[\s!?.]*$',
        r'^(what\'?s\s+up|whats\s+up|wassup)[\s!?.]*$',
        r'^(thanks?|thank\s+you|thx|ty)[\s!?.]*$',
        r'^(bye|goodbye|see\s+ya|cya|take\s+care)[\s!?.]*$',
        r'^(nice\s+to\s+meet\s+you|pleasure\s+to\s+meet)[\s!?.]*$',
        r'^(how\s+can\s+you\s+help|what\s+can\s+you\s+do)[\s!?.]*$',
    ]
    
    for pattern in casual_patterns:
        if re.match(pattern, q_lower):
            return True
    
    # Very short, non-informational queries (1-2 words) without request indicators
    tokens = _tokenize(query)
    
    # Question words indicate information requests, not small talk
    question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose', 'whom'}
    
    # Action verbs also indicate requests
    action_verbs = {'tell', 'show', 'explain', 'describe', 'find', 'give', 'help', 'get', 'provide', 'list', 'compare'}
    
    # Only flag as small talk if it's VERY short (1-2 tokens) AND doesn't contain question words or action verbs
    if len(tokens) <= 2 and not any(t in question_words or t in action_verbs for t in tokens):
        return True
    
    return False

# ----------------- main answer -----------------
def answer(query: str, k: int = DEFAULT_TOP_K, provider: str = "auto", conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Retrieval → gating → (RAG or fallback).
    Fallback behavior:
      - If OpenAI key present and evidence is weak → small-talk/general answer
      - Else → 'I don’t have this information from the documents provided.'
    """
    if conversation_history is None:
        conversation_history = []
    
    # Check for small talk BEFORE hitting the database
    if _is_small_talk(query):
        if _openai:
            # Build messages with history for context
            messages = [{"role": "system", "content": "You are a friendly, concise chatbot assistant."}]
            messages.extend(conversation_history)  # Add conversation history
            messages.append({"role": "user", "content": query})
            
            txt = _llm_openai_chat(messages, max_tokens=200, temperature=0.4)
            if txt:
                return {"answer": txt, "sources": ["openai-fallback"], "chunks": []}
        # No OpenAI but detected small talk
        return {
            "answer": "Hello! I'm here to help answer questions based on the documents provided. How can I assist you?\n\n(Note: Set OPENAI_API_KEY environment variable for natural conversation responses.)",
            "sources": [],
            "chunks": []
        }
    
    # Retrieve from documents
    try:
        hits = retrieve(query, k=k)
    except FileNotFoundError as e:
        return {"answer": f"I can’t access the document index yet. {e}", "sources": [], "chunks": []}

    # If nothing came back, it's clearly out-of-scope
    if not hits:
        if _openai:
            # Build messages with history for context
            messages = [{"role": "system", "content": "You are a friendly, concise chatbot."}]
            messages.extend(conversation_history)  # Add conversation history
            messages.append({"role": "user", "content": query})
            
            txt = _llm_openai_chat(messages, max_tokens=200, temperature=0.4)
            if txt:
                return {"answer": txt, "sources": ["openai-fallback"], "chunks": []}
        return {"answer": "I don’t have this information from the documents provided.", "sources": [], "chunks": []}

    # ------- Relevance gate (fix for 'gibberish returns docs') -------
    top = hits[0]
    top_cos = float(top.get("cos", -1.0))

    q_tokens = set(_tokenize(query))
    # keyword overlap against the *best* chunk text
    c_tokens = set(_tokenize(top.get("text","")))
    overlap = len(q_tokens.intersection(c_tokens))

    has_evidence = (top_cos >= COS_MIN) or (overlap >= OVERLAP_MIN)

    if not has_evidence:
        # treat as out-of-scope → general chat if possible
        if _openai:
            # Build messages with history for context
            messages = [{"role": "system", "content": "You are a friendly, concise chatbot."}]
            messages.extend(conversation_history)  # Add conversation history
            messages.append({"role": "user", "content": query})
            
            txt = _llm_openai_chat(messages, max_tokens=200, temperature=0.4)
            if txt:
                return {"answer": txt, "sources": ["openai-fallback"], "chunks": []}
        return {"answer": "I don’t have this information from the documents provided.", "sources": [], "chunks": []}

    # ------- RAG generation (we have evidence) -------
    prompt = build_prompt(query, hits)
    use_openai = (provider == "openai") or (provider == "auto" and _openai is not None)

    if use_openai:
        text = _llm_openai_prompt(prompt, max_tokens=300)
        if not text:
            text = _extractive_stub(hits)  # safety net
    else:
        text = _extractive_stub(hits)      # offline mode

    # unique sources
    seen, sources = set(), []
    for c in hits:
        s = c.get("source","unknown")
        if s not in seen:
            seen.add(s); sources.append(s)

    return {"answer": text, "sources": sources, "chunks": hits}
