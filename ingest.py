import os
import json
import glob
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
VSTORE_DIR = BASE_DIR / "vectorstore"
VSTORE_DIR.mkdir(exist_ok=True)
INDEX_PATH = VSTORE_DIR / "faiss.index"
META_PATH = VSTORE_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
EMBED = SentenceTransformer(MODEL_NAME)

CHUNK_SIZE = 750  # characters
CHUNK_OVERLAP = 150

def read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_pdf(p: Path) -> str:
    reader = PdfReader(str(p))
    texts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(texts)

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap
        if size <= overlap:
            break
    return chunks

def load_faq(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, qa in enumerate(data):
        q = qa.get("question", "").strip()
        a = qa.get("answer", "").strip()
        if not (q and a):
            continue
        rows.append({
            "text": f"Q: {q}\nA: {a}",
            "source": f"faq.json#{i}",
        })
    return rows

def load_docs(dirpath: Path):
    rows = []
    for p in glob.glob(str(dirpath / "**/*"), recursive=True):
        p = Path(p)
        if p.is_dir():
            continue
        text = ""
        if p.suffix.lower() in [".txt", ".md"]:
            text = read_txt(p)
        elif p.suffix.lower() == ".pdf":
            text = read_pdf(p)
        else:
            continue
        for j, ch in enumerate(chunk_text(text)):
            rows.append({
                "text": ch,
                "source": f"{p.name}#chunk{j}",
            })
    return rows

if __name__ == "__main__":
    print("[ingest] loading data…")
    entries = []
    entries += load_faq(DATA_DIR / "faq.json")
    entries += load_docs(DOCS_DIR)
    if not entries:
        raise SystemExit("No data found in data/. Add files then re-run.")

    print(f"[ingest] {len(entries)} chunks -> embedding…")
    texts = [e["text"] for e in entries]
    vecs = EMBED.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = vecs.shape[1]

    print(f"[ingest] building FAISS index (dim={dim})…")
    index = faiss.IndexFlatIP(dim)
    # normalize for inner product similarity ~ cosine
    faiss.normalize_L2(vecs)
    index.add(vecs)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest] wrote {INDEX_PATH} and {META_PATH}")