# ingest.py
# pip install pymupdf tiktoken supabase google-generativeai tqdm python-dotenv

import os, re
import fitz  # PyMuPDF
import tiktoken
from supabase import create_client, Client
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

# ---------------- Load environment ----------------
load_dotenv(find_dotenv(usecwd=True))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# ---------------- Config ----------------
PDF_PATH = "human-nutrition-text.pdf"
DOC_ID = "nutrition-v1"   # keep stable to avoid duplicates

EMBED_MODEL = "models/text-embedding-004"  # Gemini embeddings
BATCH_EMBED = 50
BATCH_INSERT = 200

# Chunking params
SENTS_PER_CHUNK = 20
SENT_OVERLAP = 2
MAX_TOKENS = 1300
MIN_TOKENS = 50

enc = tiktoken.get_encoding("cl100k_base")

# ---------------- Setup Gemini ----------------
genai.configure(api_key=GEMINI_API_KEY)

# ---------------- Utility functions ----------------
def clean_text(text: str) -> str:
    text = text.replace("\r", " ")
    text = re.sub(r"-\s*\n\s*", "", text)  # fix hyphenation
    text = re.sub(r"\s+\n", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def split_sentences(text: str):
    return re.split(r'(?<=[.!?])\s+', text)

def chunk_by_sentences(text: str):
    sents = split_sentences(text)
    i = 0
    step = max(1, SENTS_PER_CHUNK - SENT_OVERLAP)

    while i < len(sents):
        chunk_sents = sents[i:i + SENTS_PER_CHUNK]
        chunk = " ".join(chunk_sents)
        token_ids = enc.encode(chunk)

        while len(token_ids) > MAX_TOKENS and len(chunk_sents) > 1:
            chunk_sents = chunk_sents[:-1]
            chunk = " ".join(chunk_sents)
            token_ids = enc.encode(chunk)

        if len(token_ids) >= MIN_TOKENS:
            yield chunk

        i += step

def pdf_pages(path: str):
    doc = fitz.open(path)
    try:
        for i in range(len(doc)):
            yield i + 1, clean_text(doc[i].get_text("text") or "")
    finally:
        doc.close()

# ---------------- Main pipeline ----------------
def main():
    sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    # Clean existing document chunks
    sb.table("chunks").delete().eq("doc_id", DOC_ID).execute()

    print("📄 Reading PDF...")
    pages = list(pdf_pages(PDF_PATH))

    inputs, metas = [], []

    print("✂️ Chunking text...")
    for page_no, text in pages:
        if not text:
            continue
        for chunk in chunk_by_sentences(text):
            inputs.append(chunk)
            metas.append({
                "page": page_no,
                "source": PDF_PATH
            })

    print(f"✅ Created {len(inputs)} chunks")

    # -------- Generate Gemini embeddings --------
    embeddings = []
    print("🧠 Generating Gemini embeddings...")
    for i in tqdm(range(0, len(inputs), BATCH_EMBED)):
        batch = inputs[i:i + BATCH_EMBED]
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=batch
        )
        embeddings.extend(result["embedding"])

    # -------- Prepare DB rows --------
    rows = []
    for idx, (text, emb, meta) in enumerate(zip(inputs, embeddings, metas)):
        rows.append({
            "doc_id": DOC_ID,
            "chunk_index": idx,
            "content": text,
            "metadata": meta,
            "embedding": emb
        })

    # -------- Upload to Supabase --------
    print("⬆️ Uploading to Supabase...")
    for j in tqdm(range(0, len(rows), BATCH_INSERT)):
        sb.table("chunks").insert(rows[j:j + BATCH_INSERT]).execute()

    print(f"🎉 Ingestion complete for doc_id={DOC_ID}")

if __name__ == "__main__":
    main()
