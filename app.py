import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
from dotenv import load_dotenv

import google.generativeai as genai
from mistralai.client import MistralClient

# ---------------- Load env ----------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ---------------- Init services ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

genai.configure(api_key=GEMINI_API_KEY)
embedding_model = genai.get_embedding_model(
    model="models/text-embedding-004"  # free-tier
)

mistral = MistralClient(api_key=MISTRAL_API_KEY)

# ---------------- App ----------------
app = FastAPI(title="Nutrition RAG API")

# ---------------- Request schema ----------------
class QueryRequest(BaseModel):
    message: str

# ---------------- Helpers ----------------
def embed_query(text: str):
    result = embedding_model.embed_content(text)
    return result["embedding"]

def build_context(chunks):
    return "\n\n".join(
        f"[{i+1}] (Page {c['metadata'].get('page', '?')}) {c['content']}"
        for i, c in enumerate(chunks)
    )

# ---------------- RAG Endpoint ----------------
@app.post("/chat")
def chat(req: QueryRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty query")

    # 1️⃣ Embed query (Gemini free-tier)
    query_embedding = embed_query(message)

    # 2️⃣ Retrieve from Supabase
    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 8,
            "filter": {"source": "human-nutrition-text.pdf"},
        },
    ).execute()

    chunks = response.data or []

    if not chunks:
        return {
            "answer": (
                "I couldn’t find this in the provided document. "
                "Try rephrasing or asking about a different section."
            ),
            "sources": [],
        }

    # 3️⃣ Build grounded context
    context = build_context(chunks)

    # 4️⃣ Generate answer (Mistral)
    completion = mistral.chat(
        model="mistral-small-latest",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG assistant. Answer ONLY using the CONTEXT. "
                    "If the answer is not present, say: "
                    "'I couldn’t find this in the provided document.' "
                    "Cite sources like [1], [2] with page numbers."
                ),
            },
            {
                "role": "user",
                "content": f"QUESTION: {message}\n\nCONTEXT:\n{context}",
            },
        ],
        temperature=0.2,
    )

    return {
        "answer": completion.choices[0].message.content,
        "sources": chunks,
    }
