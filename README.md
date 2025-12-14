🥗 Nutrition RAG AI Agent

A Retrieval-Augmented Generation (RAG) system that answers nutrition-related questions strictly from provided dietary PDFs, ensuring grounded, safe, and non-hallucinated responses.

📌 Overview

This project implements an end-to-end document intelligence pipeline for nutrition content.
It ingests nutrition PDFs, generates semantic embeddings, stores them in a vector database, retrieves the most relevant sections for a user query, and generates source-grounded answers using a Large Language Model.

The system is designed with AI safety in mind, making it suitable for health and education domains.

🧠 System Architecture
Nutrition PDF
   ↓
Text Cleaning & Chunking
   ↓
Gemini Embeddings (Free-Tier)
   ↓
Supabase + pgvector (Vector Store)
   ↓
Similarity Search (Cosine)
   ↓
Context Retrieval
   ↓
Mistral LLM (Answer Generation)

⚙️ Tech Stack

Language: Python

Backend: FastAPI

Embeddings: Gemini text-embedding-004 (free-tier)

LLM: Mistral (open-source / API-based)

Vector Database: Supabase + pgvector

PDF Processing: PyMuPDF

Similarity Metric: Cosine similarity

🔐 AI Safety & Hallucination Control

This system follows strict safety rules:

Retrieval-Augmented Generation (RAG) for grounding

Answers generated only from retrieved document context

Explicit refusal when information is not present

Low-temperature generation to reduce hallucinations

Source citations with page numbers

If a question is outside the document scope, the system responds:

“I couldn’t find this in the provided document.”

📂 Project Structure
nutrition-rag-ai/
│── ingest.py           # PDF ingestion & embedding pipeline
│── app.py              # FastAPI RAG backend
│── requirements.txt    # Python dependencies
│── .env.example        # Environment variable template
│── README.md           # Project documentation
│── examples.md         # Sample queries & responses

🚀 How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Configure environment variables

Create a .env file:

SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
GEMINI_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key

3️⃣ Ingest the nutrition PDF
python ingest.py


This will:

Extract and clean text from the PDF

Chunk the document

Generate embeddings

Store vectors in Supabase

4️⃣ Start the API server
uvicorn app:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs

🧪 Example Queries

“What are macronutrients?”

“Explain the role of vitamins in human nutrition.”

“Does the document mention ketogenic diets?”

If the answer is not present in the PDF, the system safely refuses.

📊 Similarity Search

Uses cosine similarity via pgvector

Top-K retrieval through a PostgreSQL RPC function

Similarity scores computed inside the database

No hallucinated responses when similarity is low or missing

🎯 Use Cases

Nutrition education assistants

Health guideline Q&A systems

Academic document understanding

Safe AI applications in regulated domains

🧠 Key Learnings

Building production-style RAG pipelines

Vector similarity search with pgvector

AI safety and hallucination mitigation

End-to-end LLM system design

📌 Future Improvements

Similarity thresholding for stricter filtering

Local deployment using Ollama (Mistral)

Multi-document ingestion support

UI integration

👤 Author

Jayanth
