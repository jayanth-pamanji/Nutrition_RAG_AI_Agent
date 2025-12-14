# 🥗 Nutrition RAG AI Agent

A Retrieval-Augmented Generation (RAG) system that answers nutrition-related questions **strictly from provided dietary PDFs**, ensuring grounded, reliable, and non-hallucinated responses.

---

## 📌 Overview

This project implements an end-to-end document intelligence pipeline for nutrition content. It ingests nutrition PDFs, converts them into semantic embeddings, stores them in a vector database, retrieves relevant document sections for a user query, and generates answers using a Large Language Model constrained to the retrieved context.

The system is designed with **AI safety and reliability** in mind, making it suitable for health and education domains.

---

## 🧠 Architecture

Nutrition PDF  
→ Text Cleaning & Chunking  
→ Gemini Embeddings (Free-Tier)  
→ Supabase + pgvector  
→ Cosine Similarity Search  
→ Context Retrieval  
→ Mistral LLM (Grounded Answer Generation)

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Backend:** FastAPI  
- **Embeddings:** Gemini `text-embedding-004` (free-tier)  
- **LLM:** Mistral  
- **Vector Database:** Supabase + pgvector  
- **PDF Processing:** PyMuPDF  
- **Similarity Metric:** Cosine similarity  

---

## 🔐 AI Safety & Hallucination Control

- Retrieval-Augmented Generation (RAG) for grounding  
- Answers generated **only from retrieved document content**  
- Explicit refusal when information is not present  
- Low-temperature generation to reduce hallucinations  
- Source citations with page numbers  

If a question is outside the document scope, the system responds:

> *“I couldn’t find this in the provided document.”*

---

## 📂 Project Structure

nutrition-rag-ai/  
│── ingest.py  
│── app.py  
│── requirements.txt  
│── .env.example  
│── README.md  
│── examples.md  

---

## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Configure environment variables
Create a `.env` file:
SUPABASE_URL=  
SUPABASE_SERVICE_ROLE_KEY=  
GEMINI_API_KEY=  
MISTRAL_API_KEY=  

### 3. Ingest the PDF
python ingest.py

### 4. Start the API
uvicorn app:app --reload

Open:
http://127.0.0.1:8000/docs

---

## 🧪 Example Queries

- What are macronutrients?  
- Explain the role of vitamins in nutrition.  
- Does the document mention ketogenic diets?  

If the answer is not present, the system safely refuses.

---

## 📊 Similarity Search

- Cosine similarity computed using pgvector  
- Top-K retrieval via PostgreSQL RPC function  
- Similarity scoring handled at the database level  
- No generation when relevant context is missing  

---

## 🎯 Use Cases

- Nutrition education assistants  
- Health guideline Q&A systems  
- Document understanding for dietary content  
- Safe AI applications in regulated domains  

---

## 📌 Notes

- This project is a **local prototype**  
- Focuses on correctness, grounding, and AI safety  
- Suitable for AI Engineer / Generative AI roles  

---

## 👤 Author

**Jayanth**
