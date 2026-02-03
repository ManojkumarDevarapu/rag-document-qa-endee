RAG-Based Document Question Answering using Endee Vector Database
Overview

In many real-world scenarios—placements, hiring, HR operations, compliance, or internal knowledge bases—important information is buried inside long documents such as resumes, job descriptions, policies, or manuals.
Finding answers manually is slow and inefficient.

This project builds a Retrieval Augmented Generation (RAG) system that allows users to ask natural-language questions over documents and get accurate, context-grounded answers.

The system is designed around semantic search using vector embeddings, with Endee integrated as the vector database service.

Problem Statement

Traditional keyword-based search struggles because:

Users ask questions differently than the document wording

Documents are long and unstructured

Relevant information is spread across multiple sections

The goal of this project is to design and implement an AI system that:

Understands the meaning of a question

Retrieves the most relevant document content semantically

Generates answers strictly grounded in the source documents

What This Project Does

Accepts PDF documents (resumes, JDs, etc.)

Splits documents into meaningful semantic chunks

Converts chunks into vector embeddings

Performs similarity-based retrieval

Uses RAG to generate accurate answers

Example questions:

What technical skills are mentioned?

What is the education background?

What projects are listed?

What is the company name?

AI / ML Concepts Demonstrated

Retrieval Augmented Generation (RAG)

Semantic Search using vector embeddings

Cosine similarity–based retrieval

Context-grounded Question Answering

Chunking strategies for long documents

System Design & Technical Approach
1️.Document Ingestion

PDF documents are loaded and converted to text

Text is split into overlapping chunks to preserve semantic meaning

2️.Embedding Generation

Each chunk is converted into a dense vector using
sentence-transformers/all-MiniLM-L6-v2

3️.Vector Database Layer (Endee)

Endee is deployed as a vector database service using Docker

REST APIs are used to integrate vector search functionality

Vector search is performed based on semantic similarity

4️.Retrieval Augmented Generation

Top-K relevant chunks are retrieved

Retrieved context is passed to a Question Answering model

Final answers are generated only from retrieved context (no hallucination)

How Endee Is Used in This Project

Endee runs as the vector database service

The application integrates with Endee via REST APIs

Vector search and index management flows are designed around Endee’s API model

Important Engineering Note

While working with the open-source Docker build of Endee, it was observed that:

Vector search/read APIs are exposed

Vector write operations are restricted in the OSS image

The ingestion pipeline is fully implemented, and this limitation is clearly documented.

To ensure the RAG pipeline remains functional and demonstrable, a local fallback vector store is used for persistence, while Endee remains integrated as the vector database service.

This design mirrors real-world engineering practice, where system constraints are handled gracefully rather than ignored.

Tech Stack

Language: Python

Vector Database: Endee

Embedding Model: Sentence Transformers (MiniLM)

QA Model: DistilBERT (SQuAD)

PDF Parsing: PyPDF

Containerization: Docker

APIs: REST

Project Structure
RAG-Document-QA/
│
├── data/
│   └── documents/        # Input PDF files
│
├── src/
│   ├── rag.py             # Retrieval + QA pipeline
│   ├── ingest_endee.py    # Endee ingestion logic
│   ├── utils.py           # PDF loading & chunking utilities
│
├── requirements.txt
├── README.md

Setup & Execution
1️.Clone the Repository
git clone <repository-url>
cd RAG-Document-QA

2️.Start Endee (Docker)
docker pull endeeio/endee-server:latest
docker run -d -p 8080:8080 -v endee-data:/data --name endee-server endeeio/endee-server:latest

3️.Create Virtual Environment & Install Dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

4️.Add Documents

Place PDF files inside:

data/documents/

5.Run the RAG Application
python src/rag.py


Ask questions directly in the terminal.

Example Output
> What is the education background?
Bachelor of Technology in Computer Science

> What projects are mentioned?
VS Code Projects Resume Screening System

> What is the company name?
NovaTech Solutions

Limitations & Future Improvements

Add OCR support for scanned/image-based PDFs

Enable full vector persistence with a production Endee build

Web UI for document upload and chat interface

Multi-document filtering and metadata-based retrieval

Why This Project Matters

This project demonstrates not just AI model usage, but real-world system design:

Understanding of RAG beyond simple demos

Practical vector search integration

Honest handling of infrastructure constraints

Clean, explainable architecture

Final Note

This project reflects how AI systems are built in practice — with trade-offs, constraints, and engineering decisions — not just idealized demos.