from sentence_transformers import SentenceTransformer
from transformers import pipeline
from utils import load_pdf_text, chunk_text
import numpy as np

PDF_FOLDER = "data/documents"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

chunks = []
embeddings = []


def ingest_local():
    global chunks, embeddings
    chunks.clear()
    embeddings.clear()

    import os
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            text = load_pdf_text(f"{PDF_FOLDER}/{file}")
            ch = chunk_text(text)
            if ch:
                emb = embedder.encode(ch)
                chunks.extend(ch)
                embeddings.extend(emb)

    embeddings[:] = np.array(embeddings)


def retrieve(question, top_k=5):
    q_vec = embedder.encode([question])[0]
    sims = embeddings @ q_vec
    idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in idx]


def answer(question, contexts):
    best = {"score": 0}
    for ctx in contexts:
        out = qa(question=question, context=ctx)
        if out["score"] > best["score"]:
            best = out

    if best["score"] < 0.15:
        return "Answer not found in the document."

    return best["answer"]


if __name__ == "__main__":
    ingest_local()
    print("Ask a question (type exit to quit):")

    while True:
        q = input("> ")
        if q.lower() == "exit":
            break

        ctxs = retrieve(q)
        ans = answer(q, ctxs)

        print("\nAnswer:")
        print(ans)
        print()
