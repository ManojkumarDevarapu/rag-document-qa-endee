import os
import requests
from sentence_transformers import SentenceTransformer
from utils import load_pdf_text, chunk_text

ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "documents"
PDF_FOLDER = "data/documents"

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def ingest_file(file_path, start_id):
    text = load_pdf_text(file_path)
    chunks = chunk_text(text)

    if not chunks:
        return start_id

    embeddings = embedder.encode(chunks).tolist()
    ids = [str(start_id + i) for i in range(len(chunks))]

    payload = {
        "index_name": INDEX_NAME,
        "embeddings": embeddings,
        "documents": chunks,
        "ids": ids
    }

    response = requests.post(
        f"{ENDEE_URL}/api/v1/vector/add",
        json=payload
    )

    if response.status_code != 200:
        print("Upsert failed:", response.text)

    return start_id + len(chunks)


if __name__ == "__main__":
    current_id = 0

    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            current_id = ingest_file(path, current_id)

    print("Ingestion completed into Endee")
