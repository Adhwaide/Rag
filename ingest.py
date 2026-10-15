
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


folder = Path(__file__).parent
docs_folder = folder / "documents"

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient( path=str(folder / "chroma_db"))


def load_docs():
    docs = []

    for file in sorted(docs_folder.glob("*.txt")):
        text = file.read_text(encoding="utf-8-sig").strip()

        if text:
            docs.append(( file.name, text))

    return docs


def split_text(text, size=500, overlap=100):
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text )

    chunks = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        if len(sentence ) > size:
            if current:
                chunks.append(current)
                current = ""

            start = 0
            while start < len(sentence):
                end = min(start + size, len(sentence))
                chunks.append( sentence[start:end])

                if end == len(sentence):
                    break

                start = end - overlap

            continue

        combined = f"{current} { sentence}".strip()

        if len(combined) <= size:
            current = combined
        else:
            chunks.append(current)
            previous = chunks[-1][-overlap:].strip()
            current = f"{previous } {sentence}".strip()

            if len(current) > size:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def build_index():
    files = load_docs()

    if not files:
        print("No documents found.")
        return

    collection_name = "interview_documents"

    try:
        db.delete_collection(collection_name )
    except Exception:
        pass

    collection = db.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadata = [], [], []

    for filename, content in files:
        for i, chunk in enumerate(split_text(content)):
            ids.append(f"{filename}_{i}")
            texts.append(chunk)
            metadata.append({"source": filename, "chunk": i})

    if not texts:
        print("No text to index.")
        return

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    for start in range( 0, len(texts), 100):
        end = start + 100

        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadata[start:end],
        )

    print(f"Indexed { collection.count()} chunks.")


if __name__ == "__main__":
    build_index()

