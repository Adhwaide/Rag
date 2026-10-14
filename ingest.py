
from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"
DATABASE_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "interview_documents"

CHUNK_SIZE = 500       # Approximate maximum characters per chunk
CHUNK_OVERLAP = 100    # Approximate overlap between chunks


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# --------------------------------------------------
# 3. Initialise ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(DATABASE_DIR)
)


# --------------------------------------------------
# 4. Load documents
# --------------------------------------------------

def load_documents(directory: Path):
    if not directory.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {directory}"
        )

    documents = []

    for file_path in sorted(directory.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8-sig").strip()

        if text:
            documents.append({
                "filename": file_path.name,
                "text": text
            })

    return documents


# --------------------------------------------------
# 5. Split text into sentence-friendly chunks
# --------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    # Normalise whitespace while preserving paragraph boundaries.
    paragraphs = re.split(r"\n\s*\n", text.strip())

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    # Split paragraphs into sentences where possible.
    sentences = []

    for paragraph in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", paragraph)

        for part in parts:
            part = part.strip()

            if part:
                sentences.append(part)

    chunks = []
    current_chunk = ""

    def save_current_chunk():
        nonlocal current_chunk

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        current_chunk = ""

    for sentence in sentences:

        # Handle unusually long sentences separately.
        if len(sentence) > chunk_size:
            save_current_chunk()

            start = 0

            while start < len(sentence):
                end = min(start + chunk_size, len(sentence))

                piece = sentence[start:end].strip()

                if piece:
                    chunks.append(piece)

                if end == len(sentence):
                    break

                start = end - overlap

            continue

        candidate = (
            f"{current_chunk} {sentence}".strip()
            if current_chunk
            else sentence
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate

        else:
            save_current_chunk()

            # Carry forward recent context for overlap.
            if overlap > 0 and chunks:
                previous = chunks[-1]

                overlap_text = previous[-overlap:].strip()

                # Avoid starting with an incomplete word.
                if " " in overlap_text:
                    overlap_text = overlap_text.split(" ", 1)[1]

                if overlap_text:
                    current_chunk = f"{overlap_text} {sentence}"

            if not current_chunk:
                current_chunk = sentence

            # If overlap made the new chunk too large, split it safely.
            if len(current_chunk) > chunk_size:
                save_current_chunk()
                current_chunk = sentence

    save_current_chunk()

    return chunks


# --------------------------------------------------
# 6. Generate embeddings and store documents
# --------------------------------------------------

def ingest_documents():
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        raise ValueError(
            f"No non-empty .txt files found in {DOCUMENTS_DIR}"
        )

    ids = []
    texts = []
    metadatas = []

    for document in documents:
        chunks = chunk_text(document["text"])

        print(
            f"{document['filename']}: {len(chunks)} chunks"
        )

        for index, chunk in enumerate(chunks):
            ids.append(
                f"{document['filename']}_chunk_{index}"
            )

            texts.append(chunk)

            metadatas.append({
                "source": document["filename"],
                "chunk_index": index
            })

    if not texts:
        raise ValueError("No text chunks were generated.")

    print(f"\nGenerating embeddings for {len(texts)} chunks...")

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    # Rebuild the collection to remove stale chunks.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted previous document collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Store records in batches to avoid oversized database requests.
    batch_size = 100

    for start in range(0, len(texts), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end]
        )

    print("\nIngestion completed successfully.")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks stored: {collection.count()}")
    print(f"Database location: {DATABASE_DIR}")


# --------------------------------------------------
# 7. Run ingestion
# --------------------------------------------------

if __name__ == "__main__":
    ingest_documents()