from pathlib import Path
import chromadb

DATABASE_DIR = Path(__file__).resolve().parent / "chroma_db"

client = chromadb.PersistentClient(
    path=str(DATABASE_DIR)
)

collection = client.get_collection(
    name="interview_documents"
)

print("Stored chunks:", collection.count())

results = collection.get(
    limit=3,
    include=["documents", "metadatas"]
)

for document, metadata in zip(
    results["documents"],
    results["metadatas"]
):
    print("\nSource:", metadata["source"])
    print("Chunk:", metadata["chunk_index"])
    print("Text:", document)