
import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

groq_client = Groq(api_key=api_key)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(name="test_docs")


def index_documents(documents):
    for i, document in enumerate(documents):
        embedding = embedding_model.encode(document).tolist()

        collection.upsert(
            ids=[f"doc_{i}"],
            documents=[document],
            embeddings=[embedding]
        )


def retrieve_context(question):
    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2
    )

    documents = results["documents"][0]

    context = "\n".join(documents)

    return context


def generate_answer(question, context):
    prompt = f"""
Answer the question using the context below.
If the answer is not available in the context, say so.
Do not make up information.

Context:
{context}

Question:
{question}
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return answer


def main():
    documents = [
        "RAG retrieves relevant documents before generating an answer.",
        "FastAPI is a Python framework for building APIs.",
        "Embeddings represent text as numerical vectors."
    ]

    index_documents(documents)

    question = "What does RAG do?"

    context = retrieve_context(question)

    if not context.strip():
        print("No relevant documents were found.")
        return

    answer = generate_answer(question, context)

    print("Retrieved context:")
    print(context)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()

