from ingest import ingest_texts
from query import query_rag

if __name__ == "__main__":
    # Step 1: Ingest
    texts = [
        "Milvus is an open-source vector database.",
        "It supports similarity search with embeddings.",
    ]
    ingest_texts(texts)

    # Step 2: Query
    answer = query_rag("What is Milvus?")
    print("Answer:", answer)
