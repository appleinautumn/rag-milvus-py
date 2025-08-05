from ingest import ingest_texts
from query import query_rag
from milvus_client import init_milvus

if __name__ == "__main__":

    # Initialize Milvus
    collection = init_milvus()

    # Step 1: Ingest
    texts = [
        "Milvus is an open-source vector database.",
        "It supports similarity search with embeddings.",
    ]
    ingest_texts(texts, collection)

    # Step 2: Query
    answer = query_rag("What is Milvus?", collection)
    print("Answer:", answer)
