from milvus1.ingest import ingest_texts
from milvus1.query import query_rag
from milvus1.milvus_client import init_milvus
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    # Initialize Milvus
    collection = init_milvus()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")

    # Step 1: Ingest
    texts = [
        "Milvus is an open-source vector database.",
        "It supports similarity search with embeddings.",
    ]
    ingest_texts(texts, collection, api_key)

    # Step 2: Query
    answer = query_rag("What is the feature of Milvus?", collection, api_key)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
