from milvus1.embedder import get_embedding


def ingest_texts(texts: list[str], collection, api_key: str):
    embeddings = [get_embedding(text, api_key) for text in texts]
    data = [texts, embeddings]
    collection.insert(data)
    collection.flush()
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    print("Index built.")
