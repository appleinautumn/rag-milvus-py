from milvus_client import init_milvus
from embedder import get_embedding


def ingest_texts(texts: list[str]):
    collection = init_milvus()
    embeddings = [get_embedding(text) for text in texts]

    data = [texts, embeddings]
    collection.insert(data)  # ✅ This results in 2 columns: texts, embeddings

    collection.flush()

    # ✅ Build the index
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "L2",  # or "COSINE"
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    )
    print("Index built.")
