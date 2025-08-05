from milvus_client import init_milvus
from embedder import get_embedding
import openai


def query_rag(user_query: str, collection):
    collection.load()
    embedding = get_embedding(user_query)
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["text"],
    )
    context = "\n".join(hit.entity.get("text") for hit in results[0])
    prompt = (
        f"Answer the question using the context:\n{context}\nQuestion: {user_query}"
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
