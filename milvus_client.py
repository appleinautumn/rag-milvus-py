from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)


def init_milvus():
    connections.connect(alias="default", host="localhost", port="19530")

    if utility.has_collection("rag_demo"):
        collection = Collection("rag_demo")
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        ]
        schema = CollectionSchema(fields, description="RAG collection")
        collection = Collection(name="rag_demo", schema=schema)

    return collection
