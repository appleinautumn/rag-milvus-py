from openai import OpenAI


def get_embedding(text: str, api_key: str) -> list[float]:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-ada-002", input=[text]  # Wrap in list
    )
    return response.data[0].embedding
