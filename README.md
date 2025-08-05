# RAG Pipeline with Milvus and OpenAI Embeddings

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using Milvus as a vector database and OpenAI for text embeddings.

## Features

- Ingests text data and stores both raw text and embeddings in Milvus
- Builds a vector index for efficient similarity search
- Supports semantic search and question answering using OpenAI embeddings

## Dependencies

- [Milvus](https://milvus.io/) (vector database, must be running locally or accessible remotely)
- [Poetry](https://python-poetry.org/) (for dependency management)
- OpenAI API key

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/appleinautumn/rag-milvus-py
   cd your-repo
   ```

2. **Install dependencies**  
   This project uses [Poetry](https://python-poetry.org/) for dependency management:
   ```bash
   poetry install
   ```

3. **Set up environment variables**  
   - Copy `.env.example` to `.env` and fill in your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

4. **Run Milvus**  
   - Make sure you have a Milvus instance running locally on port 19530.

5. **Run the project:**
   ```
   poetry run main
   ```

## Usage

- **Ingest and Query Example**  
  Run the main script to ingest sample texts and perform a query:
  ```bash
  poetry run python main.py
  ```

- **Modify `main.py`** to change the texts ingested or the query asked.

## File Structure

- `main.py` - Entry point; ingests texts and runs a sample query.
- `ingest.py` - Handles ingestion of texts and embeddings into Milvus.
- `embedder.py` - Generates embeddings using OpenAI API.
- `milvus_client.py` - Initializes and manages the Milvus collection.
- `query.py` - Handles querying the vector database.
- `.env.example` - Template for required environment variables.

## Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key for generating embeddings.

## License

MIT License (add your license here if different)
