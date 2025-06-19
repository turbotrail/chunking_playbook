# Table RAG with Gemma and ChromaDB

This project implements a Retrieval-Augmented Generation (RAG) system specifically designed for handling table-intensive PDF documents. It uses the Gemma language model through Ollama for text generation and ChromaDB as the vector store.

## Features

- Extracts tables and text from PDF documents using Unstructured.io
- Processes tables with contextual information to improve understanding
- Converts tables to markdown format for better representation
- Uses ChromaDB for efficient vector storage and retrieval
- Implements RAG using Langchain with the Gemma model

## Prerequisites

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull the Gemma model:
```bash
ollama pull gemma:2b
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your PDF document in the appropriate directory (default: `table/report.pdf`)

2. Run the main script:
```bash
python table_rag.py
```

3. The script will:
   - Extract tables and text from the PDF
   - Process tables with context
   - Create a vector store using ChromaDB
   - Set up a question-answering chain
   - Run example questions

## Customization

You can modify the following in `table_rag.py`:
- Change the model by updating `model_name` in `TableRAG` initialization
- Adjust the vector store settings in `create_vector_store`
- Modify the prompt template in `setup_qa_chain`
- Add your own questions in the `main` function

## How It Works

1. **PDF Processing**:
   - Uses Unstructured.io to extract tables and text
   - Maintains document structure with chunking strategy

2. **Table Processing**:
   - Adds contextual information to each table
   - Converts tables to markdown format
   - Preserves table structure and relationships

3. **Vector Store**:
   - Uses ChromaDB for efficient storage
   - Maintains metadata about chunk types
   - Persists data for reuse

4. **Question Answering**:
   - Uses Gemma model through Ollama
   - Retrieves relevant context from vector store
   - Generates natural language answers

## Requirements

See `requirements.txt` for the complete list of dependencies.

## License

This project is licensed under the terms of the license included in the repository.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements. 