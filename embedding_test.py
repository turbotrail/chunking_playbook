from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Load the chunks from document_chunks.json
with open('document_chunks.json', 'r') as f:
    chunks = json.load(f)

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for each chunk's content
chunk_embeddings = []
for chunk in chunks:
    content = chunk['content']
    # If content is a dict (e.g., table_structure), convert it to a string
    if isinstance(content, dict):
        content = json.dumps(content)
    embedding = model.encode(content)
    chunk_embeddings.append(embedding)

# Example query
query = "What are the main sections of the document?"
query_embedding = model.encode(query)

# Compute cosine similarity between query and each chunk
similarities = []
for embedding in chunk_embeddings:
    similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
    similarities.append(similarity)

# Get the top 3 most similar chunks
top_indices = np.argsort(similarities)[-3:][::-1]
print("Top 3 most similar chunks for query: '{}'".format(query))
for idx in top_indices:
    print("Chunk ID: {}, Content: {}, Similarity: {:.4f}".format(
        chunks[idx]['chunk_id'],
        chunks[idx]['content'],
        similarities[idx]
    )) 