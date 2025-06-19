import os
import fitz
import nltk
import chromadb
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tabulate import tabulate

# Download required NLTK data
nltk.download('punkt', quiet=True)

class TableRAG:
    def __init__(self, model_name: str = "gemma3:4b"):
        """Initialize the TableRAG system.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.llm = Ollama(model=model_name)
        self.vector_store = None
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from a PDF file for context."""
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def process_table(self, table_content: str, document_context: str) -> str:
        """Process a table by adding context and converting to markdown format."""
        # Prompt template for table processing
        prompt = f"""
        Given the following table and its context from the original document,
        provide a detailed description of the table and convert it to markdown format.
        
        Original Document Context:
        {document_context}
        
        Table Content:
        {table_content}
        
        Please provide:
        1. A comprehensive description of what this table represents
        2. The table in clean markdown format
        """
        
        # Get response from LLM
        response = self.llm.invoke(prompt)
        return str(response)

    def extract_and_process_tables(self, pdf_path: str) -> List[Dict[str, str]]:
        """Extract tables from PDF and process them with context."""
        # Extract document elements
        elements = partition_pdf(
            pdf_path,
            strategy="hi_res",
            chunking_strategy="by_title",
        )
        
        # Get full document context
        document_context = self.extract_text_from_pdf(pdf_path)
        
        processed_chunks = []
        
        # Process each element
        for element in elements:
            if hasattr(element, 'metadata') and element.metadata.get('table_number') is not None:
                # This is a table element
                processed_content = self.process_table(str(element), document_context)
                processed_chunks.append({
                    "content": processed_content,
                    "type": "table",
                    "source": pdf_path
                })
            else:
                # This is a text element
                processed_chunks.append({
                    "content": str(element),
                    "type": "text",
                    "source": pdf_path
                })
        
        return processed_chunks

    def create_vector_store(self, chunks: List[Dict[str, str]], persist_dir: str = None):
        """Create and persist the vector store from processed chunks."""
        # Create documents for Chroma
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [{"type": chunk["type"], "source": chunk["source"]} for chunk in chunks]
        
        # Initialize Chroma vector store
        self.vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=persist_dir
        )
        
        if persist_dir:
            self.vector_store.persist()

    def setup_qa_chain(self) -> RetrievalQA:
        """Set up the question-answering chain."""
        # Custom prompt template for better table handling
        prompt_template = """
        You are an AI assistant specialized in analyzing documents with tables.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain

    def process_pdf_and_setup(self, pdf_path: str, persist_dir: str = None):
        """Process a PDF file and set up the RAG system."""
        # Extract and process tables
        chunks = self.extract_and_process_tables(pdf_path)
        
        # Create vector store
        self.create_vector_store(chunks, persist_dir)
        
        # Return the QA chain
        return self.setup_qa_chain()

def main():
    # Initialize the TableRAG system
    rag = TableRAG(model_name="gemma3:4b")
    
    # Process a PDF file
    pdf_path = "table/report.pdf"  # Update with your PDF path
    persist_dir = "chroma_db"
    
    # Process PDF and setup QA chain
    qa_chain = rag.process_pdf_and_setup(pdf_path, persist_dir)
    
    # Example questions
    questions = [
        "What are the total revenue figures for 2024?",
        "Can you describe the structure of the operating expenses table?",
        "What were the research and development costs for 2024?"
    ]
    
    # Get answers
    for question in questions:
        print(f"\nQuestion: {question}")
        result = qa_chain({"query": question})
        print(f"Answer: {result['result']}")

if __name__ == "__main__":
    main() 