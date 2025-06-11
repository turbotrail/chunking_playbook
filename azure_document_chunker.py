import os
import json
from typing import List, Dict, Any
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AzureDocumentChunker:
    def __init__(self):
        """Initialize the Azure Document Chunker."""
        self.endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        if not self.endpoint or not self.key:
            raise ValueError(
                "Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
                "AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables"
            )
        
        self.client = DocumentAnalysisClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
        self.chunks = []

    def process_document(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Process a document and create chunks.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            List of chunks with their content and metadata
        """
        try:
            # Read the document file
            with open(document_path, "rb") as f:
                document = f.read()

            # Start the document analysis
            print("Starting document analysis...")
            poller = self.client.begin_analyze_document(
                "prebuilt-document", document
            )
            result = poller.result()

            # Process the result
            self.chunks = self._process_analyze_result(result)
            return self.chunks

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise

    def _process_analyze_result(self, result) -> List[Dict[str, Any]]:
        """Process the analyze result and create chunks."""
        chunks = []
        
        # Process paragraphs
        for paragraph in result.paragraphs:
            chunk = {
                "chunk_id": f"chunk_{len(chunks)}",
                "content_type": "paragraph",
                "content": paragraph.content,
                "metadata": {
                    "role": paragraph.role if hasattr(paragraph, 'role') else None
                }
            }
            chunks.append(chunk)
        
        # Process tables
        for table in result.tables:
            # Create a structured representation of the table
            table_data = []
            if hasattr(table, 'cells'):
                # Group cells by row
                rows = {}
                for cell in table.cells:
                    row_index = cell.row_index
                    if row_index not in rows:
                        rows[row_index] = []
                    rows[row_index].append(cell)
                
                # Sort cells in each row by column index
                for row_index in rows:
                    rows[row_index].sort(key=lambda x: x.column_index)
                
                # Convert to list of lists
                for row_index in sorted(rows.keys()):
                    row_cells = rows[row_index]
                    row_data = [cell.content for cell in row_cells]
                    table_data.append(row_data)
            
            chunk = {
                "chunk_id": f"chunk_{len(chunks)}",
                "content_type": "table",
                "content": table_data,
                "metadata": {
                    "confidence": table.confidence if hasattr(table, 'confidence') else None,
                    "row_count": len(table_data),
                    "column_count": len(table_data[0]) if table_data else 0
                }
            }
            chunks.append(chunk)
        
        # Process images
        for page in result.pages:
            if hasattr(page, 'images'):
                for img in page.images:
                    chunk = {
                        "chunk_id": f"chunk_{len(chunks)}",
                        "content_type": "image",
                        "content": {
                            "content_type": img.content.content_type if hasattr(img, 'content') else None,
                            "content_length": len(img.content.bytes) if hasattr(img, 'content') else None
                        },
                        "metadata": {
                            "confidence": img.confidence if hasattr(img, 'confidence') else None,
                            "bounding_box": img.bounding_box if hasattr(img, 'bounding_box') else None,
                            "page_number": page.page_number,
                            "annotations": [
                                {
                                    "content": annotation.content,
                                    "confidence": annotation.confidence if hasattr(annotation, 'confidence') else None
                                }
                                for annotation in (img.annotations if hasattr(img, 'annotations') else [])
                            ]
                        }
                    }
                    chunks.append(chunk)
        
        # Save chunks to JSON file
        with open('document_chunks.json', 'w') as f:
            json.dump(chunks, f, indent=2)
        
        return chunks

    def get_chunks(self) -> List[Dict[str, Any]]:
        """Get the processed chunks."""
        return self.chunks

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "="*50)
    print("="*20, title, "="*20)
    print("="*50 + "\n")

if __name__ == "__main__":
    # Create chunker instance
    chunker = AzureDocumentChunker()
    
    # Process the document
    document_path = "ticket.pdf"  # Change this to your document path
    chunks = chunker.process_document(document_path)
    
    # Print results
    print_section("Processing Results")
    print(f"Total chunks created: {len(chunks)}")
    
    # Print first few chunks
    print("\nFirst few chunks:")
    for chunk in chunks[:3]:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Content Type: {chunk['content_type']}")
        print(f"Content: {chunk['content']}")
        print(f"Metadata: {chunk['metadata']}")
    
    print("\nFull results saved to document_chunks.json") 