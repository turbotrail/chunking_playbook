import os
import json
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "="*50)
    print("="*20, title, "="*20)
    print("="*50 + "\n")

def analyze_document(endpoint: str, key: str, document_path: str):
    """
    Analyze a document using Azure Document Intelligence.
    
    Args:
        endpoint: Azure Document Intelligence endpoint URL
        key: Azure Document Intelligence API key
        document_path: Path to the document file
    """
    converted_path = None
    try:
        # Initialize the Document Intelligence client
        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )

        # Read the document file
        with open(document_path, "rb") as f:
            document = f.read()

        # Start the document analysis
        print("Starting document analysis...")
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-document", document
        )
        result = poller.result()

        # 1. Document Information
        print_section("Document Information")
        print(f"Number of Pages: {len(result.pages)}")
        if hasattr(result, 'languages') and result.languages:
            print(f"Languages: {[lang.language_code for lang in result.languages]}")

        # 2. Text Content
        print_section("Text Content")
        print("First few paragraphs:")
        for paragraph in result.paragraphs[:3]:  # Show first 3 paragraphs
            print(f"\nParagraph:")
            print(f"Content: {paragraph.content}")
            if hasattr(paragraph, 'role'):
                print(f"Role: {paragraph.role}")

        # 3. Tables
        print_section("Tables")
        print(f"Number of tables found: {len(result.tables)}")
        
        for i, table in enumerate(result.tables):
            print(f"\nTable {i+1}:")
            if hasattr(table, 'confidence'):
                print(f"Confidence: {table.confidence:.2f}")
            
            # Get cells from the table
            cells = table.cells if hasattr(table, 'cells') else []
            if cells:
                # Group cells by row
                rows = {}
                for cell in cells:
                    row_index = cell.row_index
                    if row_index not in rows:
                        rows[row_index] = []
                    rows[row_index].append(cell)
                
                # Sort cells in each row by column index
                for row_index in rows:
                    rows[row_index].sort(key=lambda x: x.column_index)
                
                # Print table content
                if rows:
                    print("\nTable Content:")
                    for row_index in sorted(rows.keys()):
                        row_cells = rows[row_index]
                        row_content = [cell.content for cell in row_cells]
                        print(row_content)

        # 4. Images Analysis
        print_section("Images Analysis")
        
        # Analyze images from each page
        for page_num, page in enumerate(result.pages):
            print(f"\nAnalyzing Page {page_num + 1}:")
            
            # Get page images
            if hasattr(page, 'images'):
                print(f"Number of images on page {page_num + 1}: {len(page.images)}")
                
                for img_idx, img in enumerate(page.images):
                    print(f"\nImage {img_idx + 1}:")
                    
                    # Print image properties
                    if hasattr(img, 'confidence'):
                        print(f"Confidence: {img.confidence:.2f}")
                    
                    if hasattr(img, 'bounding_box'):
                        print(f"Bounding Box: {img.bounding_box}")
                    
                    # Get image content if available
                    if hasattr(img, 'content'):
                        print(f"Content Type: {img.content.content_type}")
                        print(f"Content Length: {len(img.content.bytes)} bytes")
                    
                    # Get image annotations if available
                    if hasattr(img, 'annotations'):
                        print("\nAnnotations:")
                        for annotation in img.annotations:
                            print(f"- {annotation.content}")

        # 5. Key-Value Pairs
        print_section("Key-Value Pairs")
        print(f"Number of key-value pairs found: {len(result.key_value_pairs)}")
        
        for i, kv_pair in enumerate(result.key_value_pairs[:3]):  # Show first 3 pairs
            if kv_pair.key and kv_pair.value:
                print(f"\nPair {i+1}:")
                print(f"Key: {kv_pair.key.content}")
                print(f"Value: {kv_pair.value.content}")

        # 6. Save Detailed Results
        print_section("Saving Results")
        results = {
            "pages": len(result.pages),
            "languages": [lang.language_code for lang in result.languages] if hasattr(result, 'languages') else [],
            "paragraphs": [
                {
                    "content": p.content,
                    "role": p.role if hasattr(p, 'role') else None
                }
                for p in result.paragraphs
            ],
            "tables": [
                {
                    "confidence": t.confidence if hasattr(t, 'confidence') else None,
                    "cells": [
                        {
                            "content": cell.content,
                            "row_index": cell.row_index,
                            "column_index": cell.column_index
                        }
                        for cell in (t.cells if hasattr(t, 'cells') else [])
                    ]
                }
                for t in result.tables
            ],
            "images": [
                {
                    "page_number": page_num + 1,
                    "image_index": img_idx + 1,
                    "confidence": img.confidence if hasattr(img, 'confidence') else None,
                    "bounding_box": img.bounding_box if hasattr(img, 'bounding_box') else None,
                    "content_type": img.content.content_type if hasattr(img, 'content') else None,
                    "content_length": len(img.content.bytes) if hasattr(img, 'content') else None,
                    "annotations": [
                        {
                            "content": annotation.content,
                            "confidence": annotation.confidence if hasattr(annotation, 'confidence') else None
                        }
                        for annotation in (img.annotations if hasattr(img, 'annotations') else [])
                    ]
                }
                for page_num, page in enumerate(result.pages)
                for img_idx, img in enumerate(page.images if hasattr(page, 'images') else [])
            ]
        }
        
        output_file = "document_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {output_file}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise
    finally:
        # Clean up temporary file if it was created
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
                print(f"Cleaned up temporary file: {converted_path}")
            except Exception as e:
                print(f"Warning: Could not clean up temporary file: {str(e)}")

if __name__ == "__main__":
    # Get Azure Document Intelligence credentials from environment variables
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    if not endpoint or not key:
        raise ValueError(
            "Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
            "AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables"
        )
    
    # Analyze the document
    document_path = "ticket.pdf"  # Change this to your document path
    analyze_document(endpoint, key, document_path) 