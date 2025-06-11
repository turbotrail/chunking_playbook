import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def analyze_document(document_path: str):
    """
    Analyze a document using Azure Document Intelligence and save raw results.
    
    Args:
        document_path: Path to the document file
    """
    # Get Azure Document Intelligence credentials
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    if not endpoint or not key:
        raise ValueError(
            "Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
            "AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables"
        )
    
    # Initialize the client
    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    try:
        # Read the document file
        with open(document_path, "rb") as f:
            document = f.read()

        # Start the analysis
        print("Starting document analysis...")
        poller = client.begin_analyze_document(
            "prebuilt-document", document
        )
        result = poller.result()

        # Save raw result to file
        output_file = "raw_analysis_output.txt"
        with open(output_file, 'w') as f:
            f.write(str(result))
        print(f"\nRaw analysis output saved to {output_file}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    # Analyze the document
    document_path = "ticket.pdf"  # Change this to your document path
    analyze_document(document_path) 