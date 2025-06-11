import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
import subprocess
import tempfile
import pandas as pd
from PIL import Image
import io
import base64
from docx.shared import Inches
import pytesseract
from docx.oxml.shape import CT_Picture
from docx.oxml.text.run import CT_R
from docx.text.run import Run
from sentence_transformers import SentenceTransformer
import requests

# Load environment variables from .env file
load_dotenv()

# Get Ollama configuration from environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

def summarize_with_ollama(text, model=OLLAMA_MODEL):
    payload = {
        "model": model,
        "prompt": f"Summarize the following content:\n\n{text}\n\nSummary:",
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"[Summary error: {e}]"

class ContentType:
    TITLE = "title"
    SUBTITLE = "subtitle"
    PARAGRAPH = "paragraph"
    SUBPARAGRAPH = "subparagraph"
    TABLE = "table"
    IMAGE = "image"

class MultimodalChunker:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.chunks = []
        self.current_level = 0
        self.parent_stack = []

    def _generate_chunk_id(self) -> str:
        return f"chunk_{len(self.chunks)}"

    def _summarize_table(self, table_structure: Dict[str, Any]) -> str:
        """Generate a summary for a table using Ollama."""
        # Convert table to markdown for LLM input
        headers = table_structure["headers"]
        rows = table_structure["data"]
        table_md = '| ' + ' | '.join(headers) + ' |\n'
        table_md += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
        for row in rows[:10]:  # Limit to first 10 rows for prompt size
            table_md += '| ' + ' | '.join(str(row.get(h, '')) for h in headers) + ' |\n'
        summary = summarize_with_ollama(table_md)
        return summary

    def _summarize_image(self, image_data: Dict[str, Any]) -> str:
        """Generate a summary for an image using captioning and OCR."""
        caption = "Image description"  # Replace with actual image captioning model output
        if isinstance(image_data, dict) and "metadata" in image_data:
            ocr_text = image_data["metadata"].get("extracted_text", "")
        else:
            ocr_text = ""
        return f"Image: {caption}. Extracted text: {ocr_text}"

    def _extract_image(self, run: Run) -> Optional[Dict[str, Any]]:
        """Extract image data from a run element."""
        for element in run._element.iter():
            if isinstance(element, CT_Picture):
                blip_elems = element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip')
                for blip_elem in blip_elems:
                    embed = blip_elem.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                    if embed:
                        return {
                            "image_id": embed,
                            "type": "embedded"
                        }
        return None

    def _process_image(self, image_data: Dict[str, Any], doc: Document) -> Dict[str, Any]:
        """Process an image and extract its metadata."""
        try:
            image_part = doc.part.related_parts[image_data["image_id"]]
            image_bytes = image_part.blob
            image = Image.open(io.BytesIO(image_bytes))
            metadata = {
                "format": image.format,
                "size": image.size,
                "mode": image.mode,
                "width": image.width,
                "height": image.height
            }
            buffered = io.BytesIO()
            image.save(buffered, format=image.format)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            try:
                text = pytesseract.image_to_string(image)
                if text.strip():
                    metadata["extracted_text"] = text.strip()
            except:
                pass
            return {
                "metadata": metadata,
                "base64_data": img_str
            }
        except Exception as e:
            return {
                "error": str(e),
                "metadata": {}
            }

    def _parse_table(self, table: Table) -> Dict[str, Any]:
        """Parse a table into a structured format using pandas."""
        data = []
        headers = []
        for cell in table.rows[0].cells:
            headers.append(cell.text.strip())
        for row in table.rows[1:]:
            row_data = []
            for cell in row.cells:
                row_data.append(cell.text.strip())
            data.append(row_data)
        df = pd.DataFrame(data, columns=headers)
        column_types = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        table_structure = {
            "headers": headers,
            "data": df.to_dict('records'),
            "shape": df.shape,
            "summary": {
                "column_types": column_types,
                "missing_values": df.isnull().sum().to_dict()
            }
        }
        return table_structure

    def process_content(self, content: Any, style: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        content_type = ContentType.PARAGRAPH
        if style.lower().startswith(('heading', 'title')):
            content_type = ContentType.TITLE
        elif style.lower().startswith(('list', 'bullet')):
            content_type = ContentType.SUBPARAGRAPH
        elif metadata and 'table_structure' in metadata:
            content_type = ContentType.TABLE
        elif metadata and 'base64_data' in metadata:
            content_type = ContentType.IMAGE
        chunk = {
            "chunk_id": self._generate_chunk_id(),
            "content_type": content_type,
            "content": content,
            "metadata": {
                "original_length": len(str(content)),
                "style": style
            }
        }
        if metadata:
            chunk["metadata"].update(metadata)
        level = 1
        if style.lower().startswith('heading'):
            try:
                level = int(style.split()[-1])
            except (ValueError, IndexError):
                pass
        chunk["level"] = level
        chunk["parent_id"] = self._determine_parent_id(level)
        # Summarize with Ollama only for tables and images
        if content_type == ContentType.TABLE:
            chunk["summary"] = self._summarize_table(content)
        elif content_type == ContentType.IMAGE:
            chunk["summary"] = self._summarize_image(content)
        self.chunks.append(chunk)

    def _determine_parent_id(self, level: int) -> Optional[str]:
        """Determine the parent ID for a chunk based on its level."""
        if not self.chunks:
            return None
        for chunk in reversed(self.chunks):
            if chunk["level"] < level:
                return chunk["chunk_id"]
        return None

    def _convert_doc_to_docx(self, doc_path: str) -> str:
        """Convert .doc file to .docx format using LibreOffice."""
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, 'converted_document.docx')
        try:
            subprocess.run([
                'soffice',
                '--headless',
                '--convert-to', 'docx',
                '--outdir', temp_dir,
                doc_path
            ], check=True)
            converted_path = os.path.join(temp_dir, os.path.splitext(os.path.basename(doc_path))[0] + '.docx')
            if os.path.exists(converted_path):
                return converted_path
            else:
                raise FileNotFoundError("Conversion failed: Output file not found")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to convert document: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during conversion: {str(e)}")

    def process_word_document(self, doc_path: str) -> None:
        """Process a Word document and chunk its contents."""
        if doc_path.lower().endswith('.doc'):
            doc_path = self._convert_doc_to_docx(doc_path)
        doc = Document(doc_path)
        for element in doc.element.body:
            if isinstance(element, CT_P):
                paragraph = Paragraph(element, doc)
                if paragraph.text.strip():
                    for run in paragraph.runs:
                        image_data = self._extract_image(run)
                        if image_data:
                            processed_image = self._process_image(image_data, doc)
                            self.process_content(
                                f"[image: {processed_image['metadata'].get('extracted_text', 'Image')}]",
                                paragraph.style.name,
                                processed_image
                            )
                    self.process_content(paragraph.text, paragraph.style.name)
            elif isinstance(element, CT_Tbl):
                table = Table(element, doc)
                table_structure = self._parse_table(table)
                self.process_content(
                    table_structure,
                    "Table",
                    {"table_structure": table_structure}
                )

    def get_chunks(self) -> List[Dict[str, Any]]:
        """Return the processed chunks in a serializable format."""
        return self.chunks

    def save_chunks(self, output_file: str) -> None:
        """Save the chunks to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.get_chunks(), f, indent=2)

if __name__ == "__main__":
    chunker = MultimodalChunker(max_chunk_size=1000)
    chunker.process_word_document('sample.doc')
    chunker.save_chunks('multimodal_chunks.json') 