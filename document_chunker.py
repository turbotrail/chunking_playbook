from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
from enum import Enum
import json
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
import subprocess
import os
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

class ContentType(Enum):
    TITLE = "title"
    SUBTITLE = "subtitle"
    PARAGRAPH = "paragraph"
    SUBPARAGRAPH = "subparagraph"
    TABLE = "table"
    IMAGE = "image"

@dataclass
class ContentChunk:
    content_type: ContentType
    content: str
    metadata: Dict[str, Any]
    level: int
    parent_id: Optional[str] = None
    chunk_id: Optional[str] = None

class DocumentChunker:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
        self.chunks: List[ContentChunk] = []
        self.current_level = 0
        self.parent_stack = []

    def _generate_chunk_id(self) -> str:
        return f"chunk_{len(self.chunks)}"

    def _detect_content_type(self, content: str, style: Optional[str] = None) -> ContentType:
        # Enhanced content type detection using Word styles
        content = content.strip()
        
        if style:
            style_lower = style.lower()
            if 'title' in style_lower:
                return ContentType.TITLE
            elif 'heading' in style_lower:
                return ContentType.SUBTITLE
            elif 'list' in style_lower:
                return ContentType.SUBPARAGRAPH

        # Check for tables (assuming they contain | or - characters)
        if '|' in content or '-' in content:
            return ContentType.TABLE
            
        # Check for images (assuming they're marked with [image: description])
        if content.startswith('[image:') and content.endswith(']'):
            return ContentType.IMAGE
            
        # Check for subparagraphs (indented content)
        if content.startswith('    ') or content.startswith('\t'):
            return ContentType.SUBPARAGRAPH
            
        return ContentType.PARAGRAPH

    def _parse_table(self, table: Table) -> Dict[str, Any]:
        """Parse a table into a structured format using pandas."""
        data = []
        headers = []
        
        # Extract headers
        for cell in table.rows[0].cells:
            headers.append(cell.text.strip())
        
        # Extract data
        for row in table.rows[1:]:
            row_data = []
            for cell in row.cells:
                row_data.append(cell.text.strip())
            data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Convert dtypes to strings for JSON serialization
        column_types = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        
        # Convert to structured format
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

    def _extract_image(self, run: Run) -> Optional[Dict[str, Any]]:
        """Extract image data from a run element."""
        for element in run._element.iter():
            if isinstance(element, CT_Picture):
                # Find the blip element and get the r:embed attribute
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
            # Get the image part from the document
            image_part = doc.part.related_parts[image_data["image_id"]]
            image_bytes = image_part.blob
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Extract image metadata
            metadata = {
                "format": image.format,
                "size": image.size,
                "mode": image.mode,
                "width": image.width,
                "height": image.height
            }
            
            # Convert image to base64 for storage
            buffered = io.BytesIO()
            image.save(buffered, format=image.format)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Try OCR if the image might contain text
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

    def _split_paragraph(self, text: str) -> List[str]:
        """Split long paragraphs into smaller chunks while preserving sentence boundaries."""
        if len(text) <= self.max_chunk_size:
            return [text]

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if len(' '.join(current_chunk + [sentence])) <= self.max_chunk_size:
                current_chunk.append(sentence)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _determine_parent_id(self, level: int) -> Optional[str]:
        """Determine the parent ID for a chunk based on its level."""
        if not self.chunks:
            return None
            
        # Find the most recent chunk with a lower level
        for chunk in reversed(self.chunks):
            if chunk["level"] < level:
                return chunk["chunk_id"]
        return None

    def process_content(self, content: str, style: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Process content and determine its type based on style and content."""
        # Determine content type based on style and content
        content_type = ContentType.PARAGRAPH
        
        # Check if it's a title/heading
        if style.lower().startswith(('heading', 'title')):
            content_type = ContentType.TITLE
        # Check if it's a list item
        elif style.lower().startswith(('list', 'bullet')):
            content_type = ContentType.SUBPARAGRAPH
        # Check if it's a table (based on metadata)
        elif metadata and 'table_structure' in metadata:
            content_type = ContentType.TABLE
        # Check if it's an image (based on metadata)
        elif metadata and 'base64_data' in metadata:
            content_type = ContentType.IMAGE
            
        # Create chunk
        chunk = {
            "chunk_id": f"chunk_{len(self.chunks)}",
            "content_type": content_type.value,
            "content": content,
            "metadata": {
                "original_length": len(str(content)),
                "style": style
            }
        }
        
        # Add additional metadata if provided
        if metadata:
            chunk["metadata"].update(metadata)
            
        # Determine level based on style
        level = 1
        if style.lower().startswith('heading'):
            try:
                level = int(style.split()[-1])
            except (ValueError, IndexError):
                pass
                
        chunk["level"] = level
        chunk["parent_id"] = self._determine_parent_id(level)
        
        self.chunks.append(chunk)

    def _convert_doc_to_docx(self, doc_path: str) -> str:
        """Convert .doc file to .docx format using LibreOffice."""
        # Create a temporary file for the .docx output
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, 'converted_document.docx')
        
        try:
            # Try to convert using LibreOffice
            subprocess.run([
                'soffice',
                '--headless',
                '--convert-to', 'docx',
                '--outdir', temp_dir,
                doc_path
            ], check=True)
            
            # Get the converted file path
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
        # Convert .doc to .docx if necessary
        if doc_path.lower().endswith('.doc'):
            doc_path = self._convert_doc_to_docx(doc_path)
        
        doc = Document(doc_path)
        
        for element in doc.element.body:
            if isinstance(element, CT_P):  # Paragraph
                paragraph = Paragraph(element, doc)
                if paragraph.text.strip():
                    # Check for images in the paragraph
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
            elif isinstance(element, CT_Tbl):  # Table
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

# Example usage
if __name__ == "__main__":
    # Example of processing a Word document
    chunker = DocumentChunker(max_chunk_size=1000)
    
    # Process a Word document
    chunker.process_word_document('sample.doc')
    
    # Save the chunks
    chunker.save_chunks('document_chunks.json') 