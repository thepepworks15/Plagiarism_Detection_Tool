"""
Document Parser Module
======================
Extracts text content from PDF, DOCX, and TXT files.

Supported Formats:
- PDF: Uses PyPDF2 to extract text from each page
- DOCX: Uses python-docx to read paragraphs from Word documents
- TXT: Direct file reading with encoding detection
"""

import os
from PyPDF2 import PdfReader
from docx import Document


class DocumentParser:
    """Parses documents and extracts plain text."""

    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

    @staticmethod
    def extract_text(file_path):
        """
        Extract text from a document file.

        Args:
            file_path: Path to the document file

        Returns:
            str: Extracted text content

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext not in DocumentParser.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}. Supported: {DocumentParser.SUPPORTED_EXTENSIONS}")

        if ext == '.pdf':
            return DocumentParser._extract_from_pdf(file_path)
        elif ext == '.docx':
            return DocumentParser._extract_from_docx(file_path)
        elif ext == '.txt':
            return DocumentParser._extract_from_txt(file_path)

    @staticmethod
    def _extract_from_pdf(file_path):
        """Extract text from PDF using PyPDF2."""
        text_parts = []
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return '\n'.join(text_parts)

    @staticmethod
    def _extract_from_docx(file_path):
        """Extract text from DOCX using python-docx."""
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n'.join(paragraphs)

    @staticmethod
    def _extract_from_txt(file_path):
        """Extract text from TXT file with encoding fallback."""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError(f"Could not decode file: {file_path}")

    @staticmethod
    def get_file_info(file_path):
        """Get metadata about a document file."""
        stat = os.stat(file_path)
        return {
            'name': os.path.basename(file_path),
            'size': stat.st_size,
            'extension': os.path.splitext(file_path)[1].lower(),
            'size_readable': f"{stat.st_size / 1024:.1f} KB"
        }
