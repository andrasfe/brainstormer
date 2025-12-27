"""File parsing utilities for text and PDF files."""

from pathlib import Path

from pypdf import PdfReader

from .logging import get_logger

logger = get_logger(__name__)


def parse_text(file_path: Path) -> str:
    """Parse a text file and return its contents."""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        return file_path.read_text(encoding="latin-1")


def parse_pdf(file_path: Path) -> str:
    """Parse a PDF file and extract text content."""
    reader = PdfReader(file_path)
    text_parts = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            text_parts.append(f"--- Page {i + 1} ---\n{text}")

    return "\n\n".join(text_parts)


def parse_file(file_path: Path) -> dict:
    """
    Parse a file and return its contents with metadata.

    Returns:
        dict with keys: 'path', 'name', 'type', 'content', 'size'
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        content = parse_pdf(file_path)
        file_type = "pdf"
    elif suffix in {".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".xml", ".csv"}:
        content = parse_text(file_path)
        file_type = "text"
    else:
        # Try to read as text
        try:
            content = parse_text(file_path)
            file_type = "text"
        except Exception as e:
            logger.warning(f"Could not parse {file_path}: {e}")
            raise ValueError(f"Unsupported file type: {suffix}") from e

    return {
        "path": str(file_path.absolute()),
        "name": file_path.name,
        "type": file_type,
        "content": content,
        "size": file_path.stat().st_size,
    }


def parse_files(file_paths: list[Path]) -> list[dict]:
    """Parse multiple files and return their contents."""
    results = []
    for path in file_paths:
        try:
            results.append(parse_file(path))
        except Exception as e:
            logger.error(f"Failed to parse {path}: {e}")
    return results
