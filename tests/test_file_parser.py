"""Tests for file parsing utilities."""

from pathlib import Path

import pytest

from brainstormer.utils.file_parser import parse_file, parse_files, parse_text


class TestParseText:
    """Tests for text file parsing."""

    def test_parse_text_utf8(self, temp_dir):
        """Test parsing UTF-8 text file."""
        file_path = temp_dir / "test.txt"
        content = "Hello, World!\nLine 2"
        file_path.write_text(content, encoding="utf-8")

        result = parse_text(file_path)
        assert result == content

    def test_parse_text_with_unicode(self, temp_dir):
        """Test parsing text with unicode characters."""
        file_path = temp_dir / "unicode.txt"
        content = "Hello 世界! 🌍"
        file_path.write_text(content, encoding="utf-8")

        result = parse_text(file_path)
        assert result == content


class TestParseFile:
    """Tests for generic file parsing."""

    def test_parse_text_file(self, sample_text_file):
        """Test parsing a text file."""
        result = parse_file(sample_text_file)

        assert result["name"] == "sample.txt"
        assert result["type"] == "text"
        assert "sample text content" in result["content"]
        assert result["size"] > 0

    def test_parse_markdown_file(self, temp_dir):
        """Test parsing markdown file."""
        file_path = temp_dir / "readme.md"
        file_path.write_text("# Title\n\nContent here.")

        result = parse_file(file_path)
        assert result["type"] == "text"
        assert "# Title" in result["content"]

    def test_parse_nonexistent_file(self, temp_dir):
        """Test parsing a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parse_file(temp_dir / "nonexistent.txt")

    def test_parse_file_metadata(self, sample_text_file):
        """Test file metadata extraction."""
        result = parse_file(sample_text_file)

        assert "path" in result
        assert Path(result["path"]).exists()
        assert isinstance(result["size"], int)
        assert result["size"] > 0


class TestParseFiles:
    """Tests for batch file parsing."""

    def test_parse_multiple_files(self, temp_dir):
        """Test parsing multiple files."""
        files = []
        for i in range(3):
            file_path = temp_dir / f"file{i}.txt"
            file_path.write_text(f"Content {i}")
            files.append(file_path)

        results = parse_files(files)

        assert len(results) == 3
        assert all(r["type"] == "text" for r in results)

    def test_parse_files_with_invalid(self, temp_dir):
        """Test parsing with some invalid files."""
        valid = temp_dir / "valid.txt"
        valid.write_text("valid content")

        invalid = temp_dir / "nonexistent.txt"

        results = parse_files([valid, invalid])

        # Should only return valid file
        assert len(results) == 1
        assert results[0]["name"] == "valid.txt"
