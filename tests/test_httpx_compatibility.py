"""Test httpx compatibility with web scraping libraries."""

import html2text
import httpx
import pytest
from bs4 import BeautifulSoup
from readability import Document


class TestHttpxCompatibility:
    """Test that httpx works with our web scraping stack."""

    @pytest.mark.asyncio
    async def test_httpx_with_beautifulsoup(self):
        """Test httpx response works with BeautifulSoup."""
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Header</h1>
                <p>Test paragraph with <a href="/link">link</a></p>
            </body>
        </html>
        """

        # Create a mock httpx response
        response = httpx.Response(
            200,
            content=html_content.encode("utf-8"),
            headers={"content-type": "text/html; charset=utf-8"},
        )

        # Test BeautifulSoup parsing
        soup = BeautifulSoup(response.text, "html.parser")

        assert soup.title.string == "Test Page"
        assert soup.find("h1").text == "Test Header"
        assert soup.find("a")["href"] == "/link"

    @pytest.mark.asyncio
    async def test_httpx_with_html2text(self):
        """Test httpx response works with html2text."""
        html_content = """
        <html>
            <body>
                <h1>Test Header</h1>
                <p>Test paragraph with <strong>bold</strong> text.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
        </html>
        """

        response = httpx.Response(
            200,
            content=html_content.encode("utf-8"),
            headers={"content-type": "text/html"},
        )

        # Test html2text conversion
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True

        markdown = converter.handle(response.text)

        assert "# Test Header" in markdown
        assert "**bold**" in markdown
        assert "* Item 1" in markdown
        assert "* Item 2" in markdown

    @pytest.mark.asyncio
    async def test_httpx_with_readability(self):
        """Test httpx response works with readability-lxml."""
        html_content = """
        <html>
            <head><title>Article Title</title></head>
            <body>
                <nav>Navigation menu</nav>
                <article>
                    <h1>Main Article Title</h1>
                    <p>This is the main content that should be extracted.</p>
                    <p>Another paragraph of important content.</p>
                </article>
                <footer>Footer content</footer>
            </body>
        </html>
        """

        response = httpx.Response(
            200,
            content=html_content.encode("utf-8"),
            headers={"content-type": "text/html; charset=utf-8"},
        )

        # Test readability extraction
        doc = Document(response.text)

        assert doc.title() == "Article Title"

        # Get main content
        main_text = doc.summary()
        assert "Main Article Title" in main_text
        assert "main content that should be extracted" in main_text
        # Note: readability-lxml's behavior may vary, so we're testing core functionality
        # The important thing is that it processes the content without errors

    @pytest.mark.asyncio
    async def test_httpx_encoding_handling(self):
        """Test httpx handles different encodings correctly."""
        # Test UTF-8
        utf8_content = "<html><body><p>UTF-8 content: café</p></body></html>"
        response_utf8 = httpx.Response(
            200,
            content=utf8_content.encode("utf-8"),
            headers={"content-type": "text/html; charset=utf-8"},
        )

        assert "café" in response_utf8.text

        # Test with BeautifulSoup
        soup = BeautifulSoup(response_utf8.text, "html.parser")
        assert soup.find("p").text == "UTF-8 content: café"

    @pytest.mark.asyncio
    async def test_httpx_binary_content(self):
        """Test httpx handles binary content."""
        binary_data = b"\x00\x01\x02\x03\x04\x05"

        response = httpx.Response(
            200,
            content=binary_data,
            headers={"content-type": "application/octet-stream"},
        )

        # Binary content should be accessible
        assert response.content == binary_data

        # Text decoding should handle errors gracefully
        text = response.text  # Should use 'replace' error handling by default
        assert isinstance(text, str)

    def test_httpx_response_properties(self):
        """Test httpx response properties match our needs."""
        response = httpx.Response(
            200,
            content=b"Test content",
            headers={"content-type": "text/plain", "content-length": "12"},
        )

        # Properties we use in web_client.py
        assert response.status_code == 200
        assert response.content == b"Test content"
        assert response.text == "Test content"
        assert response.headers.get("content-type") == "text/plain"
        assert response.encoding is not None  # Should auto-detect


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
