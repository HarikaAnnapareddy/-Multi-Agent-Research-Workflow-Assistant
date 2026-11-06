import hashlib
import io

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.schemas import Chunk
from app.utils import chunk_text


def _to_chunks(text: str, source: str) -> list[Chunk]:
    return [
        Chunk(id=hashlib.md5(f"{source}:{i}:{c[:50]}".encode()).hexdigest(), text=c, source=source)
        for i, c in enumerate(chunk_text(text))
    ]


def load_pdf(data: bytes, name: str) -> list[Chunk]:
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return _to_chunks(text, name)


def load_url(url: str) -> list[Chunk]:
    resp = requests.get(url, timeout=20, headers={"User-Agent": "research-assistant/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return _to_chunks(soup.get_text(" "), url)
