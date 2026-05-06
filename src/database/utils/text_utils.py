import re

from PyPDF2 import PdfReader
import pysbd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------- text utilities ----------------------------- #
WORD_PATTERN = re.compile(r"\S+")
LANGCHAIN_WORD_SEPARATORS = [
    r"\n\s*\n+",
    r"\n+",
    r"(?<=[.!?])\s+",
    r"(?<=[;:])\s+",
    r"(?<=,)\s+",
    r"\s+",
    "",
]


def count_words(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def clean_pdf_text(text: str) -> str:
    # Remove hyphenation at line breaks and normalize whitespace
    text = text.replace("-\n", "")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    raw_text = "\n".join(texts)
    return clean_pdf_text(raw_text)


def chunk_text_with_pysbd(
    text: str,
    min_chars: int = 500,
    language: str = "en",
) -> list:
    segmenter = pysbd.Segmenter(language=language, clean=True)
    sentences = segmenter.segment(text)

    chunks = []
    current = " "

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        candidate = (current + " " + sent) if current else sent

        if len(candidate) < min_chars:
            current = candidate
        else:
            chunks.append(candidate.strip())
            current = " "

    if current:
        chunks.append(current.strip())

    return chunks


def chunk_text_with_langchain(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 60,
    length_function=count_words,
    ) -> list:
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=LANGCHAIN_WORD_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=True,
    )
    return text_splitter.split_text(text)
