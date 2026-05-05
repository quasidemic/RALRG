from PyPDF2 import PdfReader
import pysbd

# ----------------------------- text utilities ----------------------------- #
def clean_pdf_text(text: str) -> str:
    # Remove hyphenation at line breaks and normalize whitespace
    text = text.replace("-\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
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