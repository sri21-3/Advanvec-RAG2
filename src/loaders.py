import base64
from pathlib import Path
from typing import List, Optional

import pymupdf  # PyMuPDF
import pdfplumber
from docx import Document as DocxDocument
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings


def _rows_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """Convert a 2D list of cells into a clean Markdown table."""
    cleaned = [
        [(cell or "").strip().replace("\n", " ") for cell in row]
        for row in rows
        if any((cell or "").strip() for cell in row)
    ]
    if not cleaned:
        return ""

    col_count = max(len(r) for r in cleaned)
    padded = [r + [""] * (col_count - len(r)) for r in cleaned]

    header = "| " + " | ".join(padded[0]) + " |"
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    body = ["| " + " | ".join(row) + " |" for row in padded[1:]]
    return "\n".join([header, separator, *body])


def _describe_image_with_groq(
    image_path: Path,
    page_context: str,
    settings: Settings,
) -> str:
    """Use a Groq VLM (if configured) to convert diagrams/workflows/images into searchable text."""
    if not settings.vision_model:
        return (
            f"[Embedded Image: {image_path.name}]\n"
            f"Surrounding page context:\n{page_context[:800]}"
        )

    try:
        client = Groq(api_key=settings.groq_api_key)
        b64_img = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        ext = image_path.suffix.lstrip(".").lower() or "png"
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"

        prompt = (
            "You are indexing an image extracted from a technical/business document for a RAG system. "
            "Describe everything visible in detail: if it is a workflow, architecture diagram, or flowchart, "
            "list every node, arrow direction, decision branch, and label step-by-step. "
            "If it is a chart or screenshot, transcribe all text, axes, numbers, and takeaways.\n\n"
            f"Surrounding page text for context:\n{page_context[:600]}"
        )

        resp = client.chat.completions.create(
            model=settings.vision_model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64_img}"},
                        },
                    ],
                }
            ],
        )
        description = (resp.choices[0].message.content or "").strip()
        return f"[Image / Diagram Summary - {image_path.name}]\n{description}"
    except Exception as exc:
        print(f"  [Warn] Vision captioning failed for {image_path.name}: {exc}")
        return (
            f"[Embedded Image: {image_path.name}]\n"
            f"Surrounding page context:\n{page_context[:800]}"
        )


def load_docx_file(file_path: Path) -> List[Document]:
    """Extract paragraphs and tables from a .docx file."""
    doc = DocxDocument(str(file_path))
    docs: List[Document] = []

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    if paragraphs:
        docs.append(
            Document(
                page_content="\n\n".join(paragraphs),
                metadata={
                    "source": file_path.name,
                    "file_type": "docx",
                    "content_type": "text",
                },
            )
        )

    for idx, table in enumerate(doc.tables, start=1):
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        md_table = _rows_to_markdown(rows)
        if md_table:
            docs.append(
                Document(
                    page_content=f"[DOCX Table {idx} from {file_path.name}]\n{md_table}",
                    metadata={
                        "source": file_path.name,
                        "file_type": "docx",
                        "content_type": "table",
                        "table_index": idx,
                    },
                )
            )

    return docs


def load_pdf_file(
    file_path: Path,
    images_output_dir: Path,
    settings: Settings,
) -> List[Document]:
    """Extract text, Markdown tables, and summarized images from a PDF."""
    docs: List[Document] = []
    images_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract tables cleanly with pdfplumber
    page_texts: dict[int, str] = {}
    with pdfplumber.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            page_texts[page_num] = text
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "file_type": "pdf",
                            "content_type": "text",
                            "page": page_num,
                        },
                    )
                )

            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables, start=1):
                md_table = _rows_to_markdown(table)
                if md_table:
                    docs.append(
                        Document(
                            page_content=(
                                f"[PDF Table {t_idx} | {file_path.name} | Page {page_num}]\n"
                                f"{md_table}"
                            ),
                            metadata={
                                "source": file_path.name,
                                "file_type": "pdf",
                                "content_type": "table",
                                "page": page_num,
                                "table_index": t_idx,
                            },
                        )
                    )

    # 2. Extract embedded images/diagrams with PyMuPDF
    pdf_doc = pymupdf.open(str(file_path))
    seen_xrefs: set[int] = set()

    for page_idx in range(len(pdf_doc)):
        page_num = page_idx + 1
        page = pdf_doc[page_idx]
        for img_idx, img_info in enumerate(page.get_images(full=True), start=1):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_image = pdf_doc.extract_image(xref)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Skip tiny icons, bullets, or decorative lines
            if width < 100 or height < 100:
                continue

            ext = base_image.get("ext", "png")
            img_filename = f"{file_path.stem}_p{page_num}_img{img_idx}.{ext}"
            img_path = images_output_dir / img_filename
            img_path.write_bytes(base_image["image"])

            summary = _describe_image_with_groq(
                image_path=img_path,
                page_context=page_texts.get(page_num, ""),
                settings=settings,
            )

            docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        "source": file_path.name,
                        "file_type": "pdf",
                        "content_type": "image",
                        "page": page_num,
                        "image_path": str(img_path).replace("\\", "/"),
                    },
                )
            )

    pdf_doc.close()
    return docs


def load_and_chunk_documents(
    data_dir: Path,
    settings: Settings,
) -> List[Document]:
    """Load all PDFs and DOCX files in data_dir and chunk prose while preserving tables/image summaries."""
    raw_docs: List[Document] = []
    images_dir = data_dir / "extracted_images"

    for file_path in sorted(data_dir.iterdir()):
        if file_path.is_dir():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            print(f"Loading PDF: {file_path.name}")
            raw_docs.extend(load_pdf_file(file_path, images_dir, settings))
        elif suffix == ".docx":
            print(f"Loading DOCX: {file_path.name}")
            raw_docs.extend(load_docx_file(file_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks: List[Document] = []
    for doc in raw_docs:
        # Keep tables and image summaries intact unless unusually huge
        if doc.metadata.get("content_type") in ("table", "image") and len(doc.page_content) <= 2500:
            final_chunks.append(doc)
        else:
            final_chunks.extend(splitter.split_documents([doc]))

    return final_chunks