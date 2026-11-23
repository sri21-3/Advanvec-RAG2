from pathlib import Path

from src.config import get_embeddings, get_settings, get_supabase_client
from src.loaders import load_and_chunk_documents


def run_ingestion(clear_existing: bool = True) -> None:
    settings = get_settings()
    data_dir = Path("data")

    if not data_dir.exists():
        raise FileNotFoundError("data/ directory not found.")

    chunks = load_and_chunk_documents(data_dir=data_dir, settings=settings)
    if not chunks:
        print("No PDF or DOCX content found inside data/.")
        return

    print(f"Prepared {len(chunks)} chunks (text, tables, and image descriptions).")

    supabase = get_supabase_client(settings)
    embeddings = get_embeddings(settings)

    if clear_existing:
        supabase.table(settings.supabase_table).delete().neq("id", 0).execute()
        print(f"Cleared existing rows in '{settings.supabase_table}'.")

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(texts)

        rows = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": vec,
            }
            for doc, vec in zip(batch, vectors)
        ]

        supabase.table(settings.supabase_table).insert(rows).execute()
        print(f"Uploaded chunks {i + 1} to {min(i + batch_size, len(chunks))}...")

    print("Ingestion complete!")


if __name__ == "__main__":
    run_ingestion(clear_existing=True)