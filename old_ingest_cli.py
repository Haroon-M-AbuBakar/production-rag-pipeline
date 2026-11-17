import sys
from chunker import parse_pdf, chunker
from qdrant import ingest_chunks

if __name__ == "__main__":
    pages = parse_pdf(sys.argv[1])
    chunks = chunker(pages)
    print(f"Generated {len(chunks)} chunks")
    ingest_chunks(chunks)