import uuid

from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings


@lru_cache  # load the model once, reuse it afterwards
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # unit-length vectors
    )


DENSE_NAME = "dense"    # name of the dense vector slot
SPARSE_NAME = "sparse"  # name of the sparse (BM25) vector slot


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL)


def create_collection_if_missing(client: QdrantClient):
    name = settings.qdrant_collection
    if client.collection_exists(name):
        return

    # Ask the model for its vector size instead of hard-coding 384
    dim = len(get_embeddings().embed_query("dimension probe"))

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_NAME: models.VectorParams(size=dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    # Payload indexes: make filtering fast. LangChain nests metadata under "metadata."
    indexes = {
        "metadata.doc_id": models.PayloadSchemaType.KEYWORD,
        "metadata.filename": models.PayloadSchemaType.KEYWORD,
        "metadata.page": models.PayloadSchemaType.INTEGER,
    }
    for field, schema in indexes.items():
        client.create_payload_index(name, field_name=field, field_schema=schema)


def get_store(mode: RetrievalMode = RetrievalMode.DENSE) -> QdrantVectorStore:
    # The sparse model is only needed when sparse vectors are written or searched
    sparse = None if mode == RetrievalMode.DENSE else FastEmbedSparse(model_name="Qdrant/bm25")
    return QdrantVectorStore(
        client=get_client(),
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
        sparse_embedding=sparse,
        retrieval_mode=mode,
        vector_name=DENSE_NAME,
        sparse_vector_name=SPARSE_NAME,
    )


def ingest_chunks(chunks: list) -> int:
    """Store chunks (LangChain Documents) from ONE pdf. Returns how many were stored."""
    create_collection_if_missing(get_client())

    ids = []
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_idx"] = i
        # Same doc + same position -> same ID, so re-ingesting overwrites, not duplicates
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.metadata['doc_id']}-{i}")))

    # HYBRID mode writes BOTH the dense and the sparse vector for every chunk
    get_store(RetrievalMode.HYBRID).add_documents(chunks, ids=ids)
    return len(chunks)

    print("chunks ingested into qdrant succesfully")

