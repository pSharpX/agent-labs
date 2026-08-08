from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

collection_name = "docs"

def initialize_pg_vector_and_docs_(documents: list[Document], embeddings_model, connection: str) -> PGVector:
    return PGVector.from_documents(documents, embeddings_model, connection=connection, collection_name=collection_name)

def get_pg_vector(embeddings_model, connection: str) -> PGVector:
    return PGVector(
        embeddings=embeddings_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

def get_record_manager():
    pass