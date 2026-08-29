import pickle
import uuid
from pathlib import Path
from rich import print

import anydoc
from langchain_community.docstore import InMemoryDocstore

import faiss

from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import DatabaseSettings, BaseModelSettings, BaseToolSettings, PDFLoader
from helpers import classify_resource

collection_name = "agents"
db_settings = DatabaseSettings()
model_settings = BaseModelSettings()
tools_settings = BaseToolSettings()
pdf_loader = PDFLoader(tools_settings.pdf_loader)

def call_anydoc(resource: str, resource_type: str) -> list[Document]:
    content = anydoc.to_markdown(resource)
    return [Document(
        page_content=content,
        metadata={
            "source": resource,
            "type": resource_type,
        }
    )]

def call_pdf_loader(resource: str) -> list[Document]:
    if pdf_loader is PDFLoader.FIRECRAWL_ANYDOC:
        return call_anydoc(resource, "pdf")
    elif pdf_loader is PDFLoader.DOC7:
        raise ValueError("Provider not implemented")
    loader = PyPDFLoader(resource)
    return loader.load()


class IndexingStageRAG:
    def __init__(self, index = None, docstore = InMemoryDocstore(), index_to_docstore_id = dict()):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Embeddings
        # model="text-embedding-3-small", "text-embedding-3-large"
        self.embeddings_model = OpenAIEmbeddings()
        # Determine the embedding dimension
        self.embedding_dimension = len(
            self.embeddings_model.embed_query("test")
        )
        # Create an empty FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension) if index is None else index
        # Init vector store
        self.vector_store = FAISS(
            embedding_function=self.embeddings_model,
            index=self.faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

    @classmethod
    def from_local(cls, index_name: str, folder_path: str = "faiss"):
        path = Path(folder_path)
        index = faiss.read_index(str(path / f"{index_name}.faiss"))
        # load docstore and index_to_docstore_id
        with open(path / f"{index_name}.pkl", "rb") as f:
            (
                docstore,
                index_to_docstore_id,
            ) = pickle.load(f)

        return cls(index, docstore, index_to_docstore_id)

    @staticmethod
    def __load(resource: str) -> list[Document]:
        resource_type: str = classify_resource(resource)
        if resource_type == "web":
            loader = WebBaseLoader(resource)
            return loader.load()
        elif resource_type in ["doc", "docx", "xls", "xlsx"]:
            return call_anydoc(resource, resource_type)
        elif resource_type == "pdf":
            return call_pdf_loader(resource)
        loader = TextLoader(resource)
        return loader.load()

    def __chunk(self, docs: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(docs)

    def __embed(self, chunks: list[Document]):
        embeddings = self.embeddings_model.embed_documents([
            chunk.page_content for chunk in chunks
        ])
        return embeddings

    def save_local(self, index_name: str):
        self.vector_store.save_local(folder_path="faiss", index_name=index_name)

    def process(self, resource: str):
        docs = IndexingStageRAG.__load(resource)
        chunks = self.__chunk(docs)
        #embeddings = self.__embed(chunks)

        # Insert documents
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.vector_store.add_documents(documents=chunks, ids=ids)
        print("Indexing task completed")


class RetrievalStageRAG:
    def __init__(self, index_name: str, folder_path: str = "faiss"):
        # Embeddings
        # model="text-embedding-3-small", "text-embedding-3-large"
        self.embeddings_model = OpenAIEmbeddings()
        path = Path(folder_path)
        index = faiss.read_index(str(path / f"{index_name}.faiss"))
        # load docstore and index_to_docstore_id
        with open(path / f"{index_name}.pkl", "rb") as f:
            (
                docstore,
                index_to_docstore_id,
            ) = pickle.load(f)

        # Init vector store
        self.vector_store = FAISS(
            embedding_function=self.embeddings_model,
            index=index,
            index_to_docstore_id=index_to_docstore_id,
            docstore=docstore
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(input=query)


#indexing_stage_rag = IndexingStageRAG()
retrieval_stage_rag = RetrievalStageRAG("temp_store")

def display_results(results: list[Document]):
    for index, doc in enumerate(results):
        print(f"{index}. {doc.page_content}")


def main():
    print("Welcome to RAG!")
    print("Start searching text (type 'c' for exit) >> ")
    while True:
        question = input()
        if question == "c":
            break
        elif question.strip() == "":
            continue
        results = retrieval_stage_rag.retrieve(question)
        display_results(results)


if __name__ == '__main__':
    #indexing_stage_rag.process("./docs/agent/Catalogo_Productos_Servicios_ZEIT_2026.docx")
    #indexing_stage_rag.save_local("temp_store")
    main()


