import weaviate
import chromadb
import uuid
import anydoc

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres.vectorstores import PGVector
from langchain_chroma.vectorstores import Chroma
from langchain_redis.vectorstores import RedisVectorStore, RedisConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_core.documents import Document

from settings import VectorStoreProvider, StoreSettings, BaseModelSettings
from utils import classify_resource


class VectorStoreResolver:
    def __init__(self, config: StoreSettings):
        self.settings = config

    def resolve(self, provider: VectorStoreProvider, embeddings_model, collection_name: str) -> VectorStore:
        if provider is VectorStoreProvider.IN_MEMORY:
            return InMemoryVectorStore(embedding=embeddings_model)
        elif provider is VectorStoreProvider.CHROMA:
            client = chromadb.HttpClient(host=self.settings.host, port=self.settings.port)
            return Chroma(
                collection_name=collection_name,
                embedding_function=embeddings_model,
                client=client,
            )
        elif provider is VectorStoreProvider.REDIS:
            config = RedisConfig(
                index_name="self.settings.index_name",
                redis_url=self.settings.url,
                distance_metric="COSINE"
            )
            return RedisVectorStore(
                embeddings=embeddings_model,
                config=config
            )
        elif provider is VectorStoreProvider.ELASTIC_SEARCH:
            return ElasticsearchStore(
                index_name="self.settings.index_name",
                embedding=embeddings_model,
                es_url=self.settings.url,
            )
        elif provider is VectorStoreProvider.WEAVIATE:
            weaviate_client = weaviate.connect_to_local()
            return WeaviateVectorStore(
                client=weaviate_client,
                index_name="self.settings.index_name",
                text_key="self.settings.text_key",
                embedding=embeddings_model
            )
        elif provider is VectorStoreProvider.COSMOSDB:
            raise NotImplementedError("Provider not implemented")
        elif provider is VectorStoreProvider.MONGODB:
            raise NotImplementedError("Provider not implemented")
        elif provider is VectorStoreProvider.COSMOSDB:
            raise NotImplementedError("Provider not implemented")

        return PGVector(
            embeddings=embeddings_model,
            collection_name=collection_name,
            connection=self.settings.url,
            use_jsonb=True,
        )

class DocumentRepository:
    def __init__(self, store: VectorStore):
        self.store = store
        self.retriever = self.store.as_retriever()

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids = [str(uuid.uuid4()) for _ in documents]
        return self.store.add_documents(documents=documents, ids=ids)

    def search_documents(self, query: str) -> list[Document]:
        return self.retriever.invoke(input=query)


class IndexingStageRAG:
    def __init__(self, repository: DocumentRepository):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Init document repository
        self.document_repository = repository

    @staticmethod
    def __load(resource) -> list[Document]:
        resource_type: str = classify_resource(resource)
        if resource_type == "web":
            loader = WebBaseLoader(resource)
            return loader.load()
        elif resource_type == "pdf":
            content = anydoc.to_markdown(resource)
            return [Document(
                        page_content=content,
                        metadata={
                            "source": resource,
                            "type": resource_type,
                        }
                    )]
        loader = TextLoader(resource)
        return loader.load()

    def __chunk(self, docs: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(docs)

    def process(self, resource):
        docs = IndexingStageRAG.__load(resource)
        chunks = self.__chunk(docs)

        # Insert documents
        self.document_repository.add_documents(documents=chunks)
        print("Indexing task completed")

# Splitters
#md_splitter = RecursiveCharacterTextSplitter.from_language(chunk_size=1000, chunk_overlap=200, language=Language.MARKDOWN)
#md_docs_chunks = md_splitter.split_documents(md_docs)

class RAGPoweredChatOpenAI:
    def __init__(self, settings: BaseModelSettings, repository: DocumentRepository):
        self.document_repository = repository
        self.model_settings = settings
        self.model = ChatOpenAI(model=self.model_settings.model_name, temperature=self.model_settings.temperature, verbose=True)

        self.prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
        {context}

        Question: {question}
        """)

    def __ask(self, question: str):
        docs = self.document_repository.search_documents(query=question)
        prompt = self.prompt.invoke({"context": docs, "question": question})
        answer = self.model.invoke(prompt)
        print(prompt)
        return answer

    def initialize(self):
        print("Ask your question: ")
        question = input()
        answer = self.__ask(question)
        print(answer)


# Embeddings
# model="text-embedding-3-small", "text-embedding-3-large"
embeddings_model = OpenAIEmbeddings()

collection_name = "natural_language_processing"
store_settings = StoreSettings()
model_settings = BaseModelSettings()

vector_store_resolver = VectorStoreResolver(store_settings)
vector_store = vector_store_resolver.resolve(
    provider=VectorStoreProvider(store_settings.provider),
    embeddings_model=embeddings_model,
    collection_name=collection_name
)

document_repository = DocumentRepository(vector_store)

indexing_stage_rag = IndexingStageRAG(repository=document_repository)
main_chat = RAGPoweredChatOpenAI(repository=document_repository, settings=model_settings)

if __name__ == '__main__':
    main_chat.initialize()
    #indexing_stage_rag.process("./docs/NaturalLanguageProcessing.pdf")