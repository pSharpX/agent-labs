import uuid
from pathlib import Path
from urllib.parse import urlparse

from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader, PDFPlumberLoader, UnstructuredPDFLoader, AmazonTextractPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import DatabaseSettings, BaseModelSettings

collection_name = "agents"
db_settings = DatabaseSettings()
model_settings = BaseModelSettings()

class IndexingStageRAG:
    def __init__(self, settings: DatabaseSettings):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Embeddings
        # model="text-embedding-3-small", "text-embedding-3-large"
        self.embeddings_model = OpenAIEmbeddings()
        # Init vector store
        self.vector_store = PGVector(
            embeddings=self.embeddings_model,
            collection_name=collection_name,
            connection=settings.url,
            use_jsonb=True,
        )

    @staticmethod
    def __classify_resource(resource: str) -> str:
        """
        Classify a resource as 'web', 'pdf', or 'text'.

        Args:
            resource: URL or local file path.

        Returns:
            One of: 'url', 'pdf', 'text'

        Raises:
            ValueError: If the resource type cannot be determined.
        """
        resource = resource.strip()

        # Check if it is a URL
        parsed = urlparse(resource)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return "web"

        # Check local file extension
        path = Path(resource)
        extension = path.suffix.lower()

        if extension == ".pdf":
            return "pdf"

        # Common text-based files
        text_extensions = {
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".xml",
            ".html",
            ".htm",
            ".yaml",
            ".yml",
        }

        if extension in text_extensions:
            return "text"

        raise ValueError(f"Unsupported resource type: {resource}")

    @staticmethod
    def __load(resource) -> list[Document]:
        resource_type: str = IndexingStageRAG.__classify_resource(resource)
        if resource_type == "web":
            loader = WebBaseLoader(resource)
            return loader.load()
        elif resource_type == "pdf":
            loader = PyPDFLoader(resource)
            return loader.load()
        loader = TextLoader(resource)
        return loader.load()

    def __chunk(self, docs: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(docs)

    def __embed(self, chunks: list[Document]):
        embeddings = self.embeddings_model.embed_documents([
            chunk.page_content for chunk in chunks
        ])
        return embeddings

    def process(self, resource):
        docs = IndexingStageRAG.__load(resource)
        chunks = self.__chunk(docs)
        #embeddings = self.__embed(chunks)

        # Insert documents
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.vector_store.add_documents(documents=chunks, ids=ids)
        print("Indexing task completed")

# Splitters
#md_splitter = RecursiveCharacterTextSplitter.from_language(chunk_size=1000, chunk_overlap=200, language=Language.MARKDOWN)
#md_docs_chunks = md_splitter.split_documents(md_docs)

# Search
#results = vector_store.similarity_search(query="what is llm?", k=4)
#print(results)

# "https://docs.langchain.com/oss/python/langchain/agents"
# ./docs/NaturalLanguageProcessing.pdf
# ./docs/langchain.md"
# "./docs/what_is_llm.txt"


class RetrievalStageRAG:
    def __init__(self, settings: DatabaseSettings):
        # Embeddings
        # model="text-embedding-3-small", "text-embedding-3-large"
        self.embeddings_model = OpenAIEmbeddings()
        # Init vector store
        self.vector_store = PGVector(
            embeddings=self.embeddings_model,
            collection_name=collection_name,
            connection=settings.url,
            use_jsonb=True,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})

    def retrieve(self, query: str):
        return self.retriever.invoke(input=query)


class RAGPoweredChatOpenAI:
    def __init__(self, settings: BaseModelSettings):
        self.rag = RetrievalStageRAG(db_settings)
        self.model_settings = settings
        self.model = ChatOpenAI(model=self.model_settings.model_name, temperature=self.model_settings.temperature, verbose=True)

        self.prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
        {context}

        Question: {question}
        """)

    def __ask(self, question: str):
        docs = self.rag.retrieve(question)
        prompt = self.prompt.invoke({"context": docs, "question": question})
        answer = self.model.invoke(prompt)
        return answer

    def initialize(self):
        print("Ask your question: ")
        question = input()
        answer = self.__ask(question)
        print(answer)

indexing_stage_rag = IndexingStageRAG(settings=db_settings)
main_chat = RAGPoweredChatOpenAI(settings=model_settings)

if __name__ == '__main__':
    main_chat.initialize()
    #indexing_stage_rag.process("./docs/what_is_llm.txt")
    #indexing_stage_rag.process("./docs/NaturalLanguageProcessing.pdf")
    #indexing_stage_rag.process("https://docs.langchain.com/oss/python/langchain/agents")

