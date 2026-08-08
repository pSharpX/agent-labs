import uuid

from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings

from settings import DatabaseSettings
from vector_store import get_pg_vector

db_settings = DatabaseSettings()

# Loaders
text_loader = TextLoader("./docs/what_is_llm.txt")
text_docs = text_loader.load()

md_loader = TextLoader("./docs/langchain.md")
md_docs = md_loader.load()

web_loader = WebBaseLoader("https://docs.langchain.com/oss/python/langchain/agents")
web_docs = web_loader.load()

pdf_loader = PyPDFLoader("./docs/NaturalLanguageProcessing.pdf")
pdf_docs = pdf_loader.load()

# Splitters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_docs_chunks = text_splitter.split_documents(text_docs)

md_splitter = RecursiveCharacterTextSplitter.from_language(chunk_size=1000, chunk_overlap=200, language=Language.MARKDOWN)
md_docs_chunks = md_splitter.split_documents(md_docs)

#print(text_docs_chunks)

# Embeddings
# model="text-embedding-3-small
# model="text-embedding-3-large
embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents([
    chunk.page_content for chunk in text_docs_chunks
])

#print(embeddings)

# Init vector store
vector_store = get_pg_vector(embeddings_model, db_settings.url)

# Insert documents
ids = [str(uuid.uuid4()) for _ in text_docs_chunks]
#[
#   Document(
#       metadata={"source": "./docs/what_is_llm.txt"},
#       page_content="To compute attention, each embedding is projected into three distinct vectors using learned weight matrices: a query, a key, and a value. The query represents what a given token is “seeking,” the key represents the information that each token contains, and the value “returns” the information from each key vector, scaled by its respective attention weight.\n\nAlignment scores are then computed as the similarity between queries and keys. These scores, once normalized into attention weights, determine how much of each value vector flows into the representation of the current token. This process allows the model to flexibly focus on relevant context while ignoring less important tokens (like “tree”).",
#   )
# ]
#vector_store.add_documents(documents = text_docs_chunks, ids=ids)

# Search
results = vector_store.similarity_search(query="what is llm?", k=4)
print(results)

