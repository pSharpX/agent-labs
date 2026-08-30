import anydoc
from langchain_community.docstore import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate

import faiss
import numpy as np
import pickle
import umap
import uuid

from pathlib import Path
from rich import print

from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.mixture import GaussianMixture
from sentence_transformers import SentenceTransformer

from settings import BaseModelSettings


class RaptorNode:
    """
    Represent a node in the RAPTOR hierarchical tree.

    A node contains its text, embedding, layer, and references to child nodes.
    """
    def __init__(self, text, embedding, layer, children=None):
        """
        Initialize a RAPTOR tree node.

        Args:
            text (str): Text or summary represented by the node.
            embedding: Vector embedding of the node's text.
            layer (int): Hierarchical layer of the node.
            children (list, optional): Child nodes represented by this node.
        """
        self.text = text
        self.embedding = embedding
        self.layer = layer
        self.children = children or []


class IndexingStageRAG:
    def __init__(self, chunk_size = 1000, chunk_overlap = 200, n_components = 10):
        # n_components: Number of dimensions in the reduced output.
        self.reducer = umap.UMAP(n_components=n_components, metric="cosine")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # Embeddings
        self.show_progress_bar = True
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # Determine the embedding dimension
        # Convert to a 2D float32 NumPy array (CRITICAL STEP)
        embeddings = self.embedder.encode("test", show_progress_bar=True)
        embeddings_np = np.array([embeddings]).astype('float32')
        # Initialize the FAISS index (Match the dimension of your vectors)
        self.embedding_dimension = embeddings_np.shape[1]
        # Create an empty FAISS index
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dimension)
        self.model_settings = BaseModelSettings()
        self.summarizer_model = init_chat_model(
            model=self.model_settings.model_name,
            model_provider=self.model_settings.provider,
            temperature=self.model_settings.temperature,
            max_tokens=500,
        )
        self.docstore = InMemoryDocstore()
        self.index_to_docstore_id = {}

    def __summarize_cluster(self, texts: list[str]) -> str:
        """
        Generate a dense, information-preserving summary of a text cluster.

        Combines the cluster's texts and uses the configured chat model to retain
        key facts, numbers, and entities in a concise paragraph.

        Args:
            texts (list[str]): Texts belonging to the cluster.

        Returns:
            str: A dense summary of the cluster's content.
        """
        combined = "\n\n".join(texts)
        message = f"""
        Summarize the following text into a dense, information-preserving paragraph.
        Retain key facts, numbers, and entities:\n\n{combined}
        """
        response = self.summarizer_model.invoke([HumanMessage(content=message)])
        return response.content

    @staticmethod
    def __load_resource(resource: str) -> list[Document]:
        content = anydoc.to_markdown(resource)
        return [Document(
            page_content=content,
            metadata={
                "source": resource,
                "type": "pdf",
            }
        )]

    def __chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split a collection of documents into overlapping text chunks for RAPTOR-based RAG.

        This function uses LangChain's ``RecursiveCharacterTextSplitter`` to
        recursively divide documents while attempting to preserve natural text
        boundaries such as paragraphs, lines, sentences, and words.

        Args:
            documents (Iterable[Document]): Collection of text documents to split into
                smaller chunks.

        Returns:
            list[Document]: A flat list containing all generated document chunks from the
            provided documents.
        """
        return self.text_splitter.split_documents(documents)

    def __embed_documents(self, chunks: list[Document]):
        """
        Generate dense vector embeddings for a collection of texts documents.

        Uses the ``all-MiniLM-L6-v2`` SentenceTransformer model to convert texts
        into semantic embeddings. In a RAPTOR-based RAG pipeline, these embeddings
        can be used for clustering, similarity search, and retrieval.

        Args:
            chunks (list[Document]): Documents to convert into embeddings.

        Returns:
            numpy.ndarray: Embedding vectors for the input texts.
        """
        embeddings = self.embedder.encode([
            chunk.page_content for chunk in chunks
        ], show_progress_bar=self.show_progress_bar)
        return embeddings

    def __reduce_dimensions(self, embeddings):
        """
        Reduce embedding dimensionality using UMAP.

        Uses cosine distance to preserve semantic relationships while producing
        lower-dimensional vectors for efficient clustering.

        Args:
            embeddings: High-dimensional embedding vectors.

        Returns:
            numpy.ndarray: Reduced-dimensional embeddings.
        """
        return self.reducer.fit_transform(embeddings)

    def __cluster_embeddings(self, embeddings, max_clusters=10, threshold=0.1):
        """
        Cluster embeddings using a Gaussian Mixture Model (GMM).

        The optimal number of clusters is selected using the lowest Bayesian
        Information Criterion (BIC). Soft cluster assignment allows an embedding
        to belong to multiple clusters when its membership probability exceeds
        the specified threshold.

        Args:
            embeddings: Embedding vectors to cluster.
            max_clusters (int): Maximum number of clusters to evaluate.
            threshold (float): Minimum membership probability for assigning an
                embedding to a cluster.

        Returns:
            tuple: A tuple containing the cluster labels for each embedding and
            the selected number of clusters.
        """
        reduced = self.__reduce_dimensions(embeddings)

        best_n, best_bic = 1, np.inf
        for n in range(1, min(max_clusters, len(reduced))):
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(reduced)
            bic = gmm.bic(reduced)
            if bic < best_bic:
                best_bic, best_n = bic, n
        gmm = GaussianMixture(n_components=best_n, random_state=42)
        gmm.fit(reduced)
        probs = gmm.predict_proba(reduced)
        # Soft assignment: a point can belong to multiple clusters above threshold
        cluster_labels = [np.where(row > threshold)[0].tolist() for row in probs]
        return cluster_labels, best_n

    def __build_raptor_tree(self, chunks: list[Document], max_layers=4, min_cluster_size=2):
        """
        Build a hierarchical RAPTOR tree from document chunks.

        Embeds the initial chunks, recursively clusters nodes, and creates summary
        nodes for each cluster. Each summary node references its cluster members
        as children, forming a multi-layer hierarchical representation.

        Args:
            chunks (list[Document]): Initial text document chunks used as leaf nodes.
            max_layers (int): Maximum number of hierarchical layers to create.
            min_cluster_size (int): Minimum number of nodes required to continue
                building the next layer.

        Returns:
            list[RaptorNode]: All nodes created across the RAPTOR tree layers.
        """
        layer_0 = [
            RaptorNode(text=c.page_content, embedding=e, layer=0)
            for c, e in zip(chunks, self.__embed_documents(chunks))
        ]

        all_nodes = list(layer_0)
        current_layer_nodes = layer_0

        for layer_num in range(1, max_layers + 1):
            if len(current_layer_nodes) <= min_cluster_size:
                break
            embeddings = np.array([n.embedding for n in current_layer_nodes])
            cluster_labels, n_clusters = self.__cluster_embeddings(embeddings)
            next_layer_nodes = []

            for cluster_id in range(n_clusters):
                members = [
                    n for n, labels in zip(current_layer_nodes, cluster_labels)
                    if cluster_id in labels
                ]
                if not members:
                    continue
                summary_text = self.__summarize_cluster([m.text for m in members])
                summary_embedding = self.__embed_documents([Document(
                    page_content=summary_text,
                )])[0]

                new_node = RaptorNode(
                    text=summary_text,
                    embedding=summary_embedding,
                    layer=layer_num,
                    children=members
                )
                next_layer_nodes.append(new_node)

            all_nodes.extend(next_layer_nodes)
            current_layer_nodes = next_layer_nodes
            if len(next_layer_nodes) <= 1:
                break

        return all_nodes

    def __build_index(self, all_nodes: list[RaptorNode]):
        """
        Build a FAISS vector index from RAPTOR tree nodes.

        Indexes node embeddings for similarity search while keeping the
        corresponding RAPTOR nodes separately for retrieving their text,
        layer, and child relationships.

        Args:
            all_nodes (list[RaptorNode]): Nodes from the RAPTOR tree to index.

        Returns:
            tuple: A FAISS index, docstore, and the list of index_doctore_ids in the same order
            as their vectors in the index.
        """
        embeddings = np.array(
            [node.embedding for node in all_nodes],
            dtype=np.float32
        )

        documents = {}
        for _, node in enumerate(all_nodes):
            doc_id = str(uuid.uuid4())

            documents[doc_id] = Document(
                page_content=node.text,
                metadata={
                    "layer": node.layer,
                },
            )
            self.index_to_docstore_id[doc_id] = doc_id

        self.docstore.add(documents)
        self.faiss_index.add_with_ids(embeddings, self.index_to_docstore_id)
        return  self.faiss_index, self.docstore, self.index_to_docstore_id

    def save_local(self, index_name: str):
        raise NotImplementedError("Method not implemented")

    def run_pipeline(self, resource: str):
        docs = IndexingStageRAG.__load_resource(resource)
        chunks = self.__chunk_documents(docs)
        all_nodes = self.__build_raptor_tree(chunks)

        # Insert documents
        ids = [str(uuid.uuid4()) for _ in all_nodes]
        #self.vector_store.add_documents(documents=chunks, ids=ids)
        self.__build_index(all_nodes)
        print("Indexing task completed")

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        query_embedding = self.embedder.encode([query], show_progress_bar=self.show_progress_bar)[0]
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for score, index_id in zip(scores[0], indices[0]):
            if index_id == -1:
                continue

            doc_id = self.index_to_docstore_id[index_id]
            document = self.docstore.search(doc_id)

            if document is not None:
                document.metadata["score"] = float(score)
                results.append(document)

        return results



indexing_stage_rag = IndexingStageRAG()
#retrieval_stage_rag = RetrievalStageRAG("temp_store")

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
        results = indexing_stage_rag.search(question)
        display_results(results)


if __name__ == '__main__':
    indexing_stage_rag.run_pipeline("./docs/agent/Catalogo_Productos_Servicios_ZEIT_2026.docx")
    #indexing_stage_rag.save_local("temp_store")
    main()