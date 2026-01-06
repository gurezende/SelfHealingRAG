import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding 
from fastembed.rerank.cross_encoder import TextCrossEncoder


# Text
text = [
    "In deep learning, the transformer is an artificial neural network architecture based on the multi-head attention mechanism, in which text is converted to numerical representations called tokens, and each token is converted into a vector via lookup from a word embedding table.", 
    "At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism, allowing the signal for key tokens to be amplified and less important tokens to be diminished.",
    "Transformers have the advantage of having no recurrent units, therefore requiring less training time than earlier recurrent neural architectures (RNNs) such as long short-term memory (LSTM)."
    "Later variations have been widely adopted for training large language models (LLMs) on large (language) datasets.",
    "The modern version of the transformer was proposed in the 2017 paper 'Attention Is All You Need' by researchers at Google.",
    "The predecessors of transformers were developed as an improvement over previous architectures for machine translation, but have found many applications since.",
    "They are used in large-scale natural language processing, computer vision (vision transformers), reinforcement learning,[6][7] audio,[8] multimodal learning, robotics,[9] and even playing chess.[10] It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs)[11] and BERT[12] (bidirectional encoder representations from transformers).",
    "Transformers are the foundational neural network architecture enabling modern Large Language Models (LLMs) like ChatGPT, allowing them to process sequences of text efficiently using a self-attention mechanism to weigh word importance, leading to deep contextual understanding for tasks like text generation, translation, and summarization, essentially powering AI's ability to understand and create human-like language.",
    "LLMs use stacked transformer blocks (encoders/decoders) to predict the next word by understanding context from vast datasets, making them powerful tools for complex NLP applications.",
    "The transformer model is a type of neural network architecture that excels at processing sequential data, most prominently associated with large language models (LLMs).",
    "Transformer models have also achieved elite performance in other fields of artificial intelligence (AI), such as computer vision, speech recognition and time series forecasting."
]

def initialize_models():
    "Initialize the three embedding models for Hybrid search"
    print("Initializing models...")
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    print("Dense embedding model loaded (all-MiniLM-L6-v2)")
    bm25_model = SparseTextEmbedding("Qdrant/bm25")
    print("BM25 embedding model loaded (Qdrant/bm25)")
    late_interaction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
    print("Late interaction embedding model loaded (colbert-ir/colbertv2.0)")

    return dense_model, bm25_model, late_interaction_model


# Function to embed documents
def embed_docs(documents=text):

    # Embedding Model
    """
    Embeds a list of documents into a Qdrant vector store.

    Args:
        text (list[str]): A list of documents to embed.

    Returns:
        None
    """

    # Initialize models
    dense_model, bm25_model, late_interaction_model = initialize_models()
    
    # Qdrant Client
    client = QdrantClient(":memory:")

    # Embedding Documents
    dense_embeddings = list(dense_model.embed(doc for doc in documents))
    bm25_embeddings = list(bm25_model.embed(doc for doc in documents))
    late_interaction_embeddings = list(late_interaction_model.embed(doc for doc in documents))

    # Creating a collection
    if not client.collection_exists("test_collection"):
        client.create_collection(
            collection_name="test_collection",
            vectors_config={
                "all-MiniLM-L6-v2": models.VectorParams(
                size=len(dense_embeddings[0]),
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=len(late_interaction_embeddings[0][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0)  #  Disable HNSW for reranking
            ),
        },
    sparse_vectors_config={
        "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )

    # Upload data to Qdrant
    points = []
    for idx, (dense_embedding, bm25_embedding, late_interaction_embedding, doc) in enumerate(zip(dense_embeddings, bm25_embeddings, late_interaction_embeddings, documents)):
        point = PointStruct(
            id=idx,
            vector={
                "all-MiniLM-L6-v2": dense_embedding,
                "bm25": bm25_embedding.as_object(),
                "colbertv2.0": late_interaction_embedding,
            },
        payload={"document": doc}
        )
        points.append(point)

    # Upsert data (Update/Insert)
    client.upsert(
        collection_name="test_collection",
        points=points
        )

    print("Embedding done!")
    print(f'There are {client.get_collection("test_collection").points_count} points in the collection\n')

    return client


## Function to get documents from Qdrant
def get_doc_answer(docs, query: str, k: int = 3) -> list[str]:
    """
    Retrieves k documents from Qdrant based on query.

    Args:
        docs (QdrantClient): The Qdrant client.
        query (str): The query to search for.
        k (int): The number of documents to retrieve. Defaults to 2.

    Returns:
        list[str]: A list of k document descriptions.
    """
    print("Retrieving nodes")

    # Initialize models
    dense_embeddings, bm25_embeddings, late_interaction_embeddings = initialize_models()

    # Embedding Model
    dense_vectors = next(dense_embeddings.query_embed(query))
    sparse_vectors = next(bm25_embeddings.query_embed(query))
    late_vectors = next(late_interaction_embeddings.query_embed(query))

    # Embedd Query
    prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=k,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="bm25",
            limit=k,
        ),
    ]

    # Results
    client=docs

    results = client.query_points(
         "test_collection",
        prefetch=prefetch,
        query=late_vectors,
        using="colbertv2.0",
        with_payload=True,
        limit=k,
)

    return results



if __name__ == "__main__":
    
    query= "nothing to see here"

    embedded_docs = embed_docs()

    retrieved_docs = get_doc_answer(docs=embedded_docs, query=query, k=3)
    
    print('\n ---')
    print('Results:\n')
    print(retrieved_docs)