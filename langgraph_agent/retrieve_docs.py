import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding


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

# Function to embed documents
def embed_docs(text=text):

    # Embedding Model
    """
    Embeds a list of documents into a Qdrant vector store.

    Args:
        text (list[str]): A list of documents to embed.

    Returns:
        None
    """
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = TextEmbedding(model_name=encoder_name)

    # Creating vector store
    vector_store = list(
        embedding_model.embed(text) )

    # Qdrant Client
    client = QdrantClient(":memory:")

    # Creating a collection
    if not client.collection_exists("test_collection"):
        client.create_collection(
            collection_name="test_collection",
            vectors_config={
                "embedding": VectorParams(
                    size=client.get_embedding_size("sentence-transformers/all-MiniLM-L6-v2"), 
                    distance=Distance.COSINE)
            }
        )

    # Upload data to Qdrant
    client.upload_points(
        collection_name="test_collection",
        points=[
            PointStruct(
                id=idx, 
                payload={"description": description}, 
                vector={"embedding": vector}
            )
            for idx, (description, vector) in enumerate(
                zip(text, vector_store)
            )
        ],
    )

    print("Embedding done!")
    print(f'There are {client.get_collection("test_collection").points_count} points in the collection\n')

    return client


## Function to get documents from Qdrant
def get_doc_answer(docs, query: str, k: int = 2) -> list[str]:
    """
    Retrieves k documents from Qdrant based on query.

    Args:
    docs (list): List of documents to search in.
    query (str): The query to search for.
    k (int): The number of documents to retrieve. Defaults to 2.

    Returns:
    list[str]: A list of k document descriptions.
    """
    print("Retrieving nodes")

    # Embedding Model
    encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = TextEmbedding(model_name=encoder_name)

    # Embedd Query
    query_embedded = list(embedding_model.query_embed(query))[0]

    client=docs
    # Retrieve data
    initial_retrieval = client.query_points(
    collection_name="test_collection",
    using="embedding",
    query=query_embedded,
    with_payload=True,
    limit=3)

    description_hits = []
    for i, hit in enumerate(initial_retrieval.points):
        print(f'Result number {i+1} is \n"{hit.payload["description"]}\"')
        description_hits.append(hit.payload["description"])

    return description_hits


# if __name__ == "__main__":
    
#     docs = embed_docs()

#     get_doc_answer(docs=docs, 
#                    query="What is a transformer models used for?")