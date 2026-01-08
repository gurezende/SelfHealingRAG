import os
import numpy as np
import json
import openai
import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langgraph_agent.document_loader import load_document
from dotenv import load_dotenv
load_dotenv()



# Text
# text = [
#     "In deep learning, the transformer is an artificial neural network architecture based on the multi-head attention mechanism, in which text is converted to numerical representations called tokens, and each token is converted into a vector via lookup from a word embedding table.", 
#     "At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism, allowing the signal for key tokens to be amplified and less important tokens to be diminished.",
#     "Transformers have the advantage of having no recurrent units, therefore requiring less training time than earlier recurrent neural architectures (RNNs) such as long short-term memory (LSTM)."
#     "Later variations have been widely adopted for training large language models (LLMs) on large (language) datasets.",
#     "The modern version of the transformer was proposed in the 2017 paper 'Attention Is All You Need' by researchers at Google.",
#     "The predecessors of transformers were developed as an improvement over previous architectures for machine translation, but have found many applications since.",
#     "They are used in large-scale natural language processing, computer vision (vision transformers), reinforcement learning,[6][7] audio,[8] multimodal learning, robotics,[9] and even playing chess.[10] It has also led to the development of pre-trained systems, such as generative pre-trained transformers (GPTs)[11] and BERT[12] (bidirectional encoder representations from transformers).",
#     "Transformers are the foundational neural network architecture enabling modern Large Language Models (LLMs) like ChatGPT, allowing them to process sequences of text efficiently using a self-attention mechanism to weigh word importance, leading to deep contextual understanding for tasks like text generation, translation, and summarization, essentially powering AI's ability to understand and create human-like language.",
#     "LLMs use stacked transformer blocks (encoders/decoders) to predict the next word by understanding context from vast datasets, making them powerful tools for complex NLP applications.",
#     "The transformer model is a type of neural network architecture that excels at processing sequential data, most prominently associated with large language models (LLMs).",
#     "Transformer models have also achieved elite performance in other fields of artificial intelligence (AI), such as computer vision, speech recognition and time series forecasting."
# ]

text = load_document("C:/Users/gurez/OneDrive/Área de Trabalho/Guide_AB_Testing.pdf")


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

    st.caption("🔢 | Embedding done!")
    st.caption(f'➡️ | There are {client.get_collection("test_collection").points_count} points in the collection')

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
    st.caption(":black_circle: | Retrieving nodes")

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
    limit=k)

    description_hits = []
    for i, hit in enumerate(initial_retrieval.points):
        # print(f'Result number {i+1} is \n"{hit.payload["description"]}\"')
        description_hits.append(hit.payload["description"])

    return description_hits


def rerank(query, retrieved_docs):

    # Create Reranker
    reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')
    
    # Return scores between query and each document
    new_scores = list(
    reranker.rerank(query, retrieved_docs)
    )  
    
    # Sort them in order of relevance defined by reranker
    ranking = [ (i, score) for i, score in enumerate(new_scores) ]
    ranking.sort(
        key=lambda x: x[1], reverse=True
    )  

    # Print reranked results
    description_hits = []
    for i, rank in enumerate(ranking):
        # print(f'''Reranked result number {i+1} is \"{retrieved_docs[rank[0]]}\"''')
        description_hits.append(retrieved_docs[rank[0]])

    return description_hits


# LLM-as-a-Judge Prompt
llm_judge_prompt = """
You are an expert evaluator of Retrieval-Augmented Generation systems.

User question:
{query}

Retrieved documents:
{retrieved_docs}

Generated answer:
{answer}

Evaluate the answer using the retrieved documents.

Answer the following in JSON:
{{
  "relevant_docs": true | false,
  "sufficient_context": true | false,
  "score": number between 0 and 1
}}

Guidelines:
- relevant_docs = false if documents do not address the user question
- sufficient_context = false if documents are related but incomplete
- score should reflect overall answer quality and faithfulness
"""

# Function LLM-as-a-Judge
def llm_judge(query, retrieved_docs, answer):
    """
    Evaluate the answer using the retrieved documents.

    Args:
        query (str): The user query.
        retrieved_docs (list[str]): The retrieved documents.
        answer (str): The generated answer.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    prompt = llm_judge_prompt.format(query=query, retrieved_docs=retrieved_docs, answer=answer)
    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    
    query= "What is A/B testing?"

    embedded_docs = embed_docs()

    retrieved_docs = get_doc_answer(docs=embedded_docs, query=query, k=5)
    
    print('\n ---')
    print('Reranking results...\n')

    final_docs = rerank(query=query, retrieved_docs=retrieved_docs)
    print('\n Final Docs---')
    print(final_docs)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    ai_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "Use the following documents to answer the user question: " + str(final_docs) + "If the answer cannot be found in the documents, respond with 'I didn't find any relevant documents.'"},
            {"role": "user", "content": query}
        ]
        )

    print("LLM Answer:")
    print(ai_answer.choices[0].message.content, "\n")

    print("LLM Judge:")
    print(llm_judge(query=query, 
              retrieved_docs=final_docs, 
              answer=ai_answer.choices[0].message.content))