from typing import TypedDict, List
from retrieve_docs import *

# """
# This script contains the state and nodes for the RAG agent.
# The state is a dictionary of the current state of the agent,
#   and the nodes are functions that take the state as input 
#   and return a dictionary of the next state.
#  """

# Define the state schema (just a dictionary for now)
class RAGState(TypedDict):
    query: str
    retrieved_docs: List[str]
    answer: str
    score: float
    retry_count: int
    max_retries: int

# One node retrieves
def retrieve_node(state):
    query = state["query"]
    
    # Embed Documents
    docs = embed_docs()

    # Get Answer
    results = get_doc_answer(docs=docs, 
                   query="What is a transformer models used for?")
    
    return {"retrieved_docs": results}
    


# One node generates
def generate_node(state):
    print('Generating answer...')
    return {"answer": "some text"}

# One node evaluates
def score_node(state):
    print('Scoring...')
    return {"score": state["score"]}

# One node for retry decision
def retry_node(state: RAGState):
    if state["score"] < 0.5 and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "end"

# One node for retry count
def retry_count_node(state):
    return {"retry_count": state["retry_count"] + 1}

