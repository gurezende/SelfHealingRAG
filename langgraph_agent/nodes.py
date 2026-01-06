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
    retrieval_mode: str
    retrieval_budget: int
    answer: str
    score: float
    failure_reason: str
    retry_count: int
    max_retries: int
    healing_trace: List[str]

# One node retrieves
def retrieve_node(state):
    query = state["query"]
    budget = state["retrieval_budget"]
    mode = state["retrieval_mode"]
    
    # Embed Documents
    docs = embed_docs()

    # Get Answer
    results = get_doc_answer(docs=docs,
                             query=query,
                             k=budget)
    
    # Read retrieval model
    if state["retrieval_mode"] == "dense_rerank":
        results = rerank(query=query, retrieved_docs=results)
    
    return {"retrieved_docs": results,
            "healing_trace": state["healing_trace"]}
 

# One node generates
def generate_node(state):
    print('Generating answer...')
    return {"answer": "some text"}


# One node evaluates
def score_node(state: RAGState):
    score = state["score"] 

    if score >= 0.5:
        return {"score": score, "failure_reason": ""}

    docs = state["retrieved_docs"]
    answer = state["answer"]

    # Very simple heuristic for now
    if len(docs) == 0:
        failure = "missing_context"
    else:
        failure = "irrelevant_docs"

    return {
        "score": score,
        "failure_reason": failure
    }

# One node for decision retry or end
def should_retry(state):
    if state["score"] < 0.5 and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "end"


# One node for retry decision
def retry_node(state: RAGState):
    failure = state["failure_reason"]
    trace = state.get("healing_trace", [])

    if failure == "missing_context":
        trace.append("Missing context → increased retrieval budget by 3 + rerank")
        return {
            "retrieval_budget": state["retrieval_budget"] + 3,
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace
        }

    if failure == "irrelevant_docs":
        trace.append("Irrelevant docs → enabled rerank + increased retrieval budget by 2")
        return {"retrieval_budget": state["retrieval_budget"] + 2,
                "retrieval_mode": "dense_rerank",
                "healing_trace": trace}

    return {}


# One node for retry count
def retry_count_node(state):
    return {"retry_count": state["retry_count"] + 1}