from typing import TypedDict, List
from langgraph_agent.retrieve_docs import *
import streamlit as st

# """
# This script contains the state and nodes for the RAG agent.
# The state is a dictionary of the current state of the agent,
#   and the nodes are functions that take the state as input 
#   and return a dictionary of the next state.
#  """

# Define the state schema (just a dictionary for now)
class RAGState(TypedDict):
    text: List[str]
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
    text = state["text"]
    
    # Embed Documents
    docs = embed_docs(text)

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
    st.caption(':robot: | Generating answer...')
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    ai_answer = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "Use the following documents to answer the user question: " + str(state["retrieved_docs"]) + "If the answer cannot be found in the documents, respond with 'I didn't find any relevant documents.'"},
            {"role": "user", "content": state["query"]}
        ]
        )
    
    # Return AI generated answer
    print('Answer generated:')
    print(ai_answer.choices[0].message.content)
    return {"answer": ai_answer.choices[0].message.content}


# One node evaluates
def score_node(state: RAGState):
    
    # Evaluate the generated answer
    judge = llm_judge(query=state["query"], 
                      retrieved_docs=state["retrieved_docs"], 
                      answer=state["answer"])
    
    score = judge["score"]
    relevant = judge["relevant_docs"]
    sufficient = judge["sufficient_context"]

    st.caption(f"* Score: {score}")
    st.caption(f"* Relevant: {relevant}")
    st.caption(f"* Sufficient: {sufficient}")

    # Determine failure reason
    if not relevant:
        failure_reason = "irrelevant_docs"
    elif not sufficient:
        failure_reason = "missing_context"
    else:
        failure_reason = "none"

    # Print failure reason
    st.caption(f"Failure reason: {failure_reason}")

    # Return score and failure reason
    return {
        "score": score,
        "failure_reason": failure_reason
    }

# One node for decision retry or end
def should_retry(state):
    if state["score"] < 0.8 and state["retry_count"] < state["max_retries"]:
        return "retry"
    return "end"


# One node for retry decision
def retry_node(state: RAGState):
    failure = state["failure_reason"]
    trace = state.get("healing_trace", [])

    if failure == "missing_context":
        trace.append("Missing context → increased retrieval budget by 3 + rerank")
        trace_for_log = str(trace[-1])
        st.caption(f"Healing trace: {trace_for_log}")

        return {
            "retrieval_budget": state["retrieval_budget"] + 3,
            "retrieval_mode": "dense_rerank",
            "healing_trace": trace
        }

    if failure == "irrelevant_docs":
        trace.append("Irrelevant docs → enabled rerank + increased retrieval budget by 2")
        trace_for_log = str(trace[-1])
        st.caption(f"Healing trace: {trace_for_log}")

        return {"retrieval_budget": state["retrieval_budget"] + 2,
                "retrieval_mode": "dense_rerank",
                "healing_trace": trace}

    trace.append("No healing needed")
    
    # Print and return
    trace_for_log = str(trace[-1])
    st.caption(f"Healing trace: {trace_for_log}")
    
    return {"healing_trace": trace}


# One node for retry count
def retry_count_node(state):
    return {"retry_count": state["retry_count"] + 1}