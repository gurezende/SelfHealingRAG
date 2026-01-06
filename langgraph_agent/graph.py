#%%
from retrieve_docs import get_doc_answer, embed_docs
from nodes import *
from langgraph.graph import StateGraph, END
from document_loader import load_document

def build_graph():

    # Initiate the LangGraph flow builder class
    builder = StateGraph(RAGState)

    # Add nodes to the graph
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("score", score_node)
    builder.add_node("retry", retry_node)
    builder.add_node("increment_retry", retry_count_node)

    # Define the flow
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "score")
    
    # Conditional Edge: if score < 0.5 and retry_count < 3, then retry
    builder.add_conditional_edges(
        "score", should_retry,
        {
            "retry": "retry",
            "end": END
        }
)

    builder.add_edge("retry", "increment_retry")
    builder.add_edge("increment_retry", "retrieve")
    

    # Compile the graph
    return builder.compile()
    

# Create the graph image and save png
from IPython.display import display, Image
graph = build_graph()
display(Image(graph.get_graph().draw_mermaid_png()))

# %%

# Load document
documents = load_document("C:/Users/gurez/OneDrive/Área de Trabalho/Guide_AB_Testing.pdf")

graph.invoke({
    "query": "What is an A/B Test?",
    "retrieved_docs": [],
    "retrieval_mode": "original",
    "retrieval_budget": 2,
    "failure_reason": "",
    "healing_trace": [],
    "answer": "",
    "score": "",
    "retry_count": 1,
    "max_retries": 2
})


# %%
