#%%
from retrieve_docs import get_doc_answer, embed_docs
from nodes import *
from langgraph.graph import StateGraph, END

def build_graph():

    # Initiate the LangGraph flow builder class
    builder = StateGraph(RAGState)

    # Add nodes to the graph
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("score", score_node)
    builder.add_node("retry", retry_node)
    builder.add_node("retry_count", retry_count_node)

    # Define the flow
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "score")
    
    # Conditional Edge: if score < 0.5 and retry_count < 3, then retry
    builder.add_conditional_edges(
        "score", retry_node,
        {
            "retry": "retry_count",
            "end": END
        }
)

    builder.add_edge("retry_count", "retrieve")

    # Compile the graph
    return builder.compile()
    

# Create the graph image and save png
from IPython.display import display, Image
graph = build_graph()
display(Image(graph.get_graph().draw_mermaid_png()))

# %%

graph.invoke({
    "query": "what is a transformer?",
    "retrieved_docs": [],
    "answer": "",
    "score": 0.5,
    "retry_count": 1,
    "max_retries": 2
})

# %%
