from typing import List, TypedDict
from langchain_core.documents.base import Document

class GraphState(TypedDict):
    """
    Represents the state of our adaptive RAG graph.

    Attributes:
        question (str): Original user question
        generation (str, optional): LLM generated answer
        documents (List[Document], optional): Retrieved or searched documents
    """
    question: str
    generation: str | None
    documents: List[Document]
