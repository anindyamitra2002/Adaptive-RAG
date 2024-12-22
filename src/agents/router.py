from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Route question to web search or vectorstore retrieval"
    )

def create_query_router():
    """
    Create a query router to determine data source for a given question.
    
    Returns:
        Callable: Query router function
    """
    # LLM with function call
    llm = ChatOllama(model = "llama3.2", temperature = 0.1, num_predict = 256, top_p=0.5)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Prompt
    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. Otherwise, use web-search."""
    
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    return route_prompt | structured_llm_router

def route_query(question: str):
    """
    Route a specific query to its appropriate data source.
    
    Args:
        question (str): User's input question
    
    Returns:
        str: Recommended data source
    """
    router = create_query_router()
    result = router.invoke({"question": question})
    return result.datasource

if __name__ == "__main__":
    # Example usage
    test_questions = [
        "Who will the Bears draft first in the NFL draft?",
        "What are the types of agent memory?"
    ]
    
    for q in test_questions:
        source = route_query(q)
        print(f"Question: {q}")
        print(f"Routed to: {source}\n")