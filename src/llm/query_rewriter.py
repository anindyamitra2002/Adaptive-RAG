from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

def create_query_rewriter(llm):
    """
    Create a query rewriter to optimize retrieval.
    
    Returns:
        Callable: Query rewriter function
    """
    
    # Prompt for query rewriting
    system = """You are a question re-writer that converts an input question to a better version that is optimized 
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    
    # Create query rewriter chain
    return re_write_prompt | llm | StrOutputParser()

def rewrite_query(question: str, llm):
    """
    Rewrite a given query to optimize retrieval.
    
    Args:
        question (str): Original user question
    
    Returns:
        str: Rewritten query
    """
    query_rewriter = create_query_rewriter(llm)
    try:
        rewritten_query = query_rewriter.invoke({"question": question})
        return rewritten_query
    except Exception as e:
        print(f"Query rewriting error: {e}")
        return question

if __name__ == "__main__":
    # Example usage
    test_queries = [
        "Tell me about AI agents",
        "What do we know about memory in AI systems?",
        "Bears draft strategy"
    ]
    llm = ChatOllama(model = "llama3.2", temperature = 0.1, num_predict = 256, top_p=0.5)
    
    for query in test_queries:
        rewritten = rewrite_query(query, llm)
        print(f"Original: {query}")
        print(f"Rewritten: {rewritten}\n")