from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List

class DocumentRelevance(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class HallucinationCheck(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class AnswerQuality(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

def create_llm_grader(grader_type: str, llm):
    """
    Create an LLM grader based on the specified type.
    
    Args:
        grader_type (str): Type of grader to create
    
    Returns:
        Callable: LLM grader function
    """
    # Initialize LLM
    
    # Select grader type and create structured output
    if grader_type == "document_relevance":
        structured_llm_grader = llm.with_structured_output(DocumentRelevance)
        system = """You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
    elif grader_type == "hallucination":
        structured_llm_grader = llm.with_structured_output(HallucinationCheck)
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        
    elif grader_type == "answer_quality":
        structured_llm_grader = llm.with_structured_output(AnswerQuality)
        system = """You are a grader assessing whether an answer addresses / resolves a question. 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])
    
    else:
        raise ValueError(f"Unknown grader type: {grader_type}")
    
    return prompt | structured_llm_grader

def grade_document_relevance(question: str, document: str, llm):
    """
    Grade the relevance of a document to a given question.
    
    Args:
        question (str): User's question
        document (str): Retrieved document content
    
    Returns:
        str: Binary score ('yes' or 'no')
    """
    grader = create_llm_grader("document_relevance", llm)
    result = grader.invoke({"question": question, "document": document})
    return result.binary_score

def check_hallucination(documents: List[str], generation: str, llm):
    """
    Check if the generation is grounded in the provided documents.
    
    Args:
        documents (List[str]): List of source documents
        generation (str): LLM generated answer
    
    Returns:
        str: Binary score ('yes' or 'no')
    """
    grader = create_llm_grader("hallucination", llm)
    result = grader.invoke({"documents": documents, "generation": generation})
    return result.binary_score

def grade_answer_quality(question: str, generation: str, llm):
    """
    Grade the quality of the answer in addressing the question.
    
    Args:
        question (str): User's original question
        generation (str): LLM generated answer
    
    Returns:
        str: Binary score ('yes' or 'no')
    """
    grader = create_llm_grader("answer_quality", llm)
    result = grader.invoke({"question": question, "generation": generation})
    return result.binary_score

if __name__ == "__main__":
    # Example usage
    test_question = "What are the types of agent memory?"
    test_document = "Agent memory can be classified into different types such as episodic, semantic, and working memory."
    test_generation = "Agent memory includes episodic memory for storing experiences, semantic memory for general knowledge, and working memory for immediate processing."
    llm = ChatOllama(model = "llama3.2", temperature = 0.1, num_predict = 256, top_p=0.5)
    
    print("Document Relevance:", grade_document_relevance(test_question, test_document, llm))
    print("Hallucination Check:", check_hallucination([test_document], test_generation, llm))
    print("Answer Quality:", grade_answer_quality(test_question, test_generation, llm))