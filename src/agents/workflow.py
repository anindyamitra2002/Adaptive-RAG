from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import PromptTemplate
from agents.state import GraphState
# from agents.router import route_query
import asyncio
from vectorstore.pinecone_db import get_retriever
from tools.web_search import AdvancedWebCrawler
from llm.graders import (
    grade_document_relevance, 
    check_hallucination, 
    grade_answer_quality
)
from langchain_core.output_parsers import StrOutputParser
from llm.query_rewriter import rewrite_query
from langchain_ollama import ChatOllama

def perform_web_search(question: str):
    """
    Perform web search using the AdvancedWebCrawler.
    
    Args:
        question (str): User's input question
    
    Returns:
        List: Web search results
    """
    # Initialize web crawler
    crawler = AdvancedWebCrawler(
        max_search_results=5,
        word_count_threshold=50,
        content_filter_type='f',
        filter_threshold=0.48
    )
    results = asyncio.run(crawler.search_and_crawl(question))
    
    return results


def create_adaptive_rag_workflow(retriever, llm, top_k=5, enable_websearch=False):
    """
    Create the adaptive RAG workflow graph.
    
    Args:
        retriever: Vector store retriever
    
    Returns:
        Compiled LangGraph workflow
    """
    def retrieve(state: GraphState):
        """Retrieve documents from vectorstore."""
        print("---RETRIEVE---")
        question = state['question']
        documents = retriever.invoke(question, top_k)
        print(f"Retrieved {len(documents)} documents.")
        print(documents)
        return {"documents": documents, "question": question}

    def route_to_datasource(state: GraphState):
        """Route question to web search or vectorstore."""
        print("---ROUTE QUESTION---")
        # question = state['question']
        # source = route_query(question)
       
        if enable_websearch:
            print("---ROUTE TO WEB SEARCH---")
            return "web_search"
        else:
            print("---ROUTE TO RAG---")
            return "vectorstore"

    def generate_answer(state: GraphState):
        """Generate answer using retrieved documents."""
        print("---GENERATE---")
        question = state['question']
        documents = state['documents']
        
        # Prepare context
        context = "\n\n".join([doc["page_content"] for doc in documents])
        prompt_template = PromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer:""")
        # Generate answer
        rag_chain = prompt_template | llm | StrOutputParser()

        generation = rag_chain.invoke({"context": context, "question": question})
        
        return {"generation": generation, "documents": documents, "question": question}

    def grade_documents(state: GraphState):
        """Filter relevant documents."""
        print("---GRADE DOCUMENTS---")
        question = state['question']
        documents = state['documents']
        
        # Filter documents
        filtered_docs = []
        for doc in documents:
            score = grade_document_relevance(question, doc["page_content"], llm)
            if score == "yes":
                filtered_docs.append(doc)
        
        return {"documents": filtered_docs, "question": question}

    def web_search(state: GraphState):
        """Perform web search."""
        print("---WEB SEARCH---")
        question = state['question']
        
        # Perform web search
        results = perform_web_search(question)
        web_documents = [
            {
                "page_content": result['content'], 
                "metadata": {"source": result['url']}
            } for result in results
        ]
        
        return {"documents": web_documents, "question": question}

    def check_generation_quality(state: GraphState):
        """Check the quality of generated answer."""
        print("---ASSESS GENERATION---")
        question = state['question']
        documents = state['documents']
        generation = state['generation']
 
        
        print("---Generation is not hallucinated.---")
        # Check answer quality
        quality_score = grade_answer_quality(question, generation, llm)
        if quality_score == "yes":
            print("---Answer quality is good.---")
        else:
            print("---Answer quality is poor.---")
        return "end" if quality_score == "yes" else "rewrite"

    # Create workflow
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("vectorstore", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("rewrite_query", lambda state: {
        "question": rewrite_query(state['question'], llm),
        "documents": [],
        "generation": None
    })

    # Define edges
    workflow.add_conditional_edges(
        START, 
        route_to_datasource,
        {
            "web_search": "web_search",
            "vectorstore": "vectorstore"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("vectorstore", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        lambda state: "generate" if state['documents'] else "rewrite_query"
    )
    
    workflow.add_edge("rewrite_query", "vectorstore")
    
    workflow.add_conditional_edges(
        "generate",
        check_generation_quality,
        {
            "end": END,
            "regenerate": "generate",
            "rewrite": "rewrite_query"
        }
    )

    # Compile the workflow
    app = workflow.compile()
    return app

def run_adaptive_rag(retriever, question: str, llm, top_k=5, enable_websearch=False):
    """
    Run the adaptive RAG workflow for a given question.
    
    Args:
        retriever: Vector store retriever
        question (str): User's input question
    
    Returns:
        str: Generated answer
    """
    # Create workflow
    workflow = create_adaptive_rag_workflow(retriever, llm, top_k, enable_websearch=enable_websearch)
    
    # Run workflow
    final_state = None
    for output in workflow.stream({"question": question}, config={"recursion_limit": 5}):
        for key, value in output.items():
            print(f"Node '{key}':")
            # Optionally print state details
            # print(value)
        final_state = value
    
    return final_state.get('generation', 'No answer could be generated.')

if __name__ == "__main__":
    # Example usage
    from vectorstore.pinecone_db import PINECONE_API_KEY, ingest_data,  get_retriever, load_documents, process_chunks, save_to_parquet
    from pinecone import Pinecone
    
    # Load and prepare documents
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Define input files
    file_paths=[
        # './data/2404.19756v1.pdf',
        # './data/OD429347375590223100.pdf',
        # './data/Project Report Format.docx',
        './data/UNIT 2 GENDER BASED VIOLENCE.pptx'
    ]

    # Process pipeline
    try:
        # Step 1: Load and combine documents
        print("Loading documents...")
        markdown_path = load_documents(file_paths)
        
        # Step 2: Process into chunks with embeddings
        print("Processing chunks...")
        chunks = process_chunks(markdown_path)
        
        # Step 3: Save to Parquet
        print("Saving to Parquet...")
        parquet_path = save_to_parquet(chunks)
        
        # Step 4: Ingest into Pinecone
        print("Ingesting into Pinecone...")
        ingest_data(pc,
            parquet_path=parquet_path,
            text_column="text",
            pinecone_client=pc,
        )
        
        # Step 5: Test retrieval
        print("\nTesting retrieval...")
        retriever = get_retriever(
            pinecone_client=pc,
            index_name="vector-index",
            namespace="rag"
        )
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")    

    llm = ChatOllama(model = "llama3.2", temperature = 0.1, num_predict = 256, top_p=0.5)
    
    # Test questions
    test_questions = [
        # "What are the key components of AI agent memory?",
        # "Explain prompt engineering techniques",
        # "What are recent advancements in adversarial attacks on LLMs?"
        "what are the trending papers that are published in NeurIPS 2024?"
    ]
    
    # Run workflow for each test question
    for question in test_questions:
        print(f"\n--- Processing Question: {question} ---")
        answer = run_adaptive_rag(retriever, question, llm)
        print("\nFinal Answer:", answer)