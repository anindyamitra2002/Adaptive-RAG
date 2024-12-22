from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from src.vectorstore.pinecone_db import ingest_data, get_retriever, load_documents, process_chunks, save_to_parquet
from src.agents.workflow import run_adaptive_rag
from langgraph.pregel import GraphRecursionError
import tempfile
import os
from pathlib import Path

def initialize_pinecone(api_key):
    """Initialize Pinecone client with API key."""
    try:
        return Pinecone(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        return None

def initialize_llm(api_key):
    """Initialize OpenAI LLM."""
    try:
        return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
    except Exception as e:
        print(f"Error initializing OpenAI: {str(e)}")
        return None

def process_documents(file_paths, pc):
    """Process documents and store in Pinecone."""
    if not file_paths:
        print("No documents provided.")
        return None

    print("Processing documents...")
    temp_dir = tempfile.mkdtemp()
    markdown_path = Path(temp_dir) / "combined.md"
    parquet_path = Path(temp_dir) / "documents.parquet"

    try:
        markdown_path = load_documents(file_paths, output_path=markdown_path)
        chunks = process_chunks(markdown_path, chunk_size=256, threshold=0.6)
        parquet_path = save_to_parquet(chunks, parquet_path)
        
        ingest_data(
            pc=pc,
            parquet_path=parquet_path,
            text_column="text",
            pinecone_client=pc
        )
        
        retriever = get_retriever(pc)
        print("Documents processed successfully!")
        return retriever
        
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return None
    finally:
        try:
            os.remove(markdown_path)
            os.remove(parquet_path)
            os.rmdir(temp_dir)
        except:
            pass

def main():
    # Get API keys
    pinecone_api_key = input("Enter your Pinecone API key: ")
    openai_api_key = input("Enter your OpenAI API key: ")
    
    # Initialize clients
    pc = initialize_pinecone(pinecone_api_key)
    if not pc:
        return
    
    llm = initialize_llm(openai_api_key)
    if not llm:
        return

    # Get document paths
    print("\nEnter the paths to your documents (one per line).")
    print("Press Enter twice when done:")
    
    file_paths = []
    while True:
        path = input()
        if not path:
            break
        if os.path.exists(path):
            file_paths.append(path)
        else:
            print(f"Warning: File {path} does not exist")

    # Process documents
    retriever = process_documents(file_paths, pc)
    if not retriever:
        return

    # Chat loop
    print("\nChat with your documents! Type 'exit' to quit.")
    while True:
        question = input("\nYou: ")
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        try:
            response = run_adaptive_rag(
                retriever=retriever,
                question=question,
                llm=llm,
                top_k=5,
                enable_websearch=False
            )
            print("\nAssistant:", response)
            
        except GraphRecursionError:
            print("\nAssistant: I cannot find a sufficient answer to your question in the provided documents. Please try rephrasing your question or ask something else about the content of the documents.")
            
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()