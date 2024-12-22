import streamlit as st
from src.vectorstore.pinecone_db import ingest_data, get_retriever, load_documents, process_chunks, save_to_parquet
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from src.agents.workflow import run_adaptive_rag
from langgraph.pregel import GraphRecursionError
import tempfile
import os
import time
from pathlib import Path

# Page config
st.set_page_config(page_title="RAG Chat Assistant", layout="wide")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pinecone_client" not in st.session_state:
    st.session_state.pinecone_client = None

def initialize_pinecone(api_key):
    """Initialize Pinecone client with API key."""
    try:
        return Pinecone(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

def initialize_llm(llm_option, openai_api_key=None):
    """Initialize LLM based on user selection."""
    if llm_option == "OpenAI":
        if not openai_api_key:
            st.sidebar.warning("Please enter OpenAI API key.")
            return None
        return ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    else:
        return ChatOllama(model="llama3.2", temperature=0.3, num_predict=512, top_p=0.6)

def clear_pinecone_index(pc, index_name="vector-index"):
    """Clear the Pinecone index."""
    try:
        pc.delete_index(index_name)
        st.session_state.documents_processed = False
        st.session_state.retriever = None
        st.success("Database cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")

def process_documents(uploaded_files, pc):
    """Process uploaded documents and store in Pinecone."""
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return False

    with st.spinner("Processing documents..."):
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        markdown_path = Path(temp_dir) / "combined.md"
        parquet_path = Path(temp_dir) / "documents.parquet"
        
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            file_paths.append(str(file_path))

        try:
            markdown_path = load_documents(file_paths, output_path=markdown_path)
            chunks = process_chunks(markdown_path, chunk_size=256, threshold=0.6)
            print(f"Processed chunks: {chunks}")
            parquet_path = save_to_parquet(chunks, parquet_path)
            
            ingest_data(
                pc=pc,
                parquet_path=parquet_path,
                text_column="text",
                pinecone_client=pc
            )
            
            st.session_state.retriever = get_retriever(pc)
            st.session_state.documents_processed = True
            
            return True
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False
        finally:
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

def run_rag_with_streaming(retriever, question, llm, enable_web_search=False):
    """Run RAG workflow and yield streaming results."""
    try:
        response = run_adaptive_rag(
            retriever=retriever,
            question=question,
            llm=llm,
            top_k=5,
            enable_websearch=enable_web_search
        )
        
        for word in response.split():
            yield word + " "
            time.sleep(0.03)
            
    except GraphRecursionError:
        response = "I apologize, but I cannot find a sufficient answer to your question in the provided documents. Please try rephrasing your question or ask something else about the content of the documents."
        for word in response.split():
            yield word + " "
            time.sleep(0.03)
            
    except Exception as e:
        yield f"I encountered an error while processing your question: {str(e)}"

def main():
    st.title("🤖 RAG Chat Assistant")
    
    # Sidebar configuration
    st.sidebar.title("Configuration")
    
    # API Keys in sidebar
    pinecone_api_key = st.sidebar.text_input("Enter Pinecone API Key:", type="password")
    
    # LLM Selection
    llm_option = st.sidebar.selectbox("Select Language Model:", ["OpenAI", "Ollama"])
    openai_api_key = None
    if llm_option == "OpenAI":
        openai_api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    
    # Web search tool in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tools")
    use_web_search = st.sidebar.checkbox("Web search")
    
    # Initialize Pinecone
    if pinecone_api_key:
        if st.session_state.pinecone_client is None:
            st.session_state.pinecone_client = initialize_pinecone(pinecone_api_key)
    else:
        st.sidebar.warning("Please enter Pinecone API key to continue.")
        st.stop()
    
    # Initialize LLM
    llm = initialize_llm(llm_option, openai_api_key)
    if llm is None:
        st.stop()
    
    # Clear DB Button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Database"):
        if st.session_state.pinecone_client:
            clear_pinecone_index(st.session_state.pinecone_client)
            st.session_state.messages = []  # Clear chat history
    
    # Document upload section
    if not st.session_state.documents_processed:
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "pptx", "md"]
        )
        
        if st.button("Process Documents"):
            if process_documents(uploaded_files, st.session_state.pinecone_client):
                st.success("Documents processed successfully!")
            
    # Chat interface
    if st.session_state.documents_processed:
        st.header("💬 Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Display user message
            with st.chat_message("user"):
                if use_web_search:
                    st.markdown(prompt.strip() + ''' :red-background[Web Search]''')
                else:
                    st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate and stream response
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_response = ""
                
                # Show spinner while processing
                with st.spinner("Thinking..."):
                    # Stream the response
                    for chunk in run_rag_with_streaming(
                        retriever=st.session_state.retriever,
                        question=prompt,
                        llm=llm,
                        enable_web_search=use_web_search
                    ):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                
                # Final update without cursor
                response_container.markdown(full_response)
                
                # Save to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

if __name__ == "__main__":
    main()
