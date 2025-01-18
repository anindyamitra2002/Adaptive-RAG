from src.data_processing.loader import MultiFormatDocumentLoader
from src.data_processing.chunker import SDPMChunker, BGEM3Embeddings

import pandas as pd
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os
import logging
from typing import Dict, List, Any
import time
from chonkie import RecursiveChunker, RecursiveRules


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('rag_chunking')

load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

embedding_model = BGEM3Embeddings(model_name="BAAI/bge-m3")


def load_documents(file_paths: List[str], output_path='./data/output.md'):
    """
    Load documents from multiple sources and combine them into a single markdown file
    """
    loader = MultiFormatDocumentLoader(
        file_paths=file_paths,
        enable_ocr=False,
        enable_tables=True
    )
    
    # Append all documents to the markdown file
    with open(output_path, 'w') as f:
        for doc in loader.lazy_load():
            # Add metadata as YAML frontmatter
            f.write('---\n')
            for key, value in doc.metadata.items():
                f.write(f'{key}: {value}\n')
            f.write('---\n\n')
            f.write(doc.page_content)
            f.write('\n\n')
    
    return output_path

def process_chunks(
    markdown_path: str, 
    chunk_size: int = 512,
    threshold: float = 0.7, 
    skip_window: int = 2,
    large_doc_threshold: int = 100000  # 100KB threshold for large documents
):
    """
    Process the markdown file into chunks and prepare for vector storage.
    Uses different chunking strategies based on document size.
    
    Args:
        markdown_path (str): Path to the markdown file
        chunk_size (int): Size of chunks
        threshold (float): Similarity threshold for SDPM chunker
        skip_window (int): Skip window size for SDPM chunker
        large_doc_threshold (int): Threshold in bytes to determine large documents
        
    Returns:
        List[Dict[str, Any]]: List of processed chunks with metadata
    """
    start_time = time.time()
    logger.info(f"Starting document processing: {markdown_path}")
    
    # Read the markdown file
    try:
        with open(markdown_path, 'r') as file:
            text = file.read()
    except Exception as e:
        logger.error(f"Error reading file {markdown_path}: {str(e)}")
        raise
    
    doc_length = len(text)
    logger.info(f"Document length: {doc_length} characters")
    
    # Choose chunking strategy based on document size
    if doc_length > large_doc_threshold:
        logger.info("Using RecursiveChunker for large document")
        chunker = RecursiveChunker(
            tokenizer="gpt2",
            chunk_size=chunk_size,
            rules=RecursiveRules(),
            min_characters_per_chunk=12,
        )
        chunks = chunker.chunk(text)
        chunker_type = "recursive"
    else:
        logger.info("Using SDPM Chunker for standard document")
        chunker = SDPMChunker(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            threshold=threshold,
            skip_window=skip_window
        )
        chunks = chunker.chunk(text)
        chunker_type = "sdpm"
    
    # Process chunks and collect metadata
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            chunk_data = {
                'text': chunk.text,
                'token_count': chunk.token_count,
                'start_index': chunk.start_index,
                'end_index': chunk.end_index
            }
            processed_chunks.append(chunk_data)
            
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {str(e)}")
            continue
    
    # Log processing summary
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info(
        f"Chunking completed:\n"
        f"- Processing time: {processing_time:.2f} seconds\n"
        f"- Chunker used: {chunker_type}\n"
        f"- Total chunks created: {len(processed_chunks)}\n"
        f"- Average chunk size: {sum(len(c['text']) for c in processed_chunks) / len(processed_chunks):.2f} characters"
    )
    
    return processed_chunks

def save_to_parquet(chunks: List[Dict[str, Any]], output_path='./data/chunks.parquet'):
    """
    Save processed chunks to a Parquet file
    """
    df = pd.DataFrame(chunks)
    print(f"Saving to Parquet: {output_path}")
    df.to_parquet(output_path)
    print(f"Saved to Parquet: {output_path}")
    return output_path


class PineconeRetriever:
    def __init__(
        self,
        pinecone_client: Pinecone,
        index_name: str,
        namespace: str,
        embedding_generator: BGEM3Embeddings
    ):
        """Initialize the retriever with Pinecone client and embedding generator.
        
        Args:
            pinecone_client: Initialized Pinecone client
            index_name: Name of the Pinecone index
            namespace: Namespace in the index
            embedding_generator: BGEM3Embeddings instance
        """
        self.pinecone = pinecone_client
        self.index = self.pinecone.Index(index_name)
        self.namespace = namespace
        self.embedding_generator = embedding_generator
    
    def invoke(self, question: str, top_k: int = 5):
        """Retrieve similar documents for a question.
        
        Args:
            question: Query string
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing retrieved documents
        """
        # Generate embedding for the question
        question_embedding = self.embedding_generator.embed(question)
        question_embedding = question_embedding.tolist()
        # Query Pinecone
        results = self.index.query(
            namespace=self.namespace,
            vector=question_embedding,
            top_k=top_k,
            include_values=False,
            include_metadata=True
        )
        
        # Format results
        retrieved_docs = [
            {"page_content": match.metadata["text"], "score": match.score} 
            for match in results.matches
        ]
        
        return retrieved_docs

def ingest_data(
    pc,
    parquet_path: str,
    text_column: str,
    pinecone_client: Pinecone,
    index_name= "vector-index",
    namespace= "rag",
    batch_size: int = 256,
    progress_callback=None
):
    """Ingest data from a Parquet file into Pinecone.
    
    Args:
        parquet_path: Path to the Parquet file
        text_column: Name of the column containing text data
        pinecone_client: Initialized Pinecone client
        index_name: Name of the Pinecone index
        namespace: Namespace in the index
        batch_size: Batch size for processing
    """
    # Read Parquet file
    print(f"Reading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Total records: {len(df)}")
    # Create or get index
    if not pinecone_client.has_index(index_name):
        pinecone_client.create_index(
            name=index_name,
            dimension=1024,  # BGE-M3 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
        # Wait for index to be ready
        while not pinecone_client.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    index = pinecone_client.Index(index_name)
    total_rows = len(df)
    # Process in batches
    for i in tqdm(range(0, total_rows, batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        
        # Generate embeddings for batch
        texts = batch_df[text_column].tolist()
        embeddings = embedding_model.embed_batch(texts)
        print(f"embeddings for batch: {i}")
        # Prepare records for upsert
        records = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            records.append({
                "id": str(row.name),  # Using DataFrame index as ID
                "values": embeddings[idx],
                "metadata": {"text": row[text_column]}
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=records, namespace=namespace)
        if progress_callback:
            progress_callback(min(i + batch_size, total_rows), total_rows)
        # Small delay to handle rate limits
        time.sleep(0.5)

def get_retriever(
    pinecone_client: Pinecone,
    index_name= "vector-index",
    namespace= "rag"
):
    """Create and return a PineconeRetriever instance.
    
    Args:
        pinecone_client: Initialized Pinecone client
        index_name: Name of the Pinecone index
        namespace: Namespace in the index
        
    Returns:
        Configured PineconeRetriever instance
    """
    return PineconeRetriever(
        pinecone_client=pinecone_client,
        index_name=index_name,
        namespace=namespace,
        embedding_generator=embedding_model
    )
    
def main():
    # Initialize Pinecone client
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
        # print("Loading documents...")
        # markdown_path = load_documents(file_paths)
        
        # # Step 2: Process into chunks with embeddings
        # print("Processing chunks...")
        # chunks = process_chunks(markdown_path)
        
        # # Step 3: Save to Parquet
        # print("Saving to Parquet...")
        # parquet_path = save_to_parquet(chunks)
        
        # # Step 4: Ingest into Pinecone
        # print("Ingesting into Pinecone...")
        # ingest_data(
            # pc,
        #     parquet_path=parquet_path,
        #     text_column="text",
        #     pinecone_client=pc,
        # )
        
        # Step 5: Test retrieval
        print("\nTesting retrieval...")
        retriever = get_retriever(
            pinecone_client=pc,
            index_name="vector-index",
            namespace="rag"
        )
        
        results = retriever.invoke(
            question="describe the gender based violence",
            top_k=5
        )
        
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {doc['page_content']}...")
            print(f"Score: {doc['score']}")
            
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")

if __name__ == "__main__":
    main()