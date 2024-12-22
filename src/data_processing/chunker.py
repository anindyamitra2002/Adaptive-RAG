from typing import List
import numpy as np
from chonkie.embeddings import BaseEmbeddings
from FlagEmbedding import BGEM3FlagModel
from chonkie import SDPMChunker as SDPMChunker

class BGEM3Embeddings(BaseEmbeddings):
    def __init__(self, model_name):
        self.model = BGEM3FlagModel(model_name, use_fp16=True)
        self.task = "separation"
    
    @property
    def dimension(self):
        return 1024

    def embed(self, text: str):
        e = self.model.encode([text], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
        # print(e)
        return e

    def embed_batch(self, texts: List[str]):
        embeddings = self.model.encode(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False
        )
        # print(embeddings['dense_vecs'])
        return embeddings['dense_vecs']

    def count_tokens(self, text: str):
        l = len(self.model.tokenizer.encode(text))
        # print(l)
        return l

    def count_tokens_batch(self, texts: List[str]):
        encodings = self.model.tokenizer(texts)
        # print([len(enc) for enc in encodings["input_ids"]])
        return [len(enc) for enc in encodings["input_ids"]]

    def get_tokenizer_or_token_counter(self):
        return self.model.tokenizer
    
    def similarity(self, u: "np.ndarray", v: "np.ndarray"):
        """Compute cosine similarity between two embeddings."""
        s = (u@v.T)#.item()
        # print(s)
        return s
    
    @classmethod
    def is_available(cls):
        return True

    def __repr__(self):
        return "bgem3"


def main():
    # Initialize the BGE M3 embeddings model
    embedding_model = BGEM3Embeddings(
        model_name="BAAI/bge-m3"
    )

    # Initialize the SDPM chunker
    chunker = SDPMChunker(
        embedding_model=embedding_model,
        chunk_size=256,
        threshold=0.7,
        skip_window=2
    )

    with open('./output.md', 'r') as file:
        text = file.read()

    # Generate chunks
    chunks = chunker.chunk(text)

    # Print the chunks
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"Text: {chunk.text}")
        print(f"Token count: {chunk.token_count}")
        print(f"Start index: {chunk.start_index}")
        print(f"End index: {chunk.end_index}")
        print(f"no of sentences: {len(chunk.sentences)}")
        print("-" * 80)

if __name__ == "__main__":
    main()