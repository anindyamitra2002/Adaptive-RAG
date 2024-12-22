# Adaptive RAG Model

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Demo Link](#demo-link)
- [Code Structure](#code-structure)
- [Script Explanation](#script-explanation)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

The **Adaptive Retrieval-Augmented Generation (RAG)** model is an advanced tool designed to assist in querying large sets of documents and generating human-like responses based on the retrieved information. This system utilizes cutting-edge technologies such as **Pinecone** for fast vector-based document retrieval and **OpenAI** for language generation. The Adaptive RAG model is built to support real-time queries from users, provide highly relevant information from large document collections, and generate detailed answers, making it ideal for applications like chat assistants, knowledge bases, and automated support systems.

This model leverages several innovative techniques to enhance performance and retrieval accuracy, including **advanced document parsing**, **semantic chunking**, and **multilingual embeddings**. These optimizations make it robust in handling diverse document formats, improving the quality of information retrieval, and ensuring that the generated answers are contextually accurate.

---

## Prerequisites

Before setting up and using the Adaptive RAG model, ensure you have the following prerequisites:

- **Python**: Version 3.10 or higher.
- **Pinecone Account**: To use Pinecone as the vector database for storing and retrieving documents.
- **API Keys**: You'll need API keys for both Pinecone and OpenAI to use their services.
- **Required Libraries**: The model depends on several Python libraries and tools for its operation.

---

## Installation

Follow these steps to install the Adaptive RAG model and its dependencies:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anindyamitra2002/Adaptive-RAG.git
   cd Adaptive-RAG
   ```

2. **Set up a virtual environment**:
   For better management of dependencies, it’s recommended to use a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   Once the virtual environment is activated, setup the dependency.
   ```bash
   bash execute.sh
   ```

4. **Set up Pinecone and OpenAI**:
   - Sign up for a Pinecone account and obtain your API key.
   - Sign up for OpenAI and obtain your API key if you're using the OpenAI model.

---

## Configuration

The configuration settings for the Adaptive RAG model are defined within the application. You’ll need to provide API keys for **Pinecone** and **OpenAI** (if using the OpenAI model) through the web interface or the configuration file.

### Pinecone API Key

The Pinecone API key is required for connecting to Pinecone’s vector database. You can get this key from your Pinecone account. Enter it in the sidebar of the application or configure it in the environment variables.

### OpenAI API Key (Optional)

If you choose to use OpenAI’s language model for generating responses, you’ll need to input your OpenAI API key. This key is essential for accessing OpenAI’s services.

### Document Uploads

You can upload your documents in various formats such as PDF, DOCX, PPTX, or Markdown. The system processes and stores the content in Pinecone to be retrieved during question-answering.

### Web Search Option

A web search option is available to retrieve additional information from the web to answer queries more effectively. This option is configurable via the sidebar in the application.

---

## Usage

The Adaptive RAG model is used through a simple web interface built using Streamlit. Here's how to use the model effectively:

1. **Launch the application**:
   Run the following command to start the application locally:
   ```bash
   streamlit run app.py
   ```

2. **Enter API Keys**:
   Input your Pinecone API key and OpenAI API key (if applicable) in the sidebar.

3. **Upload Documents**:
   You can upload multiple documents that the model will process and store in Pinecone. Supported formats include PDF, DOCX, PPTX, and Markdown.

4. **Ask Questions**:
   Once the documents are processed, you can interact with the system by asking questions related to the content of the uploaded documents. The model will retrieve the relevant information and generate detailed answers.

5. **Streaming Responses**:
   The system supports streaming answers, meaning you will see the response being generated in real-time, simulating a natural conversation.

6. **Web Search**:
   You can enable the web search option to allow the model to pull additional information from the internet to enhance the accuracy of the generated responses.

---

## Demo Link

You can access a live demo of the Adaptive RAG model [here](https://huggingface.co/spaces/anindya-hf-2002/Adaptive-RAG).

---

## Code Structure

The project is organized into the following structure:

```
adaptive_rag/
├── app.py                  # Main Streamlit application
├── src/
│   ├── agents/            # Agent-based RAG components
│   ├── data_processing/   # Document processing utilities
│   ├── llm/              # Language model integrations
│   ├── tools/            # Additional utilities
│   └── vectorstore/      # Vector database operations
├── data/                 # Sample documents & outputs
├── notebooks/           # Development notebooks
└── requirements.txt     # Project dependencies
```

---
## Example

### Uploading Documents

To begin using the system, you can upload a document in PDF format. For example:

1. Click the **Upload Documents** button in the application.
2. Choose the any file (supported format) from your local system.
3. Click **Process Documents**. The system will process and store the document in Pinecone.

### Asking Questions

Once the document is processed, you can start asking questions. For example, ask the system:

**Question**: "What is the summary of the document?"

The system will retrieve relevant content from the uploaded document and generate a response using the OpenAI language model.

---

## Contributing

We welcome contributions from the community. If you’d like to contribute to the project, please fork the repository, make your changes, and submit a pull request. Please ensure that your code adheres to the project’s style guidelines and is thoroughly tested.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
