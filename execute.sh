# Navigate to the adaptive_rag directory
# cd adaptive_rag/

# Install dependencies
pip3 install -r requirements.txt
crawl4ai-setup

# Set the Python path
export PYTHONPATH=.

# Install and run Ollama
curl -fsSL https://ollama.com/install.sh | sh
sleep 1
ollama run llama3.2
