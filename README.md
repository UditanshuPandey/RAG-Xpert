
# RAG-Xpert - Intelligent Document Processing

**Deployed Project Link: https://rag-xpert.streamlit.app/** 

An Enhanced Retrieval-Augmented Generation Framework that transforms documents and web content into conversational knowledge bases.


## Features

- Multi-Format Document Processing: Handles PDF, TXT, DOCX, and MD files with automatic content extraction

- Web Content Integration: Processes any website URL into a queryable knowledge source

- GROQ-Powered LLM: Utilizes the llama-3.3-70b-versatile model for high-quality responses

- Advanced RAG Pipeline: Combines ChromaDB vector storage with HuggingFace embeddings

- Context-Aware Conversations: Maintains dialog history for coherent multi-turn discussions

- Dynamic Document Management: Supports up to 10 concurrent knowledge sources


## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/yourusername/RAG-Xpert.git
cd RAG-Xpert
```
**2. Set up a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
**3. Install dependencies:**

```bash
pip install -r requirements.txt
```
**4. Configure API keys: Create a .env file with:**

```bash
GROQ_API_KEY = your_groq_api_key_here
```
## 🛠️ Technology Stack

| Component           | Technology Used |
|---------------------|-----------------|
| Backend Framework   | Python/Streamlit |
| LLM Integration     | Groq API (LLaMA 3.3 70B) |
| Vector Database     | ChromaDB |
| Document Processing | LangChain |
| Embeddings          | HuggingFace |

## Contact
**Developer:** Uditanshu Pandey\
**Email:** uditanshup114@gmail.com\
**Linkedin:** https://www.linkedin.com/in/uditanshupandey
