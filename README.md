# 📄 Document Intelligence Agent - RAG

A powerful PDF chat application that leverages Retrieval-Augmented Generation (RAG) to enable intelligent conversations with your documents.

[![Live Demo](https://img.shields.io/badge/🚀-Live%20Demo-blue)](https://huggingface.co/spaces/Pats182/Document-Intelligence-Agent-RAG)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces/Pats182/Document-Intelligence-Agent-RAG)

## 🎯 Overview

This application allows users to upload PDF documents and interact with them through natural language questions. Using advanced AI techniques, it retrieves relevant information from your documents and provides accurate, context-aware answers.

**🔗 [Try it Live](https://huggingface.co/spaces/Pats182/Document-Intelligence-Agent-RAG)**

## ✨ Features

- **📤 Easy PDF Upload**: Drag and drop PDF documents through an intuitive interface
- **🤖 AI-Powered Conversations**: Ask questions in natural language and get intelligent responses
- **⚡ Auto-Indexing**: Documents are automatically processed and indexed upon upload
- **🔍 Semantic Search**: Uses vector embeddings for accurate information retrieval
- **💬 Chat History**: Maintains conversation context throughout your session
- **🎨 Clean Interface**: Modern, user-friendly Streamlit interface

## 🏗️ Architecture

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **LLM** | Meta-Llama-3-8B-Instruct |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | FAISS |
| **Framework** | LangChain 0.3+ (LCEL) |
| **PDF Processing** | PyPDFLoader |
| **Text Splitting** | RecursiveCharacterTextSplitter |

### System Flow

```
┌─────────────┐
│  PDF Upload │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ PyPDFLoader     │
│ Extract Text    │
└──────┬──────────┘
       │
       ▼
┌─────────────────────┐
│ Text Splitter       │
│ (1000 chars, 150    │
│  overlap)           │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ HuggingFace         │
│ Embeddings          │
│ (all-MiniLM-L6-v2)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ FAISS Vector Store  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ User Query          │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Retriever (k=3)     │
│ Get relevant chunks │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ RAG Chain (LCEL)    │
│ Context + Question  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Llama 3 Model       │
│ Generate Answer     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Display Response    │
└─────────────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Hugging Face API Token

### Installation

1. **Clone the repository**
   ```bash
   git clone https://huggingface.co/spaces/Pats182/Document-Intelligence-Agent-RAG
   cd Document-Intelligence-Agent-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export HF_Token="your_huggingface_token_here"
   ```

4. **Run the application**
   ```bash
   streamlit run src/streamlit_app.py
   ```

### Usage

1. **Upload a PDF**: Click the "Upload PDF" button in the sidebar
2. **Wait for Processing**: The document will be automatically indexed
3. **Ask Questions**: Type your questions in the chat input
4. **Get Answers**: Receive AI-generated responses based on your document

## 🔧 Configuration

### Model Parameters

- **Temperature**: 0.3 (for more focused responses)
- **Max Tokens**: 512
- **Retrieval Top-K**: 3 chunks
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 150 characters

### Customization

You can modify these parameters in `streamlit_app.py`:

```python
# Adjust LLM parameters
base_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.3,  # Adjust for creativity
    max_new_tokens=512,  # Adjust response length
)

# Adjust text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust chunk size
    chunk_overlap=150  # Adjust overlap
)

# Adjust retrieval
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}  # Number of chunks to retrieve
)
```

## 📊 Key Components

### PDF Processing
The application uses `PyPDFLoader` to extract text from PDF documents with proper error handling and temporary file management.

### Vector Embeddings
Documents are converted into vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`, enabling semantic search capabilities.

### RAG Chain
Built with LangChain's modern LCEL (LangChain Expression Language), the RAG chain:
1. Retrieves relevant document chunks
2. Formats them with the user's question
3. Generates contextual responses using Llama 3

### Prompt Template
```python
Based on the following context, answer the question accurately and concisely.
If the answer is not in the context, say "I don't have enough information to answer that."

Context: {context}
Question: {question}
Answer:
```

## 🎨 Features in Detail

### Auto-Indexing
- Documents are processed immediately upon upload
- No manual indexing button required
- Visual feedback during processing

### Session Management
- Chat history persists during the session
- Current document tracking
- Clear chat option available

### Error Handling
- Comprehensive try-catch blocks
- User-friendly error messages
- Proper temporary file cleanup

## 🛠️ Advanced Usage

### Re-indexing Documents
If you need to reprocess a document:
1. Click the "🔄 Re-index" button in the sidebar
2. Wait for processing to complete

### Clearing Chat History
To start a fresh conversation:
1. Click the "🗑️ Clear Chat" button in the sidebar

## 📈 Performance

- **Fast Retrieval**: FAISS enables efficient similarity search
- **Optimized Chunks**: 1000-character chunks with 150-character overlap ensure context preservation
- **Cached Models**: Streamlit caching reduces model loading time

## 🔒 Security

- API tokens are securely managed through environment variables
- Temporary files are properly cleaned up after processing
- No data persistence beyond the session

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is open source and available under standard terms.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Powered by [Hugging Face](https://huggingface.co/)
- UI by [Streamlit](https://streamlit.io/)
- Model: [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

## 📞 Contact

For questions or feedback, please visit the [Hugging Face Space](https://huggingface.co/spaces/Pats182/Document-Intelligence-Agent-RAG).

---

**🚀 [Launch Live Demo](https://huggingface.co/spaces/Pats182/Document-Intelligence-Agent-RAG)**