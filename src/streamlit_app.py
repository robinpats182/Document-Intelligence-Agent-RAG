import streamlit as st
import os
import tempfile

# --- 1. Modern LangChain 0.3+ Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint

# Modern LCEL imports (replaces chains)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- 2. Configuration & Secrets ---
st.set_page_config(page_title="PDF AI Agent", layout="wide")
st.title("💬 Chat with your PDF")

# Securely grab token
hf_token = os.getenv("HF_Token")
if not hf_token:
    st.error("⚠️ HF_Token not found in environment variables!")
    st.stop()

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- 3. Model Initialization ---
@st.cache_resource
def load_models():
    try:
        embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Meta-Llama-3-8B-Instruct
        base_llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            huggingfacehub_api_token=hf_token,
            temperature=0.3,
            max_new_tokens=512,
        )
        llm = ChatHuggingFace(llm=base_llm)
        
        st.sidebar.info("🤖 Using: meta-llama/Meta-Llama-3-8B-Instruct")
        return embed_model, llm
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

embeddings, llm = load_models()

# --- 4. Logic: PDF Processing ---
def process_pdf(uploaded_file):
    tmp_path = None
    try:
        # Create temporary file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and process PDF
        loader = PyPDFLoader(tmp_path)
        data = loader.load()
        
        if not data:
            st.error("PDF appears to be empty or unreadable")
            return None
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(data)
        
        # Create vector store
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        st.sidebar.success(f"✅ Processed {len(chunks)} chunks from {len(data)} pages")
        return vector_db
        
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return None
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

# --- 5. Helper function to format retrieved documents ---
def format_docs(docs):
    """Combine retrieved documents into a single context string"""
    return "\n\n".join(doc.page_content for doc in docs)

# --- 6. Sidebar: Document Management ---
with st.sidebar:
    st.header("📁 Document Center")
    pdf_doc = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")
    
    # AUTO-INDEX when new file is uploaded
    if pdf_doc:
        if st.session_state.current_file != pdf_doc.name:
            with st.spinner("🔄 Auto-indexing document..."):
                st.session_state.vector_store = process_pdf(pdf_doc)
                st.session_state.current_file = pdf_doc.name
        else:
            st.info(f"📄 Current: {pdf_doc.name}")
    
    # Manual controls
    col1, col2 = st.columns(2)
    with col1:
        if pdf_doc and st.button("🔄 Re-index"):
            with st.spinner("Re-indexing..."):
                st.session_state.vector_store = process_pdf(pdf_doc)
    
    with col2:
        if st.session_state.messages and st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Status indicator
    st.divider()
    if st.session_state.vector_store:
        st.success("✅ Ready to chat")
    else:
        st.warning("⚠️ Upload a PDF to begin")

# --- 7. Main Chat Interface ---
# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if user_input := st.chat_input("Ask a question about your document..."):
    # Check if document is indexed
    if not st.session_state.vector_store:
        st.warning("⚠️ Please upload a PDF in the sidebar first!")
        st.stop()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        try:
            # Modern LCEL approach - Build RAG chain manually
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
            
            # Define prompt template - simple format works best for Flan-T5
            prompt = PromptTemplate.from_template(
                """Based on the following context, answer the question accurately and concisely.
If the answer is not in the context, say "I don't have enough information to answer that."

Context: {context}

Question: {question}

Answer:"""
            )
            
            # Build the LCEL chain (Modern approach)
            rag_chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Invoke chain
            with st.spinner("🔍 Searching document..."):
                answer = rag_chain.invoke(user_input)
                
                # Clean up the response
                answer = answer.strip()
                
                # Display answer
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
                
        except Exception as e:
            st.exception(e)   # 👈 THIS shows the real traceback
            st.session_state.messages.append({
                "role": "assistant",
                "content": str(e)
            })