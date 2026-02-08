# Document Portal - Comprehensive Project Code Flow Analysis

---

## 1. Project Overview

### Purpose
**Document Portal** is a FastAPI-based LLM application that provides three core document processing capabilities:
1. **Document Analysis** - Extract metadata (title, author, summary, sentiment) from uploaded documents
2. **Document Comparison** - Compare two sets of PDFs and identify page-wise differences
3. **Conversational RAG Chat** - Build a vector index from documents and enable Q&A with conversation history

### Technology Stack
| Component | Technology | Version |
|-----------|------------|---------|
| Web Framework | FastAPI | 0.116.1 |
| LLM Orchestration | LangChain (LCEL) | 0.3.27 |
| Vector Store | FAISS (faiss-cpu) | 1.11.0 |
| LLM Providers | Google Gemini 2.0 Flash, Groq DeepSeek R1 | - |
| Embeddings | Google text-embedding-004 | - |
| PDF Processing | PyMuPDF | 1.26.3 |
| Structured Logging | structlog | - |
| Data Validation | Pydantic | - |

### Entry Point
```
api/main.py → FastAPI application
```

### Configuration
```
config/config.yaml → YAML-based configuration for models, embeddings, retriever settings
```

---

## 2. Execution Flow (Top-to-Bottom)

### 2.1 Application Startup

```
┌─────────────────────────────────────────────────────────────────┐
│                        api/main.py                              │
│  FastAPI() instantiation → load templates/static → define routes│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    utils/model_loader.py                        │
│  ModelLoader() → ApiKeyManager() → load .env or ECS secrets     │
│  → Initialize GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    config/config.yaml                           │
│  embedding_model, llm (google/groq), retriever (top_k)          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Request Flow Diagram

```
                           ┌──────────────┐
                           │   Client     │
                           └──────┬───────┘
                                  │ HTTP Request
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        api/main.py                              │
│         GET /          │  POST /analyze  │  POST /compare       │
│         GET /health    │  POST /chat/index │ POST /chat/query   │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┬───────────────────┐
        ▼                ▼                ▼                   ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  /analyze     │ │  /compare     │ │  /chat/index  │ │  /chat/query  │
│  DocHandler   │ │  DocumentComp.│ │  ChatIngestor │ │  Conversational│
│  →Analyzer    │ │  →Comparator  │ │  →FAISS       │ │  RAG          │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
```

---

## 3. Module/Class Structure

### 3.1 Core Service Classes

#### A. Document Analysis Flow
```python
# Endpoint: POST /analyze
# Files: data_ingestion.py → data_analysis.py

DocHandler (src/document_ingestion/data_ingestion.py)
    │
    ├── save_files(files, base_path)      # Save uploaded files to disk
    ├── load_documents(paths)             # Load using PyPDF/Docx2txt/TextLoader
    └── concat_for_analysis(docs)         # Concatenate all documents
            │
            ▼
DocumentAnalyzer (src/document_analyzer/data_analysis.py)
    │
    ├── __init__(llm)                     # Initialize with LLM
    ├── build_chain()                     # Create LCEL chain with JsonOutputParser
    └── analyze(document_text) → Metadata # Return structured metadata
```

**Data Flow:**
```
UploadFile[] → DocHandler.save_files() → Path[]
Path[] → DocHandler.load_documents() → Document[]
Document[] → concat_for_analysis() → str
str → DocumentAnalyzer.analyze() → Metadata(JSON)
```

#### B. Document Comparison Flow
```python
# Endpoint: POST /compare
# Files: data_ingestion.py → document_comparator.py

DocumentComparator (src/document_ingestion/data_ingestion.py)
    │
    ├── save_files(files, base_path)      # Save reference & actual files
    ├── load_documents(paths)             # Load both document sets
    └── concat_for_comparison(ref, act)   # Combine with markers
            │
            ▼
DocumentComparatorLLM (src/document_compare/document_comparator.py)
    │
    ├── __init__(llm)                     # Initialize with LLM
    ├── build_chain()                     # LCEL chain with JsonOutputParser
    └── compare(combined_docs) → List[ChangeFormat] → DataFrame
```

**Data Flow:**
```
ref_files[], actual_files[] → DocumentComparator.save_files() × 2
→ load_documents() × 2 → Document[] × 2
→ concat_for_comparison() → str (<<REFERENCE>> ... <<ACTUAL>>)
→ DocumentComparatorLLM.compare() → DataFrame[Page, Changes]
```

#### C. Conversational RAG Chat Flow
```python
# Endpoints: POST /chat/index, POST /chat/query
# Files: data_ingestion.py → retrieval.py

ChatIngestor (src/document_ingestion/data_ingestion.py)
    │
    ├── save_files(files, base_path)      # Save chat documents
    ├── load_documents(paths)             # Load documents
    ├── split_documents(docs)             # RecursiveCharacterTextSplitter
    └── build_index(chunks) → FAISS       # Create vector store
            │
            ▼
FaissManager (src/document_ingestion/data_ingestion.py)
    │
    ├── __init__(session_id)              # Initialize session-based storage
    ├── save_index(index)                 # Persist FAISS index to disk
    └── load_index() → FAISS              # Load from disk
            │
            ▼
ConversationalRAG (src/document_chat/retrieval.py)
    │
    ├── __init__(retriever, llm)          # Initialize with retriever & LLM
    ├── build_rag_chain()                 # Create history-aware retrieval chain
    └── query(question, history) → str    # Execute conversational query
```

**Data Flow (Indexing):**
```
UploadFile[] → ChatIngestor.save_files() → Path[]
→ load_documents() → Document[]
→ split_documents() → Document[] (chunks)
→ build_index() → FAISS
→ FaissManager.save_index() → disk (session_id/faiss_index/)
```

**Data Flow (Query):**
```
question, session_id → FaissManager.load_index() → FAISS
→ retriever = faiss.as_retriever(k=10)
→ ConversationalRAG(retriever, llm).query(question, history)
→ answer (str)
```

### 3.2 Utility Classes

#### ModelLoader (utils/model_loader.py)
```python
ModelLoader
    │
    ├── __init__()                        # Load .env or ECS secrets
    ├── api_key_mgr: ApiKeyManager        # Manages GROQ_API_KEY, GOOGLE_API_KEY
    ├── config: dict                      # From config.yaml
    │
    ├── load_embeddings()                 # → GoogleGenerativeAIEmbeddings
    └── load_llm()                        # → ChatGoogleGenerativeAI or ChatGroq
```

#### ApiKeyManager (utils/model_loader.py)
```python
ApiKeyManager
    │
    ├── REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY"]
    ├── __init__()                        # Parse API_KEYS JSON or individual env vars
    └── get(key) → str                    # Retrieve specific API key
```

#### File I/O Utilities (utils/file_io.py)
```python
generate_session_id(prefix)               # → "session_YYYYMMDD_HHMMSS_uuid8"
save_uploaded_files(files, dir) → Path[]  # Save with sanitized names
```

#### Document Operations (utils/document_ops.py)
```python
load_documents(paths) → Document[]        # PyPDF/Docx2txt/TextLoader
concat_for_analysis(docs) → str           # Single document concatenation
concat_for_comparison(ref, act) → str     # Dual document with markers
FastAPIFileAdapter                        # Convert UploadFile → buffer API
```

---

## 4. API Endpoints Detail

### 4.1 GET / (Homepage)
```python
@app.get("/")
async def home():
    return templates.TemplateResponse("index.html", {"request": request})
```

### 4.2 GET /health (Health Check)
```python
@app.get("/health")
async def health():
    return {"status": "ok"}
```

### 4.3 POST /analyze (Document Analysis)
```python
@app.post("/analyze")
async def analyze(files: List[UploadFile]):
    # 1. Initialize
    handler = DocHandler(ModelLoader())
    
    # 2. Save files
    paths = handler.save_files(files, DATA_DIR / "document_analysis")
    
    # 3. Load documents
    docs = handler.load_documents(paths)
    combined = concat_for_analysis(docs)
    
    # 4. Analyze
    analyzer = DocumentAnalyzer(handler.loader.load_llm())
    result = analyzer.analyze(combined)
    
    return {"metadata": result}
```

**Request:** `multipart/form-data` with files
**Response:** `{"metadata": {"Title": "...", "Author": [...], "Summary": [...], ...}}`

### 4.4 POST /compare (Document Comparison)
```python
@app.post("/compare")
async def compare(reference_files: List[UploadFile], actual_files: List[UploadFile]):
    # 1. Initialize
    comparator = DocumentComparator(ModelLoader())
    
    # 2. Save both file sets
    ref_paths = comparator.save_files(reference_files, DATA_DIR / "ref")
    act_paths = comparator.save_files(actual_files, DATA_DIR / "act")
    
    # 3. Load and combine
    ref_docs = comparator.load_documents(ref_paths)
    act_docs = comparator.load_documents(act_paths)
    combined = concat_for_comparison(ref_docs, act_docs)
    
    # 4. Compare
    llm_comparator = DocumentComparatorLLM(comparator.loader.load_llm())
    df = llm_comparator.compare(combined)
    
    return {"comparison": df.to_dict(orient="records")}
```

**Request:** `multipart/form-data` with `reference_files[]` and `actual_files[]`
**Response:** `{"comparison": [{"Page": "1", "Changes": "..."}, ...]}`

### 4.5 POST /chat/index (Build FAISS Index)
```python
@app.post("/chat/index")
async def index_documents(files: List[UploadFile]):
    # 1. Generate session
    session_id = generate_session_id("chat")
    
    # 2. Initialize
    ingestor = ChatIngestor(ModelLoader())
    
    # 3. Process
    paths = ingestor.save_files(files, DATA_DIR / "multi_doc_chat" / session_id)
    docs = ingestor.load_documents(paths)
    chunks = ingestor.split_documents(docs)
    index = ingestor.build_index(chunks)
    
    # 4. Persist
    manager = FaissManager(session_id)
    manager.save_index(index)
    
    return {"session_id": session_id, "chunks_indexed": len(chunks)}
```

**Request:** `multipart/form-data` with files
**Response:** `{"session_id": "chat_20250115_143022_abc12345", "chunks_indexed": 42}`

### 4.6 POST /chat/query (RAG Query)
```python
@app.post("/chat/query")
async def query_chat(session_id: str, question: str, chat_history: List[dict] = []):
    # 1. Load index
    manager = FaissManager(session_id)
    index = manager.load_index()
    retriever = index.as_retriever(search_kwargs={"k": config["retriever"]["top_k"]})
    
    # 2. Initialize RAG
    rag = ConversationalRAG(retriever, ModelLoader().load_llm())
    
    # 3. Query with history
    answer = rag.query(question, chat_history)
    
    return {"answer": answer}
```

**Request:** `{"session_id": "...", "question": "...", "chat_history": [{"role": "user", "content": "..."}, ...]}`
**Response:** `{"answer": "..."}`

---

## 5. Data Flow Diagrams

### 5.1 Document Analysis Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  UploadFile │───▶│  DocHandler │───▶│    PyPDF    │───▶│  Document[] │
│    (PDF)    │    │ save_files()│    │   Loader    │    │  (pages)    │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Metadata   │◀───│   LCEL      │◀───│   Gemini    │◀───│ concat_for_ │
│   (JSON)    │    │   Chain     │    │    LLM      │    │ analysis()  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 5.2 Document Comparison Pipeline

```
┌─────────────┐                           ┌─────────────┐
│  Reference  │                           │   Actual    │
│   Files     │                           │   Files     │
└──────┬──────┘                           └──────┬──────┘
       │                                         │
       ▼                                         ▼
┌─────────────┐                           ┌─────────────┐
│  Document[] │                           │  Document[] │
│  (ref_docs) │                           │  (act_docs) │
└──────┬──────┘                           └──────┬──────┘
       │                                         │
       └─────────────────┬───────────────────────┘
                         ▼
              ┌─────────────────────┐
              │ concat_for_comparison│
              │ <<REFERENCE>>       │
              │ <<ACTUAL>>          │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │ DocumentComparatorLLM│
              │    LCEL Chain       │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  DataFrame          │
              │  [Page, Changes]    │
              └─────────────────────┘
```

### 5.3 Conversational RAG Pipeline

```
                    INDEXING PHASE
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  UploadFile │───▶│ ChatIngestor│───▶│    Split    │
│    (docs)   │    │  load_docs  │    │  (chunks)   │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                                             ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ FaissManager│◀───│   FAISS     │
                   │  save_index │    │  from_docs  │
                   └─────────────┘    └─────────────┘

                     QUERY PHASE
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Question   │───▶│   history   │───▶│contextualize│
│ + History   │    │   aware     │    │   prompt    │
└─────────────┘    │  retriever  │    └──────┬──────┘
                   └─────────────┘           │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Answer    │◀───│   Gemini    │◀───│  Retrieved  │
│   (str)     │    │    LLM      │    │   Context   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 6. Key Logic & Algorithms

### 6.1 LCEL Chain Construction (Document Analysis)

```python
# From data_analysis.py
def build_chain(self):
    parser = JsonOutputParser(pydantic_object=Metadata)
    
    chain = (
        document_analysis_prompt        # ChatPromptTemplate
        | self.llm                      # ChatGoogleGenerativeAI
        | parser                        # JsonOutputParser → Metadata
    )
    return chain
```

### 6.2 History-Aware Retrieval Chain (RAG)

```python
# From retrieval.py
def build_rag_chain(self):
    # Step 1: Contextualize question using chat history
    history_aware_retriever = create_history_aware_retriever(
        self.llm,
        self.retriever,
        contextualize_question_prompt
    )
    
    # Step 2: Build Q&A chain with context
    qa_chain = create_stuff_documents_chain(
        self.llm,
        context_qa_prompt
    )
    
    # Step 3: Combine into RAG chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )
    return rag_chain
```

### 6.3 Document Chunking Strategy

```python
# From data_ingestion.py - ChatIngestor
def split_documents(self, docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)
```

### 6.4 Session-Based FAISS Persistence

```python
# From data_ingestion.py - FaissManager
class FaissManager:
    def __init__(self, session_id: str):
        self.index_path = DATA_DIR / "multi_doc_chat" / session_id / "faiss_index"
    
    def save_index(self, index: FAISS):
        index.save_local(str(self.index_path))
    
    def load_index(self) -> FAISS:
        return FAISS.load_local(
            str(self.index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
```

---

## 7. Prompt Templates

### 7.1 Document Analysis Prompt
```python
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")
```

### 7.2 Document Comparison Prompt
```python
document_comparison_prompt = ChatPromptTemplate.from_template("""
You will be provided with content from two PDFs. Your tasks are as follows:
1. Compare the content in two PDFs
2. Identify the difference in PDF and note down the page number 
3. The output you provide must be page wise comparison content 
4. If any page do not have any change, mention as 'NO CHANGE' 

Input documents:
{combined_docs}

Your response should follow this format:
{format_instruction}
""")
```

### 7.3 RAG Prompts
```python
# Question contextualization
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given conversation history, rewrite query as standalone question..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Q&A with context
context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using provided context. If not found, say 'I don't know.'...\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
```

---

## 8. Pydantic Models

### 8.1 Metadata (Document Analysis Output)
```python
class Metadata(BaseModel):
    Summary: List[str]
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]
    SentimentTone: str
```

### 8.2 ChangeFormat (Comparison Output)
```python
class ChangeFormat(BaseModel):
    Page: str
    Changes: str

class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass
```

---

## 9. Error Handling & Logging

### 9.1 Custom Exception
```python
# exception/custom_exception.py
class DocumentPortalException(Exception):
    def __init__(self, error_message, error_details=None):
        # Extracts file name, line number, full traceback
        self.file_name = ...
        self.lineno = ...
        self.traceback_str = ...
```

### 9.2 Structured Logging
```python
# logger/__init__.py
from .custom_logger import CustomLogger
GLOBAL_LOGGER = CustomLogger().get_logger("doc_portal")

# Usage throughout codebase
from logger import GLOBAL_LOGGER as log
log.info("Loading embedding model", model=model_name)
log.error("Failed to parse API_KEYS", error=str(e))
```

---

## 10. Configuration

### config/config.yaml
```yaml
embedding_model:
  model_name: "models/text-embedding-004"

llm:
  groq:
    provider: "groq"
    model_name: "deepseek-r1-distill-llama-70b"
    temperature: 0.15
    max_output_tokens: 4096
  google:
    provider: "google"
    model_name: "gemini-2.0-flash"
    temperature: 0.2
    max_output_tokens: 4096

retriever:
  top_k: 10
```

### Environment Variables
| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Google Generative AI API key |
| `GROQ_API_KEY` | Groq API key |
| `LLM_PROVIDER` | `"google"` or `"groq"` (default: google) |
| `ENV` | `"local"` or `"production"` |
| `API_KEYS` | JSON object with keys (for ECS deployment) |

---

## 11. Dependency Summary

### Core Dependencies (requirements.txt)
```
langchain==0.3.27
langchain-community==0.3.26
langchain-google-genai==2.2.5
langchain-groq==0.3.5
faiss-cpu==1.11.0
fastapi==0.116.1
uvicorn==0.34.1
PyMuPDF==1.26.3
pydantic==2.11.2
python-dotenv==1.1.0
structlog==25.4.0
```

### Directory Structure
```
document_portal/
├── api/main.py                    # FastAPI entry point
├── config/config.yaml             # Model configuration
├── src/
│   ├── document_analyzer/         # DocumentAnalyzer
│   ├── document_chat/             # ConversationalRAG
│   ├── document_compare/          # DocumentComparatorLLM
│   └── document_ingestion/        # FaissManager, ChatIngestor, DocHandler, DocumentComparator
├── utils/                         # ModelLoader, file I/O, document ops
├── prompt/                        # Prompt templates
├── model/                         # Pydantic models
├── logger/                        # Custom structured logger
├── exception/                     # Custom exception class
├── data/                          # Runtime data storage
├── templates/                     # HTML templates
└── static/                        # CSS/JS assets
```

---

## 12. Summary

**Document Portal** is a modular, production-ready LLM application with:

1. **Clean Separation of Concerns**
   - API layer (`api/main.py`) handles HTTP
   - Service classes handle business logic
   - Utils handle cross-cutting concerns

2. **LangChain LCEL Pipelines**
   - All LLM interactions use composable chains
   - Consistent pattern: `prompt | llm | parser`

3. **Session-Based State Management**
   - FAISS indices stored per session
   - Enables multi-user concurrent usage

4. **Flexible LLM Configuration**
   - Swap between Google Gemini and Groq via env var
   - Model settings in YAML config

5. **Robust Error Handling**
   - Custom exception with traceback extraction
   - Structured logging throughout

6. **Production Deployment**
   - Dockerfile and CloudFormation template included
   - API key management for local & ECS environments
