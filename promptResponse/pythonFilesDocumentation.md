# Python Files Documentation - Document Portal

A comprehensive technical documentation of all Python classes and methods across the specified folders.

---

## 📁 api

### 📄 main.py

> FastAPI application entry point providing REST endpoints for document analysis, comparison, and RAG-based chat.

#### Module-Level Configuration

| Variable | Type | Description |
|----------|------|-------------|
| `FAISS_BASE` | str | Base directory for FAISS indices (default: `faiss_index`) |
| `UPLOAD_BASE` | str | Base directory for uploads (default: `data`) |
| `FAISS_INDEX_NAME` | str | FAISS index file prefix (default: `index`) |
| `app` | FastAPI | Main application instance |
| `BASE_DIR` | Path | Project root directory |
| `templates` | Jinja2Templates | Template engine for HTML rendering |

#### Endpoints (Functions)

| Function Name | Method | Route | Parameters | Returns | Logic | External Methods Used | Called From |
|---------------|--------|-------|------------|---------|-------|----------------------|-------------|
| `serve_ui` | GET | `/` | `request: Request` | `HTMLResponse` | Serves the homepage UI using Jinja2 template | `templates.TemplateResponse()`, `log.info()` | HTTP client |
| `health` | GET | `/health` | None | `Dict[str, str]` | Returns health check status | `log.info()` | HTTP client / Load balancer |
| `analyze_document` | POST | `/analyze` | `file: UploadFile` | `JSONResponse` | Saves PDF, reads content, analyzes metadata using LLM | `DocHandler()`, `dh.save_pdf()`, `read_pdf_via_handler()`, `DocumentAnalyzer()`, `analyzer.analyze_document()` | HTTP client |
| `compare_documents` | POST | `/compare` | `reference: UploadFile, actual: UploadFile` | `dict` | Saves both PDFs, combines text, compares using LLM | `DocumentComparator()`, `dc.save_uploaded_files()`, `dc.combine_documents()`, `DocumentComparatorLLM()`, `comp.compare_documents()` | HTTP client |
| `chat_build_index` | POST | `/chat/index` | `files: List[UploadFile], session_id: Optional[str], use_session_dirs: bool, chunk_size: int, chunk_overlap: int, k: int` | `dict` | Builds FAISS vector index from uploaded documents | `FastAPIFileAdapter()`, `ChatIngestor()`, `ci.built_retriver()` | HTTP client |
| `chat_query` | POST | `/chat/query` | `question: str, session_id: Optional[str], use_session_dirs: bool, k: int` | `dict` | Queries RAG pipeline with conversation history | `os.path.join()`, `ConversationalRAG()`, `rag.load_retriever_from_faiss()`, `rag.invoke()` | HTTP client |

---

## 📁 exception

### 📄 custom_exception.py

> Custom exception class with detailed traceback extraction for error handling across the application.

#### 🏛️ DocumentPortalException

**Inherits from:** `Exception`

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `file_name` | str | Source file where exception occurred |
| `lineno` | int | Line number of the exception |
| `error_message` | str | Normalized error message |
| `traceback_str` | str | Full formatted traceback string |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | `error_message: Any, error_details: Optional[object] = None` | `None` | Normalizes message, extracts exc_info from sys/Exception/context, walks traceback to find last frame, formats full traceback | `sys.exc_info()`, `traceback.format_exception()`, `cast()` | Throughout all modules via `raise DocumentPortalException(...)` |
| `__str__` | public | None | `str` | Returns compact, logger-friendly error message with file, line, and traceback | None | Python exception handling |
| `__repr__` | public | None | `str` | Returns repr format of exception | None | Debug/logging |

---

## 📁 logger

### 📄 custom_logger.py

> Structured JSON logging using structlog with file and console output.

#### 🏛️ CustomLogger

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `logs_dir` | str | Directory path for log files |
| `log_file_path` | str | Full path to timestamped log file |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | `log_dir: str = "logs"` | `None` | Creates logs directory if not exists, generates timestamped log file name | `os.path.join()`, `os.makedirs()`, `datetime.now().strftime()` | `logger/__init__.py` |
| `get_logger` | public | `name: str = __file__` | `structlog.BoundLogger` | Configures file+console handlers, sets up structlog with JSON renderer, timestamps, and log levels | `logging.FileHandler()`, `logging.StreamHandler()`, `logging.basicConfig()`, `structlog.configure()`, `structlog.get_logger()` | `logger/__init__.py` → `GLOBAL_LOGGER` |

---

## 📁 model

### 📄 models.py

> Pydantic data models for structured LLM outputs and API responses.

#### 🏛️ Metadata

**Inherits from:** `pydantic.BaseModel`

**Instance Variables (Fields):**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `Summary` | `List[str]` | Document summary points |
| `Title` | `str` | Document title |
| `Author` | `List[str]` | List of authors |
| `DateCreated` | `str` | Creation date |
| `LastModifiedDate` | `str` | Last modification date |
| `Publisher` | `str` | Publisher name |
| `Language` | `str` | Document language |
| `PageCount` | `Union[int, str]` | Number of pages or "Not Available" |
| `SentimentTone` | `str` | Overall sentiment of document |

---

#### 🏛️ ChangeFormat

**Inherits from:** `pydantic.BaseModel`

**Instance Variables (Fields):**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `Page` | `str` | Page number/identifier |
| `Changes` | `str` | Description of changes on that page |

---

#### 🏛️ SummaryResponse

**Inherits from:** `pydantic.RootModel[list[ChangeFormat]]`

> Wrapper for list of ChangeFormat responses from comparison.

---

#### 🏛️ PromptType

**Inherits from:** `str, Enum`

**Enum Values:**

| Value | Description |
|-------|-------------|
| `DOCUMENT_ANALYSIS` | `"document_analysis"` |
| `DOCUMENT_COMPARISON` | `"document_comparison"` |
| `CONTEXTUALIZE_QUESTION` | `"contextualize_question"` |
| `CONTEXT_QA` | `"context_qa"` |

---

## 📁 src/document_ingestion

### 📄 data_ingestion.py

> Core data ingestion classes for file handling, FAISS indexing, and document processing.

#### 🏛️ FaissManager

**Purpose:** Manages FAISS vector store with idempotent document ingestion and metadata tracking.

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `index_dir` | `Path` | Directory for FAISS index storage |
| `meta_path` | `Path` | Path to ingested metadata JSON |
| `_meta` | `Dict[str, Any]` | In-memory metadata tracking ingested docs |
| `model_loader` | `ModelLoader` | Model loader instance |
| `emb` | `GoogleGenerativeAIEmbeddings` | Embedding model |
| `vs` | `Optional[FAISS]` | FAISS vectorstore instance |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | `index_dir: Path, model_loader: Optional[ModelLoader] = None` | `None` | Creates index directory, loads metadata JSON if exists, initializes embeddings | `Path.mkdir()`, `json.loads()`, `ModelLoader()`, `model_loader.load_embeddings()` | `ChatIngestor.built_retriver()` |
| `_exists` | private | None | `bool` | Checks if FAISS index files exist on disk | `Path.exists()` | `load_or_create()` |
| `_fingerprint` | static | `text: str, md: Dict[str, Any]` | `str` | Generates unique fingerprint for document deduplication | `hashlib.sha256()` | `add_documents()` |
| `_save_meta` | private | None | `None` | Persists metadata dict to JSON file | `Path.write_text()`, `json.dumps()` | `add_documents()` |
| `add_documents` | public | `docs: List[Document]` | `int` | Adds new documents idempotently, returns count added | `self._fingerprint()`, `self.vs.add_documents()`, `self.vs.save_local()`, `self._save_meta()` | `ChatIngestor.built_retriver()` |
| `load_or_create` | public | `texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None` | `FAISS` | Loads existing index or creates new one from texts | `FAISS.load_local()`, `FAISS.from_texts()`, `self.vs.save_local()` | `ChatIngestor.built_retriver()` |

---

#### 🏛️ ChatIngestor

**Purpose:** Orchestrates document ingestion pipeline for RAG chat functionality.

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `model_loader` | `ModelLoader` | Model loader instance |
| `use_session` | `bool` | Whether to use session-based directories |
| `session_id` | `str` | Unique session identifier |
| `temp_base` | `Path` | Base directory for temp files |
| `faiss_base` | `Path` | Base directory for FAISS indices |
| `temp_dir` | `Path` | Resolved temp directory path |
| `faiss_dir` | `Path` | Resolved FAISS directory path |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | `temp_base: str = "data", faiss_base: str = "faiss_index", use_session_dirs: bool = True, session_id: Optional[str] = None` | `None` | Initializes directories and session, creates ModelLoader | `ModelLoader()`, `generate_session_id()`, `Path.mkdir()`, `self._resolve_dir()`, `log.info()` | `api/main.py: chat_build_index()` |
| `_resolve_dir` | private | `base: Path` | `Path` | Resolves directory path based on session mode | `Path.mkdir()` | `__init__()` |
| `_split` | private | `docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200` | `List[Document]` | Splits documents into chunks using recursive splitter | `RecursiveCharacterTextSplitter()`, `splitter.split_documents()`, `log.info()` | `built_retriver()` |
| `built_retriver` | public | `uploaded_files: Iterable, *, chunk_size: int = 1000, chunk_overlap: int = 200, k: int = 5` | `VectorStoreRetriever` | Full pipeline: save files → load docs → split → create/update FAISS index → return retriever | `save_uploaded_files()`, `load_documents()`, `self._split()`, `FaissManager()`, `fm.load_or_create()`, `fm.add_documents()`, `vs.as_retriever()` | `api/main.py: chat_build_index()` |

---

#### 🏛️ DocHandler

**Purpose:** Handles PDF saving and reading for document analysis workflow.

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `data_dir` | `str` | Base directory for document storage |
| `session_id` | `str` | Unique session identifier |
| `session_path` | `str` | Full path to session directory |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | `data_dir: Optional[str] = None, session_id: Optional[str] = None` | `None` | Initializes session directory for PDF storage | `os.getenv()`, `os.path.join()`, `os.makedirs()`, `generate_session_id()`, `log.info()` | `api/main.py: analyze_document()` |
| `save_pdf` | public | `uploaded_file` | `str` | Saves uploaded PDF to session directory | `os.path.basename()`, `os.path.join()`, `open()`, `uploaded_file.read()/.getbuffer()`, `log.info()` | `api/main.py: analyze_document()` |
| `read_pdf` | public | `pdf_path: str` | `str` | Reads PDF content page-by-page using PyMuPDF | `fitz.open()`, `doc.load_page()`, `page.get_text()`, `log.info()` | `api/main.py: read_pdf_via_handler()` |

---

#### 🏛️ DocumentComparator

**Purpose:** Handles PDF comparison workflow with session-based versioning.

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `base_dir` | `Path` | Base directory for comparisons |
| `session_id` | `str` | Unique session identifier |
| `session_path` | `Path` | Full path to session directory |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | `base_dir: str = "data/document_compare", session_id: Optional[str] = None` | `None` | Initializes session directory for PDF comparison | `Path()`, `generate_session_id()`, `Path.mkdir()`, `log.info()` | `api/main.py: compare_documents()` |
| `save_uploaded_files` | public | `reference_file, actual_file` | `Tuple[Path, Path]` | Saves reference and actual PDF files | `Path()`, `open()`, `fobj.read()/.getbuffer()`, `log.info()` | `api/main.py: compare_documents()` |
| `read_pdf` | public | `pdf_path: Path` | `str` | Reads PDF content with encryption check | `fitz.open()`, `doc.is_encrypted`, `doc.load_page()`, `page.get_text()`, `log.info()` | `combine_documents()` |
| `combine_documents` | public | None | `str` | Combines all PDFs in session directory | `Path.iterdir()`, `self.read_pdf()`, `log.info()` | `api/main.py: compare_documents()` |
| `clean_old_sessions` | public | `keep_latest: int = 3` | `None` | Removes old session directories | `Path.iterdir()`, `sorted()`, `shutil.rmtree()`, `log.info()` | Maintenance task |

---

## 📁 src/document_analyzer

### 📄 data_analysis.py

> LLM-based document analysis for metadata extraction.

#### 🏛️ DocumentAnalyzer

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `loader` | `ModelLoader` | Model loader instance |
| `llm` | `ChatGoogleGenerativeAI/ChatGroq` | LLM model instance |
| `parser` | `JsonOutputParser` | JSON output parser with Metadata schema |
| `fixing_parser` | `OutputFixingParser` | Auto-fixing parser for malformed JSON |
| `prompt` | `ChatPromptTemplate` | Document analysis prompt template |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | None | `None` | Initializes LLM, parsers, and prompt template | `ModelLoader()`, `loader.load_llm()`, `JsonOutputParser()`, `OutputFixingParser.from_llm()`, `PROMPT_REGISTRY[]`, `log.info()` | `api/main.py: analyze_document()` |
| `analyze_document` | public | `document_text: str` | `dict` | Analyzes document text and extracts structured metadata | `self.prompt \| self.llm \| self.fixing_parser` (LCEL chain), `chain.invoke()`, `self.parser.get_format_instructions()`, `log.info()` | `api/main.py: analyze_document()` |

---

## 📁 src/document_compare

### 📄 document_comparator.py

> LLM-based document comparison with page-wise difference detection.

#### 🏛️ DocumentComparatorLLM

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `loader` | `ModelLoader` | Model loader instance |
| `llm` | `ChatGoogleGenerativeAI/ChatGroq` | LLM model instance |
| `parser` | `JsonOutputParser` | JSON output parser with SummaryResponse schema |
| `fixing_parser` | `OutputFixingParser` | Auto-fixing parser for malformed JSON |
| `prompt` | `ChatPromptTemplate` | Document comparison prompt template |
| `chain` | `RunnableSequence` | LCEL chain: prompt → llm → parser |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | None | `None` | Initializes LLM, parsers, prompt, and builds LCEL chain | `load_dotenv()`, `ModelLoader()`, `loader.load_llm()`, `JsonOutputParser()`, `OutputFixingParser.from_llm()`, `PROMPT_REGISTRY[]`, `log.info()` | `api/main.py: compare_documents()` |
| `compare_documents` | public | `combined_docs: str` | `pd.DataFrame` | Invokes LLM chain to compare documents and returns DataFrame | `self.chain.invoke()`, `self.parser.get_format_instructions()`, `self._format_response()`, `log.info()` | `api/main.py: compare_documents()` |
| `_format_response` | private | `response_parsed: list[dict]` | `pd.DataFrame` | Converts parsed response to pandas DataFrame | `pd.DataFrame()`, `log.error()` | `compare_documents()` |

---

## 📁 src/document_chat

### 📄 retrieval.py

> LCEL-based Conversational RAG with history-aware retrieval.

#### 🏛️ ConversationalRAG

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `session_id` | `Optional[str]` | Session identifier for tracking |
| `llm` | `ChatGoogleGenerativeAI/ChatGroq` | LLM model instance |
| `contextualize_prompt` | `ChatPromptTemplate` | Prompt for question rewriting |
| `qa_prompt` | `ChatPromptTemplate` | Prompt for Q&A with context |
| `retriever` | `VectorStoreRetriever` | FAISS retriever instance |
| `chain` | `RunnableSequence` | Full LCEL RAG chain |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | `session_id: Optional[str], retriever=None` | `None` | Initializes LLM, prompts; builds chain if retriever provided | `self._load_llm()`, `PROMPT_REGISTRY[]`, `self._build_lcel_chain()`, `log.info()` | `api/main.py: chat_query()` |
| `load_retriever_from_faiss` | public | `index_path: str, k: int = 5, index_name: str = "index", search_type: str = "similarity", search_kwargs: Optional[Dict] = None` | `VectorStoreRetriever` | Loads FAISS index from disk, creates retriever, builds chain | `os.path.isdir()`, `ModelLoader().load_embeddings()`, `FAISS.load_local()`, `vectorstore.as_retriever()`, `self._build_lcel_chain()`, `log.info()` | `api/main.py: chat_query()` |
| `invoke` | public | `user_input: str, chat_history: Optional[List[BaseMessage]] = None` | `str` | Executes RAG pipeline and returns answer | `self.chain.invoke()`, `log.info()`, `log.warning()` | `api/main.py: chat_query()` |
| `_load_llm` | private | None | `LLM` | Loads LLM model via ModelLoader | `ModelLoader().load_llm()`, `log.info()` | `__init__()` |
| `_format_docs` | static | `docs` | `str` | Formats retrieved documents into single string | `getattr()`, `"\n\n".join()` | `_build_lcel_chain()` |
| `_build_lcel_chain` | private | None | `None` | Builds complete LCEL chain: question rewriter → retriever → formatter → QA | `itemgetter()`, LCEL pipe operators, `StrOutputParser()`, `log.info()` | `__init__()`, `load_retriever_from_faiss()` |

---

## 📁 utils

### 📄 model_loader.py

> Model and API key management for LLM and embedding initialization.

#### 🏛️ ApiKeyManager

**Class Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `REQUIRED_KEYS` | `List[str]` | `["GROQ_API_KEY", "GOOGLE_API_KEY"]` |

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `api_keys` | `dict` | Dictionary of loaded API keys |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | None | `None` | Loads API keys from `API_KEYS` JSON env var or individual env vars | `os.getenv()`, `json.loads()`, `log.info()`, `log.warning()`, `log.error()` | `ModelLoader.__init__()` |
| `get` | public | `key: str` | `str` | Retrieves specific API key | `dict.get()` | `ModelLoader.load_embeddings()`, `ModelLoader.load_llm()` |

---

#### 🏛️ ModelLoader

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `api_key_mgr` | `ApiKeyManager` | API key manager instance |
| `config` | `dict` | Loaded YAML configuration |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | None | `None` | Loads .env in local mode, initializes ApiKeyManager and config | `os.getenv()`, `load_dotenv()`, `ApiKeyManager()`, `load_config()`, `log.info()` | Throughout all service classes |
| `load_embeddings` | public | None | `GoogleGenerativeAIEmbeddings` | Loads Google embedding model from config | `self.config[]`, `GoogleGenerativeAIEmbeddings()`, `self.api_key_mgr.get()`, `log.info()` | `FaissManager`, `ConversationalRAG` |
| `load_llm` | public | None | `ChatGoogleGenerativeAI` or `ChatGroq` | Loads LLM based on provider config and env var | `self.config[]`, `os.getenv()`, `ChatGoogleGenerativeAI()`, `ChatGroq()`, `self.api_key_mgr.get()`, `log.info()` | `DocumentAnalyzer`, `DocumentComparatorLLM`, `ConversationalRAG` |

---

### 📄 config_loader.py

> YAML configuration file loader with flexible path resolution.

#### Functions

| Function Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|---------------|--------|------------|---------|-------|----------------------|-------------|
| `_project_root` | private | None | `Path` | Resolves project root directory from file location | `Path(__file__).resolve().parents[1]` | `load_config()` |
| `load_config` | public | `config_path: str \| None = None` | `dict` | Loads YAML config with priority: arg > env > default path | `os.getenv()`, `Path()`, `Path.is_absolute()`, `Path.exists()`, `open()`, `yaml.safe_load()` | `ModelLoader.__init__()` |

---

### 📄 file_io.py

> File I/O utilities for session management and file uploads.

#### Module Constants

| Variable | Type | Description |
|----------|------|-------------|
| `SUPPORTED_EXTENSIONS` | `set` | `{".pdf", ".docx", ".txt"}` |

#### Functions

| Function Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|---------------|--------|------------|---------|-------|----------------------|-------------|
| `generate_session_id` | public | `prefix: str = "session"` | `str` | Generates unique timestamped session ID | `ZoneInfo()`, `datetime.now()`, `uuid.uuid4()` | `ChatIngestor`, `DocHandler`, `DocumentComparator` |
| `save_uploaded_files` | public | `uploaded_files: Iterable, target_dir: Path` | `List[Path]` | Saves uploaded files with sanitized names, skips unsupported | `Path.mkdir()`, `re.sub()`, `uuid.uuid4()`, `open()`, `uf.read()/.getbuffer()`, `log.info()`, `log.warning()` | `ChatIngestor.built_retriver()` |

---

### 📄 document_ops.py

> Document loading and text processing utilities.

#### Module Constants

| Variable | Type | Description |
|----------|------|-------------|
| `SUPPORTED_EXTENSIONS` | `set` | `{".pdf", ".docx", ".txt"}` |

#### Functions

| Function Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|---------------|--------|------------|---------|-------|----------------------|-------------|
| `load_documents` | public | `paths: Iterable[Path]` | `List[Document]` | Loads documents using appropriate LangChain loader | `PyPDFLoader()`, `Docx2txtLoader()`, `TextLoader()`, `loader.load()`, `log.info()`, `log.warning()` | `ChatIngestor.built_retriver()` |
| `concat_for_analysis` | public | `docs: List[Document]` | `str` | Concatenates documents with source markers | `d.metadata.get()`, `"\n".join()` | `ChatIngestor` |
| `concat_for_comparison` | public | `ref_docs: List[Document], act_docs: List[Document]` | `str` | Concatenates reference and actual docs with markers | `concat_for_analysis()` | `api/main.py` |
| `read_pdf_via_handler` | public | `handler, path: str` | `str` | Reads PDF using handler's read method | `hasattr()`, `handler.read_pdf()` | `api/main.py: analyze_document()` |

---

#### 🏛️ FastAPIFileAdapter

**Purpose:** Adapts FastAPI UploadFile to Streamlit-like interface for file handling.

**Instance Variables:**

| Variable Name | Type | Description |
|---------------|------|-------------|
| `_uf` | `UploadFile` | Original FastAPI UploadFile |
| `name` | `str` | Filename from UploadFile |

**Methods:**

| Method Name | Access | Parameters | Returns | Logic | External Methods Used | Called From |
|-------------|--------|------------|---------|-------|----------------------|-------------|
| `__init__` | public | `uf: UploadFile` | `None` | Stores UploadFile reference and extracts filename | None | `api/main.py` endpoints |
| `getbuffer` | public | None | `bytes` | Returns file content as bytes | `self._uf.file.seek()`, `self._uf.file.read()` | File save operations |

---

### 📄 test_utils.py

> Empty file (placeholder for test utilities)

---

## Cross-Module Dependencies Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              api/main.py                                     │
│                                  │                                           │
│    ┌───────────────────────┬─────┴─────┬───────────────────────┐            │
│    ▼                       ▼           ▼                       ▼            │
│ DocHandler          ChatIngestor   DocumentComparator   ConversationalRAG   │
│    │                       │               │                   │            │
│    ▼                       ▼               ▼                   ▼            │
│ DocumentAnalyzer     FaissManager   DocumentComparatorLLM  FAISS.load_local │
└─────────────────────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
      ModelLoader                    PROMPT_REGISTRY
           │                               │
    ┌──────┴──────┐                        │
    ▼             ▼                        ▼
ApiKeyManager  load_config          prompt_library.py
    │             │
    ▼             ▼
 os.getenv   yaml.safe_load
```

---

## Summary Statistics

| Folder | Files | Classes | Methods | Functions |
|--------|-------|---------|---------|-----------|
| api | 1 | 0 | 6 | 0 |
| exception | 1 | 1 | 3 | 0 |
| logger | 1 | 1 | 2 | 0 |
| model | 1 | 4 | 0 | 0 |
| src/document_ingestion | 1 | 4 | 15 | 0 |
| src/document_analyzer | 1 | 1 | 2 | 0 |
| src/document_compare | 1 | 1 | 3 | 0 |
| src/document_chat | 1 | 1 | 6 | 0 |
| utils | 4 | 3 | 6 | 6 |
| **Total** | **12** | **16** | **43** | **6** |

---

*Generated: February 2026*
*Project: Document Portal - FastAPI LLM Application*
