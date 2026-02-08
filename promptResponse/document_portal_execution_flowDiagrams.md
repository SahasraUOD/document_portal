# Document Portal - Execution Flows & Flow Diagrams

## Step-by-Step Execution Flow (Plain English)

---

### Flow 1: Document Analysis (`/analyze`)

1. The entry point is `api/main.py`. When a user sends a POST request to `/analyze` with a PDF file, FastAPI receives the request and calls the `analyze_document()` endpoint function.

2. The endpoint function receives the uploaded file. It creates a `FastAPIFileAdapter` to wrap the FastAPI UploadFile object so it has a `.name` and `.getbuffer()` interface that other modules expect.

3. It then creates a new `DocHandler` instance. When DocHandler is created, it generates a unique session ID using `generate_session_id()` which combines a timestamp and UUID. It creates a session folder at `data/document_analysis/{session_id}/`.

4. The endpoint calls `dh.save_pdf()` and passes the wrapped file adapter. The `save_pdf()` method extracts the filename, validates it ends with `.pdf`, then writes the file bytes to the session folder. It returns the full path where the file was saved.

5. Next, the endpoint calls `read_pdf_via_handler()` and passes the DocHandler and the saved path. This helper function calls `dh.read_pdf()` internally.

6. The `read_pdf()` method opens the PDF using PyMuPDF (fitz). It loops through each page, extracts the text using `page.get_text()`, and adds page markers like "--- Page 1 ---". It joins all page texts together and returns the complete document text.

7. The endpoint now has the document text. It creates a `DocumentAnalyzer` instance. When DocumentAnalyzer is created, it initializes a `ModelLoader`. The ModelLoader loads the `.env` file (if running locally), creates an `ApiKeyManager` to get API keys, and loads the YAML config from `config/config.yaml`.

8. DocumentAnalyzer calls `self.loader.load_llm()`. The ModelLoader checks the `LLM_PROVIDER` environment variable (defaults to "google"), looks up the config for that provider, and creates either a `ChatGoogleGenerativeAI` or `ChatGroq` LLM instance with the configured model name and temperature.

9. DocumentAnalyzer also sets up a `JsonOutputParser` with the `Metadata` Pydantic model as the schema. It wraps this in an `OutputFixingParser` which uses the LLM to fix malformed JSON if needed. It gets the analysis prompt from `PROMPT_REGISTRY["document_analysis"]`.

10. The endpoint calls `analyzer.analyze_document()` and passes the document text. This method creates a LangChain chain: `prompt | llm | fixing_parser`.

11. The chain is invoked with the document text and format instructions. The prompt tells the LLM to analyze the document and return JSON with fields like Title, Author, Summary, DateCreated, SentimentTone, etc.

12. The LLM processes the document and returns a JSON response. The parser validates it against the `Metadata` schema. If the JSON is malformed, the OutputFixingParser asks the LLM to fix it.

13. The `analyze_document()` method returns the parsed dictionary. The endpoint wraps this in a `JSONResponse` and returns it to the user.

**Flow 1 Diagram:**
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              POST /analyze (PDF file)                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  api/main.py: analyze_document()                                                         │
│  ├── FastAPIFileAdapter(file)  ──────────────────────────────────────────────────────────│
│  │                                                                                       │
│  ├── DocHandler()                                                                        │
│  │   └── generate_session_id() → session_id                                              │
│  │   └── mkdir(data/document_analysis/{session_id}/)                                     │
│  │                                                                                       │
│  ├── dh.save_pdf(wrapped_file)                                                           │
│  │   └── validate .pdf extension                                                         │
│  │   └── write bytes to session folder                                                   │
│  │   └── return saved_path                                                               │
│  │                                                                                       │
│  ├── read_pdf_via_handler(dh, saved_path)                                                │
│  │   └── dh.read_pdf(path)                                                               │
│  │       └── fitz.open(path)                                                             │
│  │       └── for each page: get_text()                                                   │
│  │       └── join with "--- Page N ---" markers                                          │
│  │       └── return document_text                                                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  DocumentAnalyzer()                                                                      │
│  ├── ModelLoader()                                                                       │
│  │   ├── load_dotenv() (if local)                                                        │
│  │   ├── ApiKeyManager() → api_keys                                                      │
│  │   └── load_config() → config.yaml                                                     │
│  │                                                                                       │
│  ├── loader.load_llm()                                                                   │
│  │   └── check LLM_PROVIDER env                                                          │
│  │   └── return ChatGoogleGenerativeAI or ChatGroq                                       │
│  │                                                                                       │
│  ├── JsonOutputParser(pydantic_object=Metadata)                                          │
│  ├── OutputFixingParser.from_llm(parser, llm)                                            │
│  └── PROMPT_REGISTRY["document_analysis"]                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  analyzer.analyze_document(document_text)                                                │
│  ├── chain = prompt | llm | fixing_parser                                                │
│  ├── chain.invoke({format_instructions, document_text})                                  │
│  │   └── LLM analyzes document                                                           │
│  │   └── Returns JSON: {Title, Author, Summary, DateCreated, SentimentTone...}           │
│  │   └── Parser validates against Metadata schema                                        │
│  │   └── OutputFixingParser fixes malformed JSON if needed                               │
│  └── return parsed dict                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  JSONResponse(result)  →  Return to User                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Flow 2: Document Comparison (`/compare`)

1. The entry point is `api/main.py`. When a user sends a POST request to `/compare` with two PDF files (named `reference` and `actual`), FastAPI calls the `compare_documents()` endpoint function.

2. The endpoint receives both uploaded files. It wraps each one in a `FastAPIFileAdapter`.

3. It creates a `DocumentComparator` instance. When created, this generates a session ID and creates a session folder at `data/document_compare/{session_id}/`.

4. The endpoint calls `dc.save_uploaded_files()` and passes both wrapped file adapters. This method validates both files are PDFs, then writes each to the session folder. It returns the paths to both saved files.

5. The endpoint calls `dc.combine_documents()`. This method iterates through all PDF files in the session folder. For each PDF, it calls `self.read_pdf()`.

6. The `read_pdf()` method opens the PDF with PyMuPDF, checks if it's encrypted (raises error if so), extracts text from each page with page markers, and returns the combined text.

7. `combine_documents()` collects the text from each PDF, prefixes each with "Document: {filename}", and joins them with newlines. It returns the combined text of all documents.

8. The endpoint creates a `DocumentComparatorLLM` instance. When created, this initializes a `ModelLoader`, loads the LLM, sets up a `JsonOutputParser` with the `SummaryResponse` schema (which is a list of `ChangeFormat` objects with Page and Changes fields), and gets the comparison prompt from the registry.

9. It builds the chain: `prompt | llm | parser`.

10. The endpoint calls `comp.compare_documents()` and passes the combined text. This method invokes the chain with the combined documents and format instructions.

11. The LLM compares the two documents page by page. For each page, it identifies differences or marks "NO CHANGE". It returns a JSON list of page comparisons.

12. The parser validates the response. The `_format_response()` method converts the list of dictionaries into a pandas DataFrame with columns Page and Changes.

13. The endpoint returns a JSON response containing the DataFrame rows as records and the session ID.

**Flow 2 Diagram:**
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     POST /compare (reference PDF + actual PDF)                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  api/main.py: compare_documents()                                                        │
│  ├── FastAPIFileAdapter(reference)                                                       │
│  ├── FastAPIFileAdapter(actual)                                                          │
│  │                                                                                       │
│  ├── DocumentComparator()                                                                │
│  │   └── generate_session_id() → session_id                                              │
│  │   └── mkdir(data/document_compare/{session_id}/)                                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  dc.save_uploaded_files(ref_adapter, act_adapter)                                        │
│  ├── validate both are .pdf                                                              │
│  ├── write reference to session_path/reference.pdf                                       │
│  ├── write actual to session_path/actual.pdf                                             │
│  └── return (ref_path, act_path)                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  dc.combine_documents()                                                                  │
│  ├── for each .pdf in session_path:                                                      │
│  │   └── self.read_pdf(file)                                                             │
│  │       ├── fitz.open(file)                                                             │
│  │       ├── check not encrypted                                                         │
│  │       ├── for each page: get_text()                                                   │
│  │       └── join with "--- Page N ---" markers                                          │
│  │                                                                                       │
│  ├── prefix each: "Document: {filename}\n{content}"                                      │
│  └── return combined_text                                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  DocumentComparatorLLM()                                                                 │
│  ├── ModelLoader() → load_llm() → LLM                                                    │
│  ├── JsonOutputParser(pydantic_object=SummaryResponse)                                   │
│  │   └── SummaryResponse = list[ChangeFormat{Page, Changes}]                             │
│  ├── OutputFixingParser.from_llm(parser, llm)                                            │
│  ├── PROMPT_REGISTRY["document_comparison"]                                              │
│  └── chain = prompt | llm | parser                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  comp.compare_documents(combined_text)                                                   │
│  ├── chain.invoke({combined_docs, format_instruction})                                   │
│  │   └── LLM compares page by page                                                       │
│  │   └── Returns JSON: [{Page: "1", Changes: "..."}, {Page: "2", Changes: "NO CHANGE"}]  │
│  │                                                                                       │
│  ├── _format_response(response)                                                          │
│  │   └── pd.DataFrame(response) → columns: Page, Changes                                 │
│  └── return DataFrame                                                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Return JSON: {"rows": df.to_dict("records"), "session_id": session_id}                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Flow 3: Chat Index Building (`/chat/index`)

1. The entry point is `api/main.py`. When a user sends a POST request to `/chat/index` with one or more files and optional parameters (session_id, chunk_size, chunk_overlap, k), FastAPI calls the `chat_build_index()` endpoint.

2. The endpoint receives the files and parameters. It wraps each file in a `FastAPIFileAdapter`.

3. It creates a `ChatIngestor` instance and passes the base directories and session settings. When created, ChatIngestor initializes a `ModelLoader`, generates a session ID if not provided, and creates directories: `data/{session_id}/` for temp files and `faiss_index/{session_id}/` for the vector index.

4. The endpoint calls `ci.built_retriver()` and passes the wrapped files, chunk_size (default 1000), chunk_overlap (default 200), and k (default 5).

5. `built_retriver()` first calls `save_uploaded_files()` from `utils/file_io.py`. This function iterates through the files, validates extensions (.pdf, .docx, .txt), generates safe filenames with UUIDs, and writes each file to the temp directory. It returns a list of saved file paths.

6. Next, it calls `load_documents()` from `utils/document_ops.py`. This function iterates through the paths. Based on the file extension, it creates the appropriate loader: `PyPDFLoader` for PDFs, `Docx2txtLoader` for Word docs, or `TextLoader` for text files. It calls `loader.load()` to get LangChain Document objects and collects them all.

7. If no documents were loaded, it raises an error. Otherwise, it calls `self._split()` and passes the documents with chunk settings.

8. The `_split()` method creates a `RecursiveCharacterTextSplitter` with the specified chunk_size and overlap. It calls `splitter.split_documents()` which breaks each document into smaller chunks. Each chunk is a new Document with the text and inherited metadata.

9. `built_retriver()` creates a `FaissManager` instance and passes the FAISS directory and the ModelLoader.

10. When FaissManager is created, it creates the index directory if needed. It checks for an existing metadata file (`ingested_meta.json`) and loads it if present. It calls `self.model_loader.load_embeddings()`.

11. `load_embeddings()` checks the config for the embedding provider. If "google", it creates `GoogleGenerativeAIEmbeddings` with the model name from config. If "huggingface", it creates `HuggingFaceEmbeddings`. It returns the embeddings instance.

12. Back in `built_retriver()`, it extracts the page_content and metadata from each chunk into separate lists.

13. It calls `fm.load_or_create()` and passes the texts and metadatas.

14. `load_or_create()` first checks if `index.faiss` and `index.pkl` files exist in the directory. If they do, it calls `FAISS.load_local()` to load the existing index with the embeddings, and returns the vectorstore.

15. If no index exists, it calls `FAISS.from_texts()` which embeds all the texts using the embeddings model and creates a new FAISS index. It then calls `save_local()` to persist the index to disk.

16. Back in `built_retriver()`, it calls `fm.add_documents()` and passes the document chunks.

17. `add_documents()` iterates through each document. For each one, it calls `_fingerprint()` which generates a unique key from the source path and row_id (or a hash of the content). If the fingerprint already exists in the metadata (meaning this doc was already indexed), it skips it. Otherwise, it adds the doc to a list of new documents and records the fingerprint.

18. If there are new documents, it calls `self.vs.add_documents()` to embed them and add to the FAISS index. It then saves the index and metadata to disk.

19. `built_retriver()` calls `vs.as_retriever()` with search_type="similarity" and k parameter. This creates a retriever that will return the k most similar documents for any query.

20. The endpoint returns a JSON response with the session_id, k value, and use_session_dirs flag. The client will use the session_id for future queries.

**Flow 3 Diagram:**
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│          POST /chat/index (files[], session_id?, chunk_size?, chunk_overlap?, k?)        │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  api/main.py: chat_build_index()                                                         │
│  ├── for each file: FastAPIFileAdapter(file)                                             │
│  │                                                                                       │
│  ├── ChatIngestor(temp_base, faiss_base, use_session_dirs, session_id)                   │
│  │   ├── ModelLoader()                                                                   │
│  │   ├── generate_session_id() if not provided                                           │
│  │   ├── mkdir(data/{session_id}/)                                                       │
│  │   └── mkdir(faiss_index/{session_id}/)                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  ci.built_retriver(wrapped_files, chunk_size, chunk_overlap, k)                          │
│  │                                                                                       │
│  ├── save_uploaded_files(files, temp_dir)  [utils/file_io.py]                            │
│  │   ├── for each file:                                                                  │
│  │   │   ├── validate extension (.pdf, .docx, .txt)                                      │
│  │   │   ├── generate safe filename with UUID                                            │
│  │   │   └── write to temp_dir                                                           │
│  │   └── return List[Path]                                                               │
│  │                                                                                       │
│  ├── load_documents(paths)  [utils/document_ops.py]                                      │
│  │   ├── for each path:                                                                  │
│  │   │   ├── .pdf → PyPDFLoader                                                          │
│  │   │   ├── .docx → Docx2txtLoader                                                      │
│  │   │   └── .txt → TextLoader                                                           │
│  │   ├── loader.load() → Document objects                                                │
│  │   └── return List[Document]                                                           │
│  │                                                                                       │
│  ├── self._split(docs, chunk_size, chunk_overlap)                                        │
│  │   ├── RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)                       │
│  │   ├── splitter.split_documents(docs)                                                  │
│  │   └── return List[Document] (chunks)                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  FaissManager(faiss_dir, model_loader)                                                   │
│  ├── mkdir(index_dir)                                                                    │
│  ├── load ingested_meta.json if exists                                                   │
│  │                                                                                       │
│  ├── model_loader.load_embeddings()                                                      │
│  │   ├── check config: provider = "google" or "huggingface"                              │
│  │   ├── google → GoogleGenerativeAIEmbeddings(model_name)                               │
│  │   └── huggingface → HuggingFaceEmbeddings(model_name)                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  fm.load_or_create(texts, metadatas)                                                     │
│  │                                                                                       │
│  ├── if index.faiss + index.pkl exist:                                                   │
│  │   └── FAISS.load_local(index_dir, embeddings) → return vectorstore                    │
│  │                                                                                       │
│  └── else (new index):                                                                   │
│      ├── FAISS.from_texts(texts, embeddings, metadatas)                                  │
│      │   └── embed all texts → create FAISS index                                        │
│      ├── vs.save_local(index_dir)                                                        │
│      └── return vectorstore                                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  fm.add_documents(chunks)                                                                │
│  ├── for each chunk:                                                                     │
│  │   ├── _fingerprint(text, metadata) → unique key                                       │
│  │   ├── if key in meta["rows"]: skip (already indexed)                                  │
│  │   └── else: add to new_docs, record fingerprint                                       │
│  │                                                                                       │
│  ├── if new_docs:                                                                        │
│  │   ├── vs.add_documents(new_docs) → embed & add to index                               │
│  │   ├── vs.save_local(index_dir)                                                        │
│  │   └── save_meta()                                                                     │
│  └── return count of added docs                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  vs.as_retriever(search_type="similarity", search_kwargs={"k": k})                       │
│  └── return Retriever (ready for queries)                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Return JSON: {"session_id": session_id, "k": k, "use_session_dirs": True}               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Flow 4: Chat Query (`/chat/query`)

1. The entry point is `api/main.py`. When a user sends a POST request to `/chat/query` with a question, session_id, and optional parameters, FastAPI calls the `chat_query()` endpoint.

2. The endpoint validates that if `use_session_dirs` is True, a session_id must be provided. It constructs the index directory path: `faiss_index/{session_id}/`.

3. It checks if the directory exists. If not, it raises a 404 error telling the user the FAISS index was not found.

4. It creates a `ConversationalRAG` instance and passes the session_id. When created, ConversationalRAG calls `self._load_llm()`.

5. `_load_llm()` creates a `ModelLoader` and calls `load_llm()`. This returns the configured LLM (Google Gemini or Groq).

6. ConversationalRAG loads two prompts from the registry: `contextualize_question` (for rewriting questions with chat history context) and `context_qa` (for answering based on retrieved context).

7. At this point, the retriever and chain are None (lazy initialization).

8. The endpoint calls `rag.load_retriever_from_faiss()` and passes the index directory path, k value, and index name ("index").

9. `load_retriever_from_faiss()` first validates the directory exists. It creates a new `ModelLoader` and calls `load_embeddings()` to get the same embedding model used during indexing.

10. It calls `FAISS.load_local()` with the index path, embeddings, and index_name. This loads the persisted FAISS index from disk. The `allow_dangerous_deserialization=True` flag is needed because FAISS uses pickle.

11. It calls `vectorstore.as_retriever()` with the k parameter to create a retriever.

12. It stores the retriever and calls `self._build_lcel_chain()`.

13. `_build_lcel_chain()` constructs the LCEL (LangChain Expression Language) pipeline in three parts:

14. First, it builds a question rewriter: This takes the user input and chat history, passes them to the contextualize_prompt, sends to the LLM, and parses the output to a string. This reformulates the question to be standalone if there's chat context.

15. Second, it builds the retrieval step: The rewritten question goes to the retriever, which embeds it and finds the k most similar document chunks. The `_format_docs()` static method joins the chunk contents with newlines.

16. Third, it builds the full chain: The context (from retrieval), original input, and chat history go into the qa_prompt. This goes to the LLM. The output is parsed to a string.

17. Now `load_retriever_from_faiss()` returns the retriever, and the chain is ready.

18. The endpoint calls `rag.invoke()` and passes the user's question with an empty chat history.

19. `invoke()` validates the chain exists. It creates a payload with the input and chat_history. It calls `self.chain.invoke()` with the payload.

20. The LCEL chain executes: First, the question rewriter runs (but with empty history, it returns the question unchanged). Then the retriever finds similar documents. Then the QA prompt is filled with the context and question. The LLM generates an answer based on the retrieved context. The string parser extracts the text.

21. If the answer is empty, it returns "no answer generated." Otherwise, it returns the answer string.

22. The endpoint returns a JSON response containing the answer, session_id, k value, and engine name ("LCEL-RAG").

**Flow 4 Diagram:**
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│             POST /chat/query (question, session_id, use_session_dirs?, k?)               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  api/main.py: chat_query()                                                               │
│  ├── validate: session_id required if use_session_dirs=True                              │
│  ├── index_dir = faiss_index/{session_id}/                                               │
│  └── check index_dir exists → 404 if not found                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  ConversationalRAG(session_id)                                                           │
│  ├── _load_llm()                                                                         │
│  │   └── ModelLoader().load_llm() → ChatGoogleGenerativeAI or ChatGroq                   │
│  │                                                                                       │
│  ├── PROMPT_REGISTRY["contextualize_question"] → contextualize_prompt                    │
│  ├── PROMPT_REGISTRY["context_qa"] → qa_prompt                                           │
│  │                                                                                       │
│  └── retriever = None, chain = None (lazy init)                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  rag.load_retriever_from_faiss(index_dir, k, index_name="index")                         │
│  │                                                                                       │
│  ├── validate index_dir exists                                                           │
│  │                                                                                       │
│  ├── ModelLoader().load_embeddings() → embeddings                                        │
│  │                                                                                       │
│  ├── FAISS.load_local(index_dir, embeddings, index_name, allow_dangerous=True)           │
│  │   └── loads index.faiss + index.pkl from disk                                         │
│  │                                                                                       │
│  ├── vectorstore.as_retriever(search_kwargs={"k": k})                                    │
│  │   └── self.retriever = retriever                                                      │
│  │                                                                                       │
│  └── _build_lcel_chain()                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  _build_lcel_chain()                                                                     │
│  │                                                                                       │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐    │
│  │  │  STEP 1: Question Rewriter                                                    │    │
│  │  │  {"input": input, "chat_history": history}                                    │    │
│  │  │          │                                                                    │    │
│  │  │          ▼                                                                    │    │
│  │  │  contextualize_prompt                                                         │    │
│  │  │          │                                                                    │    │
│  │  │          ▼                                                                    │    │
│  │  │        LLM  → "Reformulate question to be standalone"                         │    │
│  │  │          │                                                                    │    │
│  │  │          ▼                                                                    │    │
│  │  │  StrOutputParser() → rewritten_question                                       │    │
│  │  └───────────────────────────────────────────────────────────────────────────────┘    │
│  │                          │                                                            │
│  │                          ▼                                                            │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐    │
│  │  │  STEP 2: Retrieval                                                            │    │
│  │  │  rewritten_question                                                           │    │
│  │  │          │                                                                    │    │
│  │  │          ▼                                                                    │    │
│  │  │     Retriever (FAISS similarity search)                                       │    │
│  │  │          │                                                                    │    │
│  │  │          ▼                                                                    │    │
│  │  │  _format_docs() → join k docs with "\n\n" → context                           │    │
│  │  └───────────────────────────────────────────────────────────────────────────────┘    │
│  │                          │                                                            │
│  │                          ▼                                                            │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐    │
│  │  │  STEP 3: Answer Generation                                                    │    │
│  │  │  {"context": context, "input": original_input, "chat_history": history}       │    │
│  │  │          │                                                                    │    │
│  │  │          ▼                                                                    │    │
│  │  │     qa_prompt                                                                 │    │
│  │  │          │                                                                    │    │
│  │  │          ▼                                                                    │    │
│  │  │        LLM  → "Answer based on context only"                                  │    │
│  │  │          │                                                                    │    │
│  │  │          ▼                                                                    │    │
│  │  │  StrOutputParser() → answer                                                   │    │
│  │  └───────────────────────────────────────────────────────────────────────────────┘    │
│  │                                                                                       │
│  └── self.chain = full_chain                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  rag.invoke(question, chat_history=[])                                                   │
│  ├── payload = {"input": question, "chat_history": []}                                   │
│  ├── self.chain.invoke(payload)                                                          │
│  │   └── LCEL executes: rewrite → retrieve → answer                                      │
│  └── return answer string                                                                │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  Return JSON: {"answer": answer, "session_id": id, "k": k, "engine": "LCEL-RAG"}         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Flow 5: Health Check (`/health`)

1. The entry point is `api/main.py`. When a user sends a GET request to `/health`, FastAPI calls the `health()` endpoint.

2. The function logs the health check and returns a simple dictionary: `{"status": "ok", "service": "document-portal"}`.

**Flow 5 Diagram:**
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    GET /health                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  api/main.py: health()                                                                   │
│  ├── log.info("Health check passed.")                                                    │
│  └── return {"status": "ok", "service": "document-portal"}                               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Flow 6: Serve UI (`/`)

1. The entry point is `api/main.py`. When a user sends a GET request to `/`, FastAPI calls the `serve_ui()` endpoint.

2. The function uses Jinja2Templates to render `templates/index.html`. It passes the request object to the template context.

3. It sets a `Cache-Control: no-store` header to prevent caching and returns the HTML response.

**Flow 6 Diagram:**
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                       GET /                                              │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│  api/main.py: serve_ui(request)                                                          │
│  ├── log.info("Serving UI homepage.")                                                    │
│  │                                                                                       │
│  ├── templates.TemplateResponse("index.html", {"request": request})                      │
│  │   └── Jinja2 renders templates/index.html                                             │
│  │                                                                                       │
│  ├── resp.headers["Cache-Control"] = "no-store"                                          │
│  └── return HTMLResponse                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```
