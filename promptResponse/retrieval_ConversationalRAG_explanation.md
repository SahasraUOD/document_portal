# ConversationalRAG Class Explanation

**File:** `src/document_chat/retrieval.py`  
**Class:** `ConversationalRAG`

---

## 1. Overview and Purpose (Business Logic)

### What Problem Does This Solve?

When users upload documents (PDFs, text files), they want to have a **conversation** with those documents—asking follow-up questions that reference previous answers. This is fundamentally different from a simple search:

| Simple Search | Conversational RAG |
|---------------|-------------------|
| "What are the payment terms?" | "What are the payment terms?" |
| "What happens if late?" ❌ Fails | "What happens if they're late?" ✅ Understands "they" = payments |

### Business Purpose

The `ConversationalRAG` class is the **core intelligence** behind the document chat feature. It:

1. **Remembers conversation context** — Users can ask "tell me more about that" and the system understands what "that" refers to
2. **Finds relevant information** — Searches through potentially thousands of document pages to find the exact passages needed
3. **Generates natural answers** — Produces human-readable responses, not just document excerpts

### How It Works (Conceptually)

```
User asks: "What happens if they're late?"
                    ↓
┌─────────────────────────────────────────────────────┐
│ Step 1: UNDERSTAND THE QUESTION                      │
│ Look at chat history: previous Q was about payments  │
│ Rewrite question: "What happens if payments are late"│
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Step 2: FIND RELEVANT INFORMATION                    │
│ Search document index for "late payments"            │
│ Return top 5 most relevant passages                  │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ Step 3: GENERATE ANSWER                              │
│ Send passages + question to LLM                      │
│ Generate: "Late payments incur a 2% monthly fee..."  │
└─────────────────────────────────────────────────────┘
```

---

## 2. Class Attributes and Their Roles

### Core Attributes

| Attribute | Type | Purpose | How It's Set Up |
|-----------|------|---------|-----------------|
| `session_id` | `Optional[str]` | **Identifies the conversation** — Used for logging and tracking which user/session this belongs to | Passed directly to `__init__` |
| `llm` | LLM instance | **The "brain"** — Language model that rewrites questions and generates answers | Loaded via `ModelLoader().load_llm()` which reads from `config/config.yaml` and uses either Google or Groq API |
| `contextualize_prompt` | `ChatPromptTemplate` | **Question rewriter instructions** — Tells the LLM how to rewrite questions with context | Loaded from `PROMPT_REGISTRY` using key `PromptType.CONTEXTUALIZE_QUESTION.value` ("contextualize_question") |
| `qa_prompt` | `ChatPromptTemplate` | **Answer generator instructions** — Tells the LLM how to answer based on retrieved context | Loaded from `PROMPT_REGISTRY` using key `PromptType.CONTEXT_QA.value` ("context_qa") |
| `retriever` | Retriever instance | **Document searcher** — Finds similar passages from the FAISS vector index | Either passed to `__init__` or loaded later via `load_retriever_from_faiss()` |
| `chain` | LCEL Chain | **The complete pipeline** — Connects question rewriting → retrieval → answer generation | Built automatically when retriever is available via `_build_lcel_chain()` |

### Why Lazy Initialization?

The `retriever` and `chain` are **not** created immediately. This is intentional:

```
Scenario 1: User just opened the app
├── ConversationalRAG is created (session_id only)
├── Retriever = None (no documents loaded yet)
└── Chain = None (can't build without retriever)

Scenario 2: User uploads a document
├── Document is processed and indexed
├── load_retriever_from_faiss() is called
├── Retriever is created from the index
└── Chain is built automatically

Scenario 3: Retriever provided upfront
├── ConversationalRAG(retriever=existing_retriever)
└── Chain is built immediately
```

---

## 3. Methods with Parameters/Returns and Functionalities

### 3.1 `__init__(self, session_id: Optional[str], retriever=None)`

**Purpose:** Initialize the RAG system with essential components.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `Optional[str]` | - | Unique identifier for this conversation (e.g., "user_123_session_456") |
| `retriever` | Any | `None` | Pre-configured retriever; if provided, chain is built immediately |

**Returns:** None (constructor)

**What Happens Inside:**

```
__init__ called
    │
    ├─→ Store session_id
    │
    ├─→ Load LLM via _load_llm()
    │       └─→ ModelLoader().load_llm()
    │               ├─→ Reads config/config.yaml for model settings
    │               ├─→ Checks LLM_PROVIDER env var (default: "google")
    │               └─→ Returns ChatGoogleGenerativeAI or ChatGroq instance
    │
    ├─→ Load contextualize_prompt from PROMPT_REGISTRY
    │       └─→ Key: "contextualize_question"
    │           Template: "Given conversation history, rewrite the query as standalone..."
    │
    ├─→ Load qa_prompt from PROMPT_REGISTRY
    │       └─→ Key: "context_qa"  
    │           Template: "Answer using the provided context. If not found, say I don't know..."
    │
    ├─→ Set retriever (None or provided value)
    │
    ├─→ Set chain = None
    │
    └─→ IF retriever is not None:
            └─→ Call _build_lcel_chain() to build the pipeline
```

**Error Handling:** Any exception is caught, logged, and re-raised as `DocumentPortalException`.

---

### 3.2 `load_retriever_from_faiss(self, index_path, k=5, index_name="index", search_type="similarity", search_kwargs=None)`

**Purpose:** Load a pre-built FAISS index from disk and prepare the system for answering questions.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index_path` | `str` | - | Directory containing FAISS index files (e.g., "data/single_document_chat/user_123") |
| `k` | `int` | `5` | Number of document chunks to retrieve per query |
| `index_name` | `str` | `"index"` | Base name of the index files (looks for `index.faiss` and `index.pkl`) |
| `search_type` | `str` | `"similarity"` | Search algorithm: "similarity" (cosine) or "mmr" (diversity-focused) |
| `search_kwargs` | `Optional[Dict]` | `None` | Additional search parameters; defaults to `{"k": k}` |

**Returns:** The configured retriever instance.

**What Happens Inside:**

```
load_retriever_from_faiss called
    │
    ├─→ Validate index_path exists
    │       └─→ Raises FileNotFoundError if directory missing
    │
    ├─→ Load embeddings model
    │       └─→ ModelLoader().load_embeddings()
    │               └─→ Returns GoogleGenerativeAIEmbeddings
    │                   (same model used during document indexing)
    │
    ├─→ Load FAISS vectorstore from disk
    │       └─→ FAISS.load_local(index_path, embeddings, index_name, 
    │                            allow_dangerous_deserialization=True)
    │               ├─→ Reads index.faiss (vector data)
    │               └─→ Reads index.pkl (metadata like document text)
    │
    ├─→ Convert vectorstore to retriever
    │       └─→ vectorstore.as_retriever(search_type, search_kwargs)
    │               └─→ Retriever configured to return top-k similar docs
    │
    ├─→ Build the LCEL chain
    │       └─→ _build_lcel_chain()
    │
    └─→ Return the retriever
```

**Why `allow_dangerous_deserialization=True`?**  
FAISS indexes contain pickled Python objects. This flag is safe when you trust the index source (your own system created it), but could be dangerous with untrusted indexes.

---

### 3.3 `invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str`

**Purpose:** Answer a user's question using the RAG pipeline.

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_input` | `str` | - | The user's question (e.g., "What are the payment terms?") |
| `chat_history` | `Optional[List[BaseMessage]]` | `None` | Previous messages in the conversation |

**Returns:** `str` — The generated answer.

**What Happens Inside:**

```
invoke("What happens if late?", chat_history=[...])
    │
    ├─→ Check if chain exists
    │       └─→ Raises exception if None (must call load_retriever first)
    │
    ├─→ Normalize chat_history (None → [])
    │
    ├─→ Create payload: {"input": user_input, "chat_history": chat_history}
    │
    ├─→ Execute the LCEL chain
    │       │
    │       ├─→ Stage 1: Question Rewriter
    │       │       Input: "What happens if late?" + history about payments
    │       │       Output: "What happens if payments are late?"
    │       │
    │       ├─→ Stage 2: Document Retrieval  
    │       │       Input: "What happens if payments are late?"
    │       │       Output: Top 5 relevant document chunks
    │       │
    │       └─→ Stage 3: Answer Generation
    │               Input: Retrieved chunks + original question + history
    │               Output: "According to section 5.2, late payments incur..."
    │
    ├─→ Handle empty response
    │       └─→ Returns "no answer generated." if LLM returns empty
    │
    └─→ Return the answer string
```

---

### 3.4 `_load_llm(self)` (Private)

**Purpose:** Internal helper to load the Language Model.

**Returns:** LLM instance (ChatGoogleGenerativeAI or ChatGroq)

**What Happens:**
1. Calls `ModelLoader().load_llm()`
2. `ModelLoader` reads `config/config.yaml` for model settings
3. Checks `LLM_PROVIDER` environment variable (defaults to "google")
4. Returns the appropriate LLM client with API key from environment

---

### 3.5 `_format_docs(docs)` (Static Method)

**Purpose:** Convert retrieved document objects into a single text string.

**Inputs:** `docs` — List of document objects with `page_content` attribute

**Returns:** `str` — All document contents joined with double newlines

**Logic:**
```python
# Takes: [Doc("Payment terms..."), Doc("Late fees..."), Doc("Penalties...")]
# Returns: "Payment terms...\n\nLate fees...\n\nPenalties..."
```

---

### 3.6 `_build_lcel_chain(self)` (Private)

**Purpose:** Construct the complete RAG pipeline using LangChain Expression Language (LCEL).

**What It Builds:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LCEL CHAIN STRUCTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT: {"input": "user question", "chat_history": [...]}          │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: QUESTION REWRITER                                   │   │
│  │                                                              │   │
│  │  {"input": itemgetter("input"),                             │   │
│  │   "chat_history": itemgetter("chat_history")}               │   │
│  │              │                                               │   │
│  │              ▼                                               │   │
│  │  contextualize_prompt (template with placeholders)          │   │
│  │              │                                               │   │
│  │              ▼                                               │   │
│  │  llm (generates rewritten question)                         │   │
│  │              │                                               │   │
│  │              ▼                                               │   │
│  │  StrOutputParser() (extracts text from LLM response)        │   │
│  │                                                              │   │
│  │  OUTPUT: "Standalone question without context dependencies"  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: DOCUMENT RETRIEVAL                                  │   │
│  │                                                              │   │
│  │  question_rewriter output                                   │   │
│  │              │                                               │   │
│  │              ▼                                               │   │
│  │  self.retriever (FAISS similarity search)                   │   │
│  │              │                                               │   │
│  │              ▼                                               │   │
│  │  _format_docs (join documents into single string)           │   │
│  │                                                              │   │
│  │  OUTPUT: "Doc1 content\n\nDoc2 content\n\nDoc3 content..."   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: ANSWER GENERATION                                   │   │
│  │                                                              │   │
│  │  {"context": retrieve_docs output,                          │   │
│  │   "input": itemgetter("input"),      ← original question    │   │
│  │   "chat_history": itemgetter("chat_history")}               │   │
│  │              │                                               │   │
│  │              ▼                                               │   │
│  │  qa_prompt (template: "Answer using context...")            │   │
│  │              │                                               │   │
│  │              ▼                                               │   │
│  │  llm (generates the final answer)                           │   │
│  │              │                                               │   │
│  │              ▼                                               │   │
│  │  StrOutputParser() (extracts text)                          │   │
│  │                                                              │   │
│  │  OUTPUT: "The answer to your question is..."                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Interactions with Other Classes/Modules

### 4.1 `ModelLoader` (from `utils/model_loader.py`)

**What It Does:** Manages API keys and loads AI models.

**How ConversationalRAG Uses It:**
- `ModelLoader().load_llm()` — Returns the language model (Google Gemini or Groq)
- `ModelLoader().load_embeddings()` — Returns the embedding model for FAISS

**Internal Details:**
- Reads configuration from `config/config.yaml`
- Handles API keys from environment variables or ECS secrets
- Supports multiple LLM providers (Google, Groq) based on `LLM_PROVIDER` env var

---

### 4.2 `PROMPT_REGISTRY` (from `prompt/prompt_library.py`)

**What It Does:** Central storage for all prompt templates.

**How ConversationalRAG Uses It:**
- `PROMPT_REGISTRY["contextualize_question"]` — Prompt for rewriting questions
- `PROMPT_REGISTRY["context_qa"]` — Prompt for generating answers

**The Prompts:**

**contextualize_question:**
```
Given a conversation history and the most recent user query, rewrite the query 
as a standalone question that makes sense without relying on the previous context.
Do not provide an answer—only reformulate the question if necessary; otherwise, 
return it unchanged.
```

**context_qa:**
```
You are an assistant designed to answer questions using the provided context. 
Rely only on the retrieved information to form your response. If the answer is 
not found in the context, respond with 'I don't know.' Keep your answer concise 
and no longer than three sentences.
```

---

### 4.3 `FAISS` (from `langchain_community.vectorstores`)

**What It Does:** Fast similarity search over document embeddings.

**How ConversationalRAG Uses It:**
- `FAISS.load_local()` — Loads pre-built index from disk
- `vectorstore.as_retriever()` — Converts to retriever interface

**Files It Reads:**
- `{index_path}/{index_name}.faiss` — Binary vector data
- `{index_path}/{index_name}.pkl` — Document metadata (text, sources)

---

### 4.4 `DocumentPortalException` (from `exception/custom_exception.py`)

**What It Does:** Custom exception with context information.

**How ConversationalRAG Uses It:**
- Wraps all errors with descriptive messages
- Includes `sys` module for stack trace information

---

### 4.5 `GLOBAL_LOGGER` (from `logger/__init__.py`)

**What It Does:** Structured logging with session context.

**How ConversationalRAG Uses It:**
- `log.info()` — Successful operations
- `log.warning()` — Empty responses
- `log.error()` — Failures before raising exceptions

---

## 5. Usage Examples

### Example 1: Basic Usage (Lazy Initialization)

```python
from src.document_chat.retrieval import ConversationalRAG

# Step 1: Create RAG instance
rag = ConversationalRAG(session_id="user_123_session_001")
# At this point: LLM and prompts loaded, but NO retriever or chain

# Step 2: Load the document index
rag.load_retriever_from_faiss(
    index_path="data/single_document_chat/user_123",
    k=5,  # Return top 5 relevant chunks
    index_name="index"
)
# Now: retriever and chain are ready

# Step 3: Ask first question
answer1 = rag.invoke(
    user_input="What are the payment terms?",
    chat_history=[]
)
print(answer1)
# Output: "Payment is due within 30 days of invoice receipt."
```

### Example 2: Multi-Turn Conversation

```python
from src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage

# Initialize and load index
rag = ConversationalRAG(session_id="user_123")
rag.load_retriever_from_faiss("data/single_document_chat/user_123")

# First question
q1 = "What are the payment terms?"
a1 = rag.invoke(q1, chat_history=[])
print(f"Q: {q1}\nA: {a1}\n")

# Build chat history
history = [
    HumanMessage(content=q1),
    AIMessage(content=a1)
]

# Follow-up question (refers to "payments" implicitly)
q2 = "What happens if they're late?"
a2 = rag.invoke(q2, chat_history=history)
print(f"Q: {q2}\nA: {a2}\n")
# The system understands "they" refers to "payments" from context

# Continue the conversation
history.extend([
    HumanMessage(content=q2),
    AIMessage(content=a2)
])

q3 = "Are there any exceptions?"
a3 = rag.invoke(q3, chat_history=history)
print(f"Q: {q3}\nA: {a3}")
```

### Example 3: With Pre-loaded Retriever

```python
from src.document_chat.retrieval import ConversationalRAG
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader

# Pre-load retriever externally (useful for sharing across instances)
embeddings = ModelLoader().load_embeddings()
vectorstore = FAISS.load_local(
    "data/shared_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Pass retriever directly — chain builds immediately
rag = ConversationalRAG(session_id="shared_session", retriever=retriever)

# Ready to use immediately
answer = rag.invoke("Summarize the key points")
```

### Example 4: Customizing Search Parameters

```python
from src.document_chat.retrieval import ConversationalRAG

rag = ConversationalRAG(session_id="advanced_user")

# Use MMR (Maximum Marginal Relevance) for diverse results
rag.load_retriever_from_faiss(
    index_path="data/large_document_index",
    k=10,
    search_type="mmr",  # Balance relevance and diversity
    search_kwargs={
        "k": 10,
        "fetch_k": 50,  # Fetch more candidates before filtering
        "lambda_mult": 0.7  # 0=max diversity, 1=max relevance
    }
)

answer = rag.invoke("Compare all the pricing tiers")
```

---

## 6. Summary: The Logical Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL RAG LIFECYCLE                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INITIALIZATION PHASE                                            │
│  ───────────────────                                             │
│  ConversationalRAG(session_id)                                   │
│       │                                                          │
│       ├─→ Load LLM (the reasoning engine)                        │
│       ├─→ Load prompts (instructions for the LLM)                │
│       └─→ Retriever/Chain = None (waiting for documents)         │
│                                                                  │
│  DOCUMENT LOADING PHASE                                          │
│  ──────────────────────                                          │
│  load_retriever_from_faiss(index_path)                           │
│       │                                                          │
│       ├─→ Load FAISS index (document embeddings)                 │
│       ├─→ Create retriever (search interface)                    │
│       └─→ Build LCEL chain (the complete pipeline)               │
│                                                                  │
│  CONVERSATION PHASE (repeats)                                    │
│  ────────────────────────────                                    │
│  invoke(question, chat_history)                                  │
│       │                                                          │
│       ├─→ Rewrite question with context                          │
│       ├─→ Search for relevant passages                           │
│       ├─→ Generate answer from passages                          │
│       └─→ Return answer (add to history for next turn)           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Key Insight:** The class separates "setup" from "execution" cleanly. You configure the system once (load models, load index), then call `invoke()` repeatedly for each conversation turn. The LCEL chain handles the complex three-stage processing transparently.
