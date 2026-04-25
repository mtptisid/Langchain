# LangChain Tutorials — From Zero to Production

> A hands-on, beginner-friendly tutorial series for building LLM-powered applications
> using LangChain and Google Gemini. Every concept explained with working code,
> real errors encountered, and honest fixes.

---

## Who This Is For

- You know Python basics but have never built an AI application
- You've heard of LangChain but don't know where to start
- You want to understand *why* things work, not just copy-paste code
- You're building a RAG pipeline or AI agent and hitting real errors

No ML degree needed. No GPU needed. Just a Google AI Studio API key (free).

---

## Tutorial Series

### Part 1 — LangChain Fundamentals (This Repo)

The complete beginner foundation. All concepts explained ground-up with Gemini as the LLM.

| Section | What You Learn |
|---|---|
| [Models](#part-1--your-first-llm-call) | Connect to Gemini in 3 lines |
| [Prompt Templates](#part-2--prompt-templates) | Reusable structured prompts with variables |
| [Output Parsers](#part-3--output-parsers) | Get strings, JSON, and typed objects from LLM |
| [Chains (LCEL)](#part-4--chains-lcel) | Connect steps with the `\|` pipe operator |
| [Memory](#part-5--memory) | Build multi-turn conversations |
| [Document Loaders](#part-6--document-loading) | Load PDFs, web pages, Wikipedia |
| [Text Splitting](#part-7--text-splitting) | Chunk documents for retrieval |
| [Vector Stores](#part-8--vector-stores-and-retrieval) | Embed, store, and search by meaning |
| [RAG Pipeline](#part-9--full-rag-chain) | Full retrieval-augmented generation |
| [Streaming](#part-10--streaming-responses) | Token-by-token output |
| [Agents](#part-11--agents-without-version-issues) | LLM that uses tools to reason and act |

### Part 2 — RAG Pipeline from Scratch *(coming soon)*

Build the same RAG system without any frameworks — raw `pypdf`, `faiss`, `numpy`, and `sentence-transformers`. Understand exactly what LangChain is doing under the hood.

### Part 3 — LangGraph Agents *(coming soon)*

Stateful multi-step agents with loops, conditionals, and human-in-the-loop using LangGraph.

---

## Notebooks

| Notebook | Description | Open in Colab |
|---|---|---|
| `01_langchain_fundamentals.ipynb` | Parts 1–5: Models, Prompts, Chains, Memory | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `02_rag_pipeline.ipynb` | Parts 6–9: Load, Chunk, Embed, Retrieve, Generate | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `03_agents.ipynb` | Parts 10–11: Streaming + Agents with Tools | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |

---

## Setup

### 1. Get a free Gemini API key

Go to [aistudio.google.com](https://aistudio.google.com) → Create API key → Copy it.
The free tier is generous enough for all tutorials here.

### 2. Install dependencies

```bash
pip install langchain langchain-community langchain-google-genai \
            langchain-text-splitters faiss-cpu pypdf \
            wikipedia duckduckgo-search google-generativeai
```

### 3. Set your API key

```python
import os
os.environ["GOOGLE_API_KEY"] = "your-key-here"

# In Google Colab, use Secrets instead (safer):
from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
```

### 4. Verify setup

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
response = llm.invoke([HumanMessage(content="Say hello in one sentence.")])
print(response.content)
# → "Hello! How can I assist you today?"
```

---

## Core Concepts at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│  1. Models     → the LLM brain (Gemini, Claude, GPT...)     │
│  2. Prompts    → structured instructions with variables      │
│  3. Chains     → steps connected with | (pipe operator)      │
│  4. Retrievers → search your documents by meaning            │
│  5. Memory     → remember conversation history               │
│  6. Agents     → LLM that decides what tools to use          │
└─────────────────────────────────────────────────────────────┘
```

The modern LangChain way (LCEL — LangChain Expression Language):

```python
chain = prompt | llm | output_parser
result = chain.invoke({"question": "What is RAG?"})
```

---

## Part 1 — Your First LLM Call

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
response = llm.invoke([HumanMessage(content="What is the capital of France?")])
print(response.content)   # → "The capital of France is Paris."
```

**Swap Gemini for any other LLM in one line — the rest of your code stays identical:**

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

from langchain_community.llms import Ollama
llm = Ollama(model="llama3")   # fully local, no API key
```

---

## Part 2 — Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain things simply."),
    ("human", "Explain {topic} in 3 bullet points.")
])

chain = prompt | llm
result = chain.invoke({"topic": "machine learning"})
```

---

## Part 3 — Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser

# Returns plain string instead of AIMessage object
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "neural networks"})
print(type(result))   # <class 'str'>
```

---

## Part 4 — Chains (LCEL)

```python
from langchain_core.runnables import RunnableParallel

# Two LLM calls run simultaneously
parallel = RunnableParallel(
    pros=pros_prompt | llm | StrOutputParser(),
    cons=cons_prompt | llm | StrOutputParser(),
)
result = parallel.invoke({"technology": "React"})
print(result["pros"])
print(result["cons"])
```

---

## Part 5 — Memory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "user_001"}}
chain_with_memory.invoke({"input": "My name is Arjun."}, config=config)
chain_with_memory.invoke({"input": "What is my name?"}, config=config)
# → "Your name is Arjun."
```

---

## Part 6 — Document Loading

```python
from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader

# Load a PDF — returns list of Document objects
pages = PyPDFLoader("docs/nist_ai_rmf.pdf").load()

# Load from Wikipedia directly
docs = WikipediaLoader(query="Large language model", load_max_docs=2).load()

print(pages[0].page_content[:300])   # text
print(pages[0].metadata)             # {"source": "...", "page": 0}
```

---

## Part 7 — Text Splitting

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(pages)
print(f"{len(pages)} pages → {len(chunks)} chunks")
```

---

## Part 8 — Vector Stores and Retrieval

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Free local embeddings — no API key needed
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build index, save to disk
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("my_index")

# Search by meaning
results = vectorstore.similarity_search("AI provider obligations", k=3)
```

---

## Part 9 — Full RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What does the GOVERN function cover in the AI RMF?")
print(answer)
```

---

## Part 10 — Streaming

```python
for chunk in rag_chain.stream("What are the risk categories in NIST AI RMF?"):
    print(chunk, end="", flush=True)
```

---

## Part 11 — Agents (Without Version Issues)

> **Real lesson learned:** LangChain's agent APIs change between versions.
> This version-proof pattern uses only `langchain_core.messages` — no `langchain.agents` import needed.

```python
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_tool   = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
search_tool = DuckDuckGoSearchRun()
tool_map    = {"wikipedia": wiki_tool, "duckduckgo_search": search_tool}

llm_with_tools = llm.bind_tools(list(tool_map.values()))

def run_agent(question: str, max_iterations: int = 5) -> str:
    messages = [HumanMessage(content=question)]
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return response.content
        for tc in response.tool_calls:
            result = tool_map[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return "Max iterations reached."

print(run_agent("Who signed the EU AI Act and when?"))
```

---

## Common Errors Encountered (and Fixed)

These are real errors hit while building these tutorials — not made up for the docs.

| Error | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'create_tool_calling_agent'` | Wrong or fake `langchain` package installed | Uninstall, reinstall `langchain==0.3.7` |
| `ImportError: cannot import name 'initialize_agent'` | Same fake package issue | See above |
| `LangChain version: 1.2.15` with 5 exports | PyPI name-squatted fake package | `pip uninstall langchain -y` then reinstall real one |
| `ModuleNotFoundError: langchain.text_splitter` | Moved to separate package | `pip install langchain-text-splitters` |
| `ModuleNotFoundError: langchain.embeddings` | Moved to community package | `pip install langchain-community` |
| `AttributeError: 'Anthropic' has no attribute 'chat'` | Variable name collision (`client` reused) | Use `openai_client`, `claude_client` separately |
| `allow_dangerous_deserialization` error | Loading FAISS index from disk | Add `allow_dangerous_deserialization=True` |

---

## Module Map

```
langchain                → core chains, prompts, base classes
langchain-google-genai   → Google Gemini chat + embedding models
langchain-anthropic      → Anthropic Claude models
langchain-community      → 100+ loaders, tools, vectorstores
langchain-text-splitters → all text splitter classes
langchain-core           → base abstractions (Runnable, BaseMessage)
langgraph                → stateful multi-agent orchestration (advanced)
```

---

## What's Coming Next

- **Part 2** — RAG from Scratch (no frameworks, raw Python)
- **Part 3** — LangGraph: stateful agents with memory and loops
- **Part 4** — Deploy your chain as a REST API with LangServe
- **Part 5** — Evaluate and debug RAG with LangSmith

---



## License

MIT — learn freely, share openly.
