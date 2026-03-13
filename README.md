# 📊 GenAI Business Analytics Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Database](https://img.shields.io/badge/Database-DuckDB-orange)
![LLM](https://img.shields.io/badge/LLM-Ollama%20Phi3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An **AI-powered Business Analytics Assistant** built with **Streamlit, DuckDB, and Ollama** that enables users to analyze datasets and documents using **natural language queries**.

The system combines:

- **Natural Language → SQL generation** for dataset analytics
- **Retrieval-Augmented Generation (RAG)** for answering questions from PDF documents

Users can upload **CSV / Excel datasets** or **PDF reports** and interact with them through a **chat-based AI interface**.

---

## 🚀 Features

### 📊 Dataset Analytics
- Upload **CSV or Excel datasets**
- Ask questions using **natural language**
- AI automatically generates **DuckDB SQL queries**
- Instant **query execution**
- Display results in **tables**

### 📄 Document Question Answering (RAG)
- Upload **PDF documents**
- Automatic **text extraction**
- Convert text into **vector embeddings**
- Store embeddings in **ChromaDB vector database**
- Retrieve relevant document context for answers

### 📈 Visual Insights
- Automatic **bar charts**
- Automatic **line charts for time-based data**
- Quick insights generated from query results

### 💬 Interactive Chat Interface
- Conversational analytics workflow
- Persistent **chat history during session**
- Ask multiple questions about the same dataset

### 📥 Export Results
- Download query results as **CSV files**

---

## 🎬 Application Demo

### 📊 Excel / Dataset Analytics Demo

(https://github.com/vaishnavi12345678999/GenAI-Business-Analytics-Assistant/issues/1#issue-4071522148)

---

### 📄 PDF Question Answering Demo

https://github.com/user-attachments/assets/PASTE-YOUR-PDF-VIDEO-LINK-HERE

---

## 🧠 System Architecture

### Dataset Analytics Pipeline
```
User Question
     │
     ▼
LLM generates SQL query
     │
     ▼
DuckDB executes query
     │
     ▼
Query results returned
     │
     ▼
Visualization + Insight
```

### PDF Analysis Pipeline (RAG)
```
PDF Upload
     │
     ▼
Text Extraction
     │
     ▼
Text Chunking
     │
     ▼
SentenceTransformer Embeddings
     │
     ▼
ChromaDB Vector Store
     │
     ▼
Semantic Retrieval
     │
     ▼
LLM Generates Answer
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Data Processing | Pandas |
| SQL Engine | DuckDB |
| Visualization | Matplotlib |
| LLM Backend | Ollama (`phi3:mini`) |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| Document Processing | PyPDF |

---

## ⚙️ Prerequisites

- Python **3.9+**
- [Ollama](https://ollama.com) installed locally
- `phi3:mini` model downloaded in Ollama

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/vaishnavi12345678999/GenAI-Business-Analytics-Assistant.git
cd GenAI-Business-Analytics-Assistant
```

### 2️⃣ Install dependencies
```bash
pip install streamlit pandas duckdb matplotlib requests chromadb sentence-transformers pypdf openpyxl
```

### 3️⃣ Install Ollama model
```bash
ollama pull phi3:mini
```

---

## ▶️ Running the Application

Start Ollama:
```bash
ollama run phi3:mini
```

Run the Streamlit app:
```bash
streamlit run app.py
```

The application will open at: `http://localhost:8501`

---

## 💡 Example Queries

### Dataset Questions
- Top 5 products by profit
- Total sales by region
- Revenue trend by year
- Best performing category

### PDF Questions
- What are the key insights from this report?
- Which region has the highest growth?
- Summarize the document.

---

## 📂 Project Structure
```
GenAI-Business-Analytics-Assistant/
│
├── app.py
├── README.md
└── screenshots/
    ├── upload.png
    ├── dataset_preview.png
    ├── query_result.png
    └── visualization.png
```

---

## ⚠️ Known Limitations

- LLM-generated SQL may occasionally produce invalid queries.
- The vector database runs in-memory and resets when the app restarts.
- Works best with text-based PDFs.
- Scanned PDFs are not supported.

---

## 🚀 Future Improvements

- Multi-dataset analytics support
- Advanced visualization options
- SQL validation layer
- Cloud deployment
- Support for additional LLM models

---

## 👩‍💻 Author

**Vaishnavi Vaitla**


---

## 📄 License

MIT
