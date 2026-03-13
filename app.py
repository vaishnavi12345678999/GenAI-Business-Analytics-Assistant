import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

st.set_page_config(page_title="AI Business Analytics Assistant", layout="wide")

st.title("📊 AI Business Analytics Assistant")
st.write("Upload dataset or PDF and analyze using AI.")

# -----------------------------
# Upload files
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
pdf_file = st.file_uploader("Upload PDF for knowledge (RAG)", type=["pdf"])


# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data(file):

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df = df.replace(",", "", regex=True)

    # Convert numeric columns automatically
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


# -----------------------------
# LLM request
# -----------------------------
def generate_response(prompt):

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        data = response.json()

        if "response" in data:
            return data["response"]

        return "Error generating response"

    except Exception as e:
        return f"LLM ERROR: {str(e)}"


# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Embedding model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Load PDF
# -----------------------------
@st.cache_data
def load_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    return text


# -----------------------------
# Split PDF text
# -----------------------------
def split_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks


# -----------------------------
# Build vector DB
# -----------------------------
@st.cache_resource
def build_vector_store(chunks):

    client = chromadb.Client()
    collection = client.get_or_create_collection("pdf_docs")

    for i, chunk in enumerate(chunks):

        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"doc_{i}"]
        )

    return collection


# -----------------------------
# Retrieve PDF context
# -----------------------------
def retrieve_context(question, collection):

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    return context


# -----------------------------
# Build RAG system
# -----------------------------
pdf_collection = None

if pdf_file:

    pdf_text = load_pdf(pdf_file)
    pdf_chunks = split_text(pdf_text)

    pdf_collection = build_vector_store(pdf_chunks)

    st.success("📄 PDF knowledge base ready!")


# -----------------------------
# Dataset section
# -----------------------------
df = None

if uploaded_file:

    df = load_data(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    duckdb.register("data", df)

    # Automatic insights
    st.subheader("📊 Automatic Dataset Insights")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols[:3]:
        st.write(f"Average {col}: {df[col].mean():.2f}")


# -----------------------------
# Chat UI
# -----------------------------
st.divider()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = None

if uploaded_file or pdf_collection:
    user_query = st.chat_input("Ask a question about your data or PDF")


# -----------------------------
# Query processing
# -----------------------------
if user_query:

    st.session_state.messages.append({"role":"user","content":user_query})
    st.chat_message("user").write(user_query)

    # -----------------------------
    # PDF RAG
    # -----------------------------
    if pdf_collection and df is None:

        context = retrieve_context(user_query, pdf_collection)

        rag_prompt = f"""
Answer using the context.

Context:
{context}

Question:
{user_query}
"""

        with st.spinner("Searching PDF knowledge..."):
            rag_answer = generate_response(rag_prompt)

        st.subheader("📄 Answer from PDF")
        st.write(rag_answer)

        st.session_state.messages.append({"role":"assistant","content":rag_answer})

        st.stop()


    # -----------------------------
    # Dataset SQL queries
    # -----------------------------
    if df is not None:

        prompt = f"""
You are an expert data analyst.

Generate valid DuckDB SQL.

Rules:
- Table name is data
- Use LIMIT instead of TOP
- Only use columns provided
- Do not invent columns
- Return SQL only
- No explanation

Columns:
{list(df.columns)}

Question:
{user_query}
"""

        with st.spinner("Generating SQL..."):
            sql_query = generate_response(prompt)

        sql_query = sql_query.replace("```sql","").replace("```","").strip()

        # Fix common mistakes
        sql_query = sql_query.replace("TOP 5","")

        if "limit" not in sql_query.lower():
            sql_query += " LIMIT 5"

        st.subheader("Generated SQL")
        st.code(sql_query)

        try:
            result = duckdb.query(sql_query).to_df()
        except Exception as e:
            st.error(f"SQL execution failed: {e}")
            st.stop()

        st.subheader("Result")
        st.dataframe(result)

        # Download results
        csv = result.to_csv(index=False)

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv"
        )


        # -----------------------------
        # Visualization (COMPACT FIX)
        # -----------------------------
        if len(result.columns) >= 2:

            x_col = result.columns[0]
            y_col = result.columns[1]

            fig, ax = plt.subplots(figsize=(7,4))   # Smaller chart

            if "year" in x_col.lower():
                result.plot(x=x_col, y=y_col, kind="line", ax=ax)
            else:
                result.plot(x=x_col, y=y_col, kind="bar", ax=ax)

            plt.xticks(rotation=30)
            plt.tight_layout()

            st.subheader("Visualization")
            st.pyplot(fig)


        # -----------------------------
        # Insight
        # -----------------------------
        st.subheader("Quick Insight")

        if len(result.columns) >= 2:

            top_row = result.iloc[result[result.columns[1]].idxmax()]

            st.write(
                f"📌 Highest value is **{top_row[1]}** for **{top_row[0]}**."
            )

        st.session_state.messages.append(
            {"role":"assistant","content":"Analysis completed."}
        )
