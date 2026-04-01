import sys
import platform
if platform.system() == "Linux":
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

st.set_page_config(page_title="AI Business Analytics Assistant", layout="wide")

st.title("📊 AI Business Analytics Assistant")
st.write("Upload dataset or PDF and analyze using AI.")

# -----------------------------
# LLM request (Groq)
# -----------------------------
def generate_response(prompt):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"LLM ERROR: {str(e)}"


# -----------------------------
# Upload files
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
pdf_file = st.file_uploader("Upload PDF for knowledge (RAG)", type=["pdf"])


# -----------------------------
# Load dataset (FIXED)
# -----------------------------
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Remove commas (for numbers like 1,000)
    df = df.replace(",", "", regex=True)

    # Convert only object columns safely
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop completely empty rows
    df = df.dropna(how="all")

    return df


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

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # -----------------------------
    # PDF RAG
    # -----------------------------
    if pdf_collection and df is None:

        context = retrieve_context(user_query, pdf_collection)

        rag_prompt = f"""
Answer using the context below. Be concise and helpful.

Context:
{context}

Question:
{user_query}
"""

        with st.spinner("Searching PDF knowledge..."):
            rag_answer = generate_response(rag_prompt)

        st.subheader("📄 Answer from PDF")
        st.write(rag_answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": rag_answer}
        )

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

        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        sql_query = sql_query.replace("TOP 5", "")

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

        csv = result.to_csv(index=False)

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv"
        )

        # -----------------------------
        # Visualization
        # -----------------------------
        if len(result.columns) >= 2:

            numeric_cols = result.select_dtypes(
                include=["int64", "float64"]
            ).columns

            if len(numeric_cols) == 0:
                st.warning("No numeric data available for visualization.")
            else:
                x_col = result.columns[0]
                y_col = numeric_cols[0]

                result[y_col] = pd.to_numeric(result[y_col], errors="coerce")
                result = result.dropna(subset=[y_col])

                num_rows = len(result)
                fig_width = max(8, num_rows * 1.2)

                fig, ax = plt.subplots(figsize=(fig_width, 5))

                if "year" in x_col.lower() or "date" in x_col.lower():
                    result.plot(
                        x=x_col, y=y_col, kind="line", ax=ax,
                        marker="o", linewidth=2.5
                    )
                else:
                    result.plot(
                        x=x_col, y=y_col, kind="bar", ax=ax,
                        edgecolor="white", width=0.6
                    )
                    for container in ax.containers:
                        ax.bar_label(container, fmt="%.1f", padding=4, fontsize=9)

                ax.set_title(
                    f"{y_col} by {x_col}",
                    fontsize=14, fontweight="bold", pad=15
                )
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.yaxis.grid(True, linestyle="--", alpha=0.7)
                plt.xticks(rotation=35, ha="right")
                plt.tight_layout()

                st.subheader("📊 Visualization")
                st.pyplot(fig, use_container_width=True)

        # -----------------------------
        # Insight
        # -----------------------------
        st.subheader("Quick Insight")

        if len(result.columns) >= 2 and len(numeric_cols) > 0:
            try:
                top_row = result.iloc[result[y_col].idxmax()]
                st.write(
                    f"📌 Highest value is **{top_row[y_col]}** for **{top_row[result.columns[0]]}**."
                )
            except:
                st.info("Insight could not be generated.")

        st.session_state.messages.append(
            {"role": "assistant", "content": "Analysis completed."}
        )
