from langchain_community.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings


def create_vector_store(text):

    text_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = text_splitter.split_text(text)

    embeddings = FastEmbedEmbeddings()

    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore


def ask_question(vectorstore, question):

    llm = Ollama(model="phi3")

    docs = vectorstore.similarity_search(question, k=3)

    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a business analytics assistant.

    Use the context to answer the question clearly.

    Context:
    {context}

    Question:
    {question}
    """

    return llm.invoke(prompt)