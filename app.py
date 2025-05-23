import streamlit as st
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import streamlit as st
api_key = st.secrets["openai"]["api_key"]

# Load tickets
@st.cache_data
def load_tickets(path="tickets.csv"):
    df = pd.read_csv(path)
    return df
from langchain_community.embeddings import OpenAIEmbeddings

embedder = OpenAIEmbeddings(openai_api_key=api_key)

# Build engine
class RAGTicketEngine:
    def __init__(self, path="tickets.csv"):
        self.data = load_tickets(path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.generator = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self._build_vector_store()

    def _build_vector_store(self):
        texts = self.data["body"].tolist()
        self.embeddings = self.embedder.encode(texts, show_progress_bar=True)
        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings).astype("float32"))

    def query(self, query_text):
        q_emb = self.embedder.encode([query_text])[0]
        D, I = self.index.search(np.array([q_emb]).astype("float32"), k=3)
        context = "\n".join(
            f"Ticket: {self.data.iloc[i].body}\nResolution: {self.data.iloc[i].resolution}"
            for i in I[0]
        )
        prompt = f"Customer asks: {query_text}\n\n{context}\n\nSummarize this:"
        result = self.generator(prompt, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
        return result

# Web UI
st.title("🧠 Customer Support Ticket Summarizer (RAG)")

uploaded_file = st.file_uploader("Upload tickets.csv file", type=["csv"])
if uploaded_file:
    with open("tickets.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Uploaded successfully. Engine is initializing...")
    engine = RAGTicketEngine("tickets.csv")

    query = st.text_input("Enter customer query:")
    if query and st.button("Generate Summary"):
        with st.spinner("Finding similar tickets and generating summary..."):
            summary = engine.query(query)
            st.subheader("🔍 Summary:")
            st.write(summary)
