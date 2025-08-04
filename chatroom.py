import streamlit as st
from utils.retrieval import retrieve
from utils.bm25_retriever import build_bm25
from utils.faiss_retriever import build_faiss_index
from utils.preprocessing import load_agri_data

st.set_page_config(page_title="🌾 Agri Chatbot", layout="centered")
st.title("👨‍🌾 Myanmar Agricultural Chatbot")

# Load everything
corpus, answers, _ = load_agri_data("data/agri.json")
bm25 = build_bm25(corpus)
faiss_index = build_faiss_index(corpus)

query = st.text_input("📥 Ask your farming question:")
if query:
    reply = retrieve(query, bm25, faiss_index, corpus, answers)
    st.chat_message("user").write(query)
    st.chat_message("bot").write(reply)
