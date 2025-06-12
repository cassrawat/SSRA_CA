import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os

st.set_page_config(page_title="Chat with Budget 2025 PDF", layout="wide")
st.title("📊 Chat with Budget 2025 FAQs")

openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    with st.spinner("Processing PDF..."):
        reader = PdfReader("faqs-budget-2025.pdf")
        raw_text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=150)
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(texts, embeddings)

    query = st.text_input("❓ Ask a question about Budget 2025:")
    if query:
        docs = db.similarity_search(query)
        llm = ChatOpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.success(response)
