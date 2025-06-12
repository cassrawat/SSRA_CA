import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import GPT4All
from PyPDF2 import PdfReader

st.set_page_config(page_title="Offline PDF Chatbot")
st.title("📄 Chat with Budget 2025 (Offline)")

# Load PDF
reader = PdfReader("faqs-budget-2025.pdf")
raw_text = ""
for page in reader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_texts(texts, embeddings)

# Load local model
llm = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin", verbose=True)

query = st.text_input("Ask a question:")
if query:
    docs = db.similarity_search(query)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    st.success(response)
