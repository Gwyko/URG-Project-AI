# Intereacts with the system files
import os
import json

# vvv File type loader vvv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

# vvv Paragraph Splitters vvv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# vvv AI MODELS vvv
from langchain_community.embeddings import OllamaEmbeddings # translates into vectors
from langchain_community.vectorstores import FAISS # index no need to convert to vector everytime
from langchain_community.llms import Ollama

# vvv Makes your life easier vvv
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Config
data = "./data"
embed_model = "nomic-embed-text"
LLM_model = "mistral"

DEBUG = False

# finding data folder
if os.path.exists(data):
    print("Found data folder")
else:
    print("Data folder not found")

# loading pdfs within the data folder
pdf_files = [f for f in os.listdir(data) if f.endswith(".pdf")] #can make it compatible with DOCX
if not pdf_files: print("No DPF files found!")

