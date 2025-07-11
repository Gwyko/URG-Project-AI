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
data_folder = "./data"
embed_model = "nomic-embed-text"
LLM_model = "mistral"

c_size = 500
c_overlap = 100

DEBUG = False
PROCESSED_CHUNKS_FILE = 'processed_chunks.json'

# finding data folder
if os.path.exists(data_folder):
    print("Found data folder")
else:
    print("Data folder not found")

# Stage 1: loading pdfs within the data folder
print("Searching for compatible PDF files...")
pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")] #can make it compatible with DOCX
rnd_files = [f for f in os.listdir(data_folder) if not f.endswith(".pdf")] #checks for incompatible files
if rnd_files: print("\n### WARNING ###", "\nIncompatible Files:\n", rnd_files, f"Total: {len(rnd_files)}\n")
if not pdf_files: 
    print("No PDF files found!")
    exit()
else:
    print("Compatible Files:\n", pdf_files, f"Total: {len(pdf_files)}\n")

# Stage 2: splitting files into data chunks

split_docs = [] #this is where we will store the chunks in memory

if os.path.exists(PROCESSED_CHUNKS_FILE): #checks if there is processed chunks previously
    print(f"Loading pre-processed chunks from '{PROCESSED_CHUNKS_FILE}'...")
    with open(PROCESSED_CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        split_docs = [Document(page_content=chunk['page_content'], metadata=chunk['metadata']) for chunk in chunks_data]
else:
    t_split = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder, pdf_file)
        print(f" > Processing {pdf_file}...")

        #creates a raw pdf file of every document
        documents = PyPDFLoader(pdf_path).load() #loads pdf into documents of chunks

        for doc in documents:
            lines = doc.page_content.split('\n') #re-organises the paragraph in lines
            chunks = t_split.split_text(doc.page_content) #recursive split

            for chunk in chunks: #adds meta data for each chunk for extra data
                new_doc = Document(
                    page_content = chunk,
                    metadata={
                        "source": os.path.basename(doc.metadata.get('source', pdf_file)),
                    }
                )
                split_docs.append(new_doc)
    
    # saving the chunks to avoid re-creation of chunks
    modified_chunks = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in split_docs]

    # creates a file and writes/inserts the chunks
    with open(PROCESSED_CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(modified_chunks, f)
    print("Saved chunks for future runs", PROCESSED_CHUNKS_FILE)

print(f"Processed Chunks: {len(split_docs)}")