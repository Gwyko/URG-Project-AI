# Intereacts with the system files
import os
import json
import random

import concurrent.futures
from functools import partial
from tqdm import tqdm

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
data_folder_path = "./data"
embed_index_path = "vector_index"

embed_model = OllamaEmbeddings(model="nomic-embed-text")
LLM_model = "mistral" 
llm = Ollama(model=LLM_model)

c_size = 500
c_overlap = 100

DEBUG = False
PROCESSED_CHUNKS_FILE = 'processed_chunks.json'

# finding data folder
if os.path.exists(data_folder_path):
    print("Found data folder")
else:
    print("Data folder not found")

# Stage 1: loading pdfs within the data folder
print("Searching for compatible PDF files...")
pdf_files = [f for f in os.listdir(data_folder_path) if f.endswith(".pdf")] #can make it compatible with DOCX
rnd_files = [f for f in os.listdir(data_folder_path) if not f.endswith(".pdf")] #checks for incompatible files
if rnd_files: print("\n### WARNING ###", "\nIncompatible Files:\n", rnd_files, f"Total: {len(rnd_files)}\n")
if not pdf_files: 
    print("No PDF files found!")
    exit()
else:
    print("Compatible Files:\n", f"Total: {len(pdf_files)}\n")

# Stage 2: splitting files into data chunks
split_docs = []

if os.path.exists(PROCESSED_CHUNKS_FILE): #checks if there is processed chunks previously
    print(f"Loading pre-processed chunks from '{PROCESSED_CHUNKS_FILE}'...")
    with open(PROCESSED_CHUNKS_FILE, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
        split_docs = [Document(page_content=chunk['page_content'], metadata=chunk['metadata']) for chunk in chunks_data]
else:
    t_split = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_folder_path, pdf_file)
        print(f" > Processing {pdf_file}...")

        #creates a raw pdf file of every document
        documents = PyPDFLoader(pdf_path).load() #loads pdf into documents
      

        for doc in documents:
            lines = doc.page_content.split('\n') #re-organises the paragraph in lines
            chunks = t_split.split_text(doc.page_content) #recursive split
            page_number = doc.metadata.get('page')

            for chunk in chunks: #adds meta data for each chunk for extra data
                new_doc = Document(
                    page_content = chunk,
                    metadata={
                        "source": os.path.basename(doc.metadata.get('source', pdf_file)),
                        "page": page_number # Add the page number here
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

# Stage 3: create and store embeddings
if os.path.exists(embed_index_path):
    print(f"Loading existing vector store from '{embed_index_path}'...")
    vector_store = FAISS.load_local(embed_index_path, embed_model, allow_dangerous_deserialization=True)
else:
    print("Creating Embeddings")
    vector_store = FAISS.from_documents(split_docs, embed_model)
    vector_store.save_local(embed_index_path)
    print(f"New vector store created and saved to '{embed_index_path}'.")

# Stage 4: THE RAG CHAIN
print("Building the Rag...")

fun_prompts = [
    "Got a question about the IT Faculty? Ask away! (or 'exit' to quit): ",
    "Curious about IT courses at Sohar University? Type your question (or 'exit' to quit): ",
    "What do you want to know about IT research or faculty at Sohar Uni? (or 'exit' to quit): ",
    "Exploring IT opportunities? Ask about programs or facilities! (or 'exit' to quit): ",
    "Ready to dive into Sohar University's IT world? Inquire here (or 'exit' to quit): "
]

verify_question =  PromptTemplate (
    template="""You are a classification assistant. Your task is to determine if a user's input is a genuine question related to Sohar University or its IT Faculty.
Do not answer the question. Just classify it.

- Greetings like 'hello', 'hi' are NOT valid questions.
- Simple statements or off-topic questions are NOT valid.
- Questions about the university, its courses, staff, facilities, or the IT faculty ARE valid and even if its about the student academic itself.

Look at the user input below and respond with only a single word: True or False.

User input: {verify}
Answer:
""",
    input_variables=["verify"],
)

prompt_enhancer_template = PromptTemplate(
    template="""
You are a helpful assistant. Your job is to take a user's question and rewrite it to be more detailed and specific for a retriever system at Sohar University.

Original question: {question}

Rewrite the question to include more keywords and be very clear. For example, if the user asks 'who is vc?', you should rewrite it as 'Who is the Vice Chancellor of Sohar University?'.

Rewritten question:
""",
    input_variables=["question"],
)

prompt = PromptTemplate(
    template="""
You are an expert AI bot for Sohar University. You answer questions based ONLY on the provided context.
Your task is to answer the user's question by providing a list of facts, in a more engaging tone.

Rules:
1. For EVERY single fact or statement in your answer, you MUST provide a direct citation from the context in the format [Source: file.pdf, Page: 1].
2. The citation must come from the context provided below.
3. Do not include any facts if you cannot find their source in the context.
4. If the context does not contain the information needed to answer the question, you MUST respond with ONLY this exact phrase: "The provided documents do not contain an answer to this question."

<context>
{context}
</context>

Question: {input}

Answer (summarize the contexts given then conclude with a short answer unless instructed to do so ending with the citation.):

""",
    input_variables=["context", "input"],
)

print("RAG chain is ready. You can now ask questions.")

# THIS THE CORE THE BRAIN THE EVERYTHINGGG!!!!
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

verification_chain = (
    {"verify": RunnablePassthrough()}
    | verify_question
    | llm
    |StrOutputParser()
)
prompt_enhancer_chain = (
    {"question": RunnablePassthrough()}
    | prompt_enhancer_template
    | llm
    | StrOutputParser()
)
rag_chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    try:
        query = input(random.choice(fun_prompts))
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue

        print("Verifying Question...")
        verification_result = verification_chain.invoke(query) #Checks if real question
        print("Verfication Status: ", verification_result)
        if not "True" in verification_result:
            print("\n--- Answer ---")
            print("Please ask a question related to the Sohar University or IT faculty.")
            print("-" * 50)
            continue

        print("\nEnhancing question...")
        enhanced_question = prompt_enhancer_chain.invoke(query) # Enhances Question

        print(enhanced_question)

        print("thinking...")
        answer = rag_chain.invoke(enhanced_question) # Invokes LLM

        print("\n--- Answer ---")
        print(answer)
        print("-" * 50)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        break

print("Exiting the RAG system. Goodbye!")