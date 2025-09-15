import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import fitz # PyMuPDF
import numpy as np
import faiss 
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

import ollama
client = ollama.Client(host='http://localhost:11434')

pdf_data_files = os.listdir("./data")
chunk_size = 1000

def get_pdf_chunks():
    pdf_chunks = []
    for file in pdf_data_files:
        if file.endswith(".pdf"):
            doc = fitz.open(f"./data/{file}")
            for page in doc:
                text = page.get_text()
                if text.strip():
                    for i in range(0, len(text), chunk_size):
                        chunk_text = text[i : i + chunk_size]
                        chunk = {
                                "content": chunk_text,
                                "source": file
                            }
                        pdf_chunks.append(chunk)
            doc.close()
        else:
            print(f"{file} file format not compatible yet...")
    return pdf_chunks

chunks = get_pdf_chunks()
chunk_contents = [chunk['content'] for chunk in chunks]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(chunk_contents, show_progress_bar=True)
print(f"   -> Embeddings created. The first chunk is now a list of {len(embeddings[0])} numbers.")

embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(np.array(embeddings).astype('float32'))
print(f"   -> FAISS index is built. It contains {index.ntotal} entries.")

user_question = "How do I apply for Peer Tutoring?"
question_embedding = embedding_model.encode([user_question])

k = 3
distances, indices = index.search(np.array(question_embedding).astype('float32'), k)
print(f"\nHere are the top {k} most relevant chunks for the question: '{user_question}'\n")

retrieved_context = ""
sources = set() # Use a set to avoid duplicate source names
for i, idx in enumerate(indices[0]):
    retrieved_context += chunks[idx]['content'] + "\n---\n"
    sources.add(chunks[idx]['source'])
    
prompt_template = f"""
CONTEXT:
{retrieved_context}

INSTRUCTIONS:
Based ONLY on the CONTEXT provided above, answer the following question.
Do not use any other information. If the answer is not in the context, say "I could not find the answer in the provided documents."

QUESTION:
{user_question}
"""

response = client.chat(model='mistral', messages=[
        {
            'role': 'user',
            'content': prompt_template
        },
    ])

print("\n--- Final Answer ---")
final_answer = response['message']['content']
print(final_answer)
print(f"\nSources: {', '.join(sources)}")