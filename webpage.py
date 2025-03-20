import streamlit as st
import openai
import fitz  # PyMuPDF for PDF extraction
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pinecone
import os
# OpenRouter API Key (Replace with your own)
API_KEY = os.getenv("OPENROUTER_API_KEY")
openai.api_key = API_KEY

# Initialize Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-aws")
index_name = "solar-ai-assistant"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)  # 384 is the embedding size of MiniLM
index = pinecone.Index(index_name)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Extract and encode PDF contents
pdf_files = ["ec3008-2016.pdf", "Standards-and-Requirements-for-Solar"]
pdf_text = "\n".join([extract_text_from_pdf(pdf) for pdf in pdf_files])
sentences = pdf_text.split(". ")
embeddings = model.encode(sentences)

# Store in FAISS
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(np.array(embeddings))

# Store in Pinecone
pinecone_vectors = [(str(i), embeddings[i].tolist(), {"sentence": sentences[i]}) for i in range(len(sentences))]
index.upsert(vectors=pinecone_vectors)

def retrieve_context(query, top_k=3):
    query_embedding = model.encode([query])
    _, faiss_indices = faiss_index.search(query_embedding, top_k)
    faiss_results = [sentences[i] for i in faiss_indices[0]]
    
    pinecone_results = index.query(vector=query_embedding.tolist()[0], top_k=top_k, include_metadata=True)
    pinecone_texts = [match["metadata"]["sentence"] for match in pinecone_results["matches"]]
    
    return "\n".join(set(faiss_results + pinecone_texts))

def get_ai_response(prompt):
    context = retrieve_context(prompt)
    full_prompt = f"Context: {context}\nQuestion: {prompt}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("Solar Industry AI Assistant ☀️")
st.write("Ask me anything about solar energy!")

# User input
user_query = st.text_input("Your question:")

if st.button("Ask"):  
    if user_query:
        response = get_ai_response(user_query)
        st.write("### AI Response:")
        st.write(response)
    else:
        st.warning("Please enter a question!")
