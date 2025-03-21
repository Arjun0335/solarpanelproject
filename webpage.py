import streamlit as st
import openai
import pypdf
import numpy as np
from sentence_transformers import SentenceTransformer
import huggingface_hub
import pinecone
import os
from sklearn.neighbors import NearestNeighbors 

# OpenRouter API Key
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("❌ OpenRouter API Key not found! Set OPENROUTER_API_KEY in environment.")
openai.api_key = API_KEY

# Initialize Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    st.error("❌ Pinecone API Key not found! Set PINECONE_API_KEY in environment.")

pinecone.init(api_key=pinecone_api_key, environment="us-east1-aws")
index_name = "solar-ai-assistant"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)  # 384 is the embedding size of MiniLM
index = pinecone.Index(index_name)

# PDF Text Extraction using pypdf
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

pdf_files = ["ec3008-2016.pdf", "Standards-and-Requirements-for-Solar.pdf"]
pdf_texts = [extract_text_from_pdf(pdf) for pdf in pdf_files if pdf]
pdf_text = "\n".join(pdf_texts)

sentences = pdf_text.split(". ")
embeddings = model.encode(sentences)

# Nearest Neighbors Model (Replaces FAISS)
knn = NearestNeighbors(n_neighbors=3, metric="cosine")
knn.fit(embeddings)

# Store in Pinecone
pinecone_vectors = [
    (str(i), embeddings[i].tolist(), {"sentence": sentences[i]})
    for i in range(len(sentences))
]
index.upsert(vectors=pinecone_vectors)

# Retrieve context function (using Nearest Neighbors)
def retrieve_context(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = knn.kneighbors(query_embedding, n_neighbors=top_k)
    
    knn_results = [sentences[i] for i in indices[0]]

    pinecone_results = index.query(
        vector=query_embedding.tolist()[0], top_k=top_k, include_metadata=True
    )
    pinecone_texts = [match["metadata"]["sentence"] for match in pinecone_results.get("matches", []) if "metadata" in match]

    return "\n".join(set(knn_results + pinecone_texts))

# OpenAI Response
def get_ai_response(prompt):
    context = retrieve_context(prompt)
    full_prompt = f"Context: {context}\nQuestion: {prompt}"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ]
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
