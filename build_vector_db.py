import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load OPENAI_API_KEY from the .env file
load_dotenv()

# Load job descriptions from the JSON file
jobs_path = "data/jobs.json"
with open(jobs_path) as f:
    jobs = json.load(f)

# Extract the plain-text descriptions and titles into separate lists
texts = [job["description"] for job in jobs]
metadatas = [{"title": job["title"]} for job in jobs]

# Initialize the OpenAI embedding model that converts text to vectors
embeddings = OpenAIEmbeddings()

# Convert each job description to a vector and store them in a FAISS index
vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

# Persist the FAISS index to disk so analyze_cv.py can load it later
index_path = "faiss_index"
vector_store.save_local(index_path)
print(f"FAISS index saved to '{index_path}' ({len(texts)} jobs indexed)")

# Run a quick sanity-check search to confirm the index works
query = "Expertise in LLMs and Python"
results = vector_store.similarity_search(query, k=1)
print(f"\nTop match for '{query}':")
print(results[0].metadata.get("title", ""), "-", results[0].page_content[:80], "...")
