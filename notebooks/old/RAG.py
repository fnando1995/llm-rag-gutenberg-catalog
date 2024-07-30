import os
import numpy as np
import pickle
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# Define the CustomEmbeddings class
class CustomEmbeddings:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        return self.model.encode(documents, show_progress_bar=True)
from datasets import load_dataset

# Load the Hugging Face dataset
print("Loading dataset...")
dataset = load_dataset("kmfoda/booksum", split="test")

# Preprocess the text
texts = [item['chapter'] for item in dataset if item['chapter']]  # or item['summary_text']
# Create the embedding class instance
embedding_class = CustomEmbeddings('all-MiniLM-L6-v2')  # Lightweight model for CPU

# Create the Chroma vector store from texts
print("Creating Chroma vector store...")
vector_store = Chroma.from_texts(texts, embedding_class,persist_directory='embeds')

retriever = vector_store.as_retriever()