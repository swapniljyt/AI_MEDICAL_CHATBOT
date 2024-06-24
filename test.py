from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from src.docsearch import create_docsearch
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

extracted_data = load_pdf("/root/src/AI_MEDICAL_CHATBOT/data/Medical_book.pdf")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

PINECONE_API_KEY =os.getenv("PINECONE_API_KEY")

index_name = "medical-chatbot"

docsearch= create_docsearch(embeddings, text_chunks, PINECONE_API_KEY, index_name)

docsearch.as_retriever()

query = "What is fever"

docs=docsearch.similarity_search(query, k=3)

print("Result", docs)