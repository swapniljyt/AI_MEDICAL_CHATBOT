from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("/root/src/AI_MEDICAL_CHATBOT/data/Medical_book.pdf")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
index_name = "medical-chatbot"
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )


#Creating Embeddings for Each of The Text Chunks & storing
docsearch = LangchainPinecone.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
)