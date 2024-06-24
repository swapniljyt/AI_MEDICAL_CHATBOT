from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY =os.getenv("PINECONE_API_KEY")

embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
index_name = "medical-chatbot"
pc = Pinecone(PINECONE_API_KEY)
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

#Loading the index
docsearch = LangchainPinecone.from_texts(
    texts=[t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(
    model="/root/src/AI_MEDICAL_CHATBOT/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 100,  # Lowered from 512 for faster response
        'temperature': 0.7,  # Slightly reduced temperature
        'top_p': 0.9,
        'repetition_penalty': 1.0
    }
)


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)