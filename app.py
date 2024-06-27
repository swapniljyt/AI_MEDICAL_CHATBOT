from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from src.docsearch import create_docsearch
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

extracted_data = load_pdf("/root/src/AI_MEDICAL_CHATBOT/data/Medical_book.pdf")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

PINECONE_API_KEY =os.getenv("PINECONE_API_KEY")

index_name = "medical-chatbot"

docsearch= create_docsearch(embeddings, text_chunks, PINECONE_API_KEY, index_name)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

from langchain.llms import CTransformers
config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 'context_length': 4000, 'temperature':0, 'gpu_layers':40}
llm = CTransformers(model="/root/src/AI_MEDICAL_CHATBOT/model/llama-2-7b-chat.ggmlv3.q4_0.bin", 
                    model_type = "llama", gpu_layers=40, config=config)

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