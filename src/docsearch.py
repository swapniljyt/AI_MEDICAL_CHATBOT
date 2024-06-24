import pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import ServerlessSpec

def create_docsearch(embedding, text_chunks, api_key, index_name):
    # Initialize Pinecone
    pc=Pinecone(api_key=api_key, environment='us-east-1')
    
    # Check if the index already exists
    if index_name not in pc.list_indexes().names():
        # Create the index
        pinecone.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        
    # Create docsearch
    docsearch = LangchainPinecone.from_texts(
        texts=[t.page_content for t in text_chunks],
        embedding=embedding,
        index_name=index_name
    )
    
    return docsearch
