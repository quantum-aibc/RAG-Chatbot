# RAG-Chatbot
import openai
import pinecone
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Set API Keys
OPENAI_API_KEY = "your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENV = "your-pinecone-environment"

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "rag-chatbot"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
vectorstore = LC_Pinecone(index_name=index_name, embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY))

# RAG-Based Chatbot Function
def build_rag_chatbot(query):
    """Retrieves relevant documents and generates responses."""
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Use the following context to answer: {context}\nQuery: {query}"}]
    )
    return response["choices"][0]["message"]["content"]

# Example Usage
query = "Explain the latest advancements in AI"
response = build_rag_chatbot(query)
print("Chatbot Response:", response)
