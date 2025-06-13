import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import OpenAI  # Updated import
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from openai import OpenAI as GroqClient

load_dotenv()

# Initialize Groq client
groq_client = GroqClient(
    api_key=os.getenv("GROQ_API"),
    base_url='https://api.groq.com/openai/v1'
)

# Query skeleton for formatting
skeleton_query = """
Based on the customer data, please find customers that match the following criteria: {}

Please provide detailed information about each matching customer including:
- Customer ID
- Monthly Charges
- Total Charges  
- Tenure (months)
- Contract Type
- Services used
"""

class GroqLLM:
    """Custom LLM wrapper for Groq API"""
    
    def __init__(self, model_name="llama3-70b-8192"):
        self.model_name = model_name
        self.client = groq_client
    
    def __call__(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate(self, prompts: List[str], **kwargs):
        """For compatibility with LangChain"""
        results = []
        for prompt in prompts:
            results.append([self(prompt, **kwargs)])
        return results

class CustomRetriever(BaseRetriever):
    """Custom retriever using Qdrant vector store"""
    
    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            # Use similarity search with Qdrant
            docs = self.vector_store.similarity_search(
                query, 
                k=5  # Return top 5 most similar documents
            )
            return docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents"""
        return self.get_relevant_documents(query)

def get_vector_store(qdrant_client: QdrantClient, collection_name: str = "churn_embeddings"):
    """Get vector store with HuggingFace embeddings"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    return vector_store

def initialize_qa():
    """Initialize the QA system with Groq LLM and Qdrant retriever"""
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            host="localhost",
            port=6333
        )
        
        # Get vector store
        vector_store = get_vector_store(qdrant_client)
        
        # Create custom retriever
        retriever = CustomRetriever(vector_store)
        
        # Initialize Groq LLM
        llm = GroqLLM()
        
        # Create a simple QA chain-like object
        class SimpleQA:
            def __init__(self, retriever, llm):
                self.retriever = retriever
                self.llm = llm
            
            def get_relevant_documents(self, query: str):
                return self.retriever.get_relevant_documents(query)
            
            def run(self, query: str):
                # Get relevant documents
                docs = self.retriever.get_relevant_documents(query)
                
                # Combine document content
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create prompt
                prompt = f"""
                Based on the following customer data, answer the query: {query}
                
                Customer Data:
                {context}
                
                Please provide a detailed answer based on the customer information above.
                """
                
                # Generate response
                response = self.llm(prompt)
                return response
        
        qa = SimpleQA(retriever, llm)
        return qa
        
    except Exception as e:
        print(f"Error initializing QA system: {e}")
        # Return a dummy QA object that returns empty results
        class DummyQA:
            def get_relevant_documents(self, query: str):
                return []
            def run(self, query: str):
                return "QA system not available"
        
        return DummyQA()

# Initialize QA system
qa = initialize_qa()