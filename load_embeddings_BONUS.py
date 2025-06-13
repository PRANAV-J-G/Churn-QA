import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed from OpenAI
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def get_vector_store(qdrant_client: QdrantClient, collection_name: str = "churn_embeddings"):
    """
    Initialize and return a Qdrant vector store with HuggingFace embeddings
    """
    # Use HuggingFace embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    return vector_store

def load_documents_to_qdrant(documents: List[Document], 
                           qdrant_client: QdrantClient, 
                           collection_name: str = "churn_embeddings"):
    """
    Load documents into Qdrant vector store
    """
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store and add documents
    vector_store = Qdrant.from_documents(
        documents,
        embeddings,
        host="localhost",
        port=6333,
        collection_name=collection_name
    )
    
    return vector_store

def create_embeddings_from_csv(csv_file_path: str, 
                             qdrant_client: QdrantClient,
                             collection_name: str = "churn_embeddings"):
    """
    Create embeddings from CSV data and store in Qdrant
    """
    import pandas as pd
    
    # Read CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert DataFrame rows to Document objects
    documents = []
    for index, row in df.iterrows():
        # Create document content from row data
        content = f"""
        Customer ID: {row.get('customer_id', 'N/A')}
        Monthly Charges: {row.get('monthly_charges', 'N/A')}
        Total Charges: {row.get('total_charges', 'N/A')}
        Tenure: {row.get('tenure', 'N/A')} months
        Contract Type: {row.get('contract_type', 'N/A')}
        Internet Service: {row.get('internet_service', 'N/A')}
        Phone Service: {row.get('phone_service', 'N/A')}
        Gender: {row.get('gender', 'N/A')}
        Senior Citizen: {row.get('senior_citizen', 'N/A')}
        Partner: {row.get('partner', 'N/A')}
        Dependents: {row.get('dependents', 'N/A')}
        """
        
        # Create metadata
        metadata = {
            'customer_id': str(row.get('customer_id', 'N/A')),
            'monthly_charges': str(row.get('monthly_charges', 'N/A')),
            'total_charges': str(row.get('total_charges', 'N/A')),
            'tenure': str(row.get('tenure', 'N/A')),
            'contract_type': str(row.get('contract_type', 'N/A')),
            'source': 'customer_data'
        }
        
        doc = Document(page_content=content.strip(), metadata=metadata)
        documents.append(doc)
    
    # Load documents to Qdrant
    vector_store = load_documents_to_qdrant(documents, qdrant_client, collection_name)
    
    return vector_store, len(documents)

# Initialize embeddings for use in other modules
def get_embeddings():
    """
    Get HuggingFace embeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )