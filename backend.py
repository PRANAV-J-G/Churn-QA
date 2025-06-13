from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from dotenv import load_dotenv
from langchain.schema import Document
from RAG_BONUS import skeleton_query, qa
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed from OpenAI
from load_embeddings_BONUS import *
# Removed deprecated imports

app = FastAPI()
load_dotenv()

class ChurnDatabaseSetup:
    def __init__(self, host: str = "localhost",
                 user: str = "postgres",
                 password: str = os.getenv('POSTGRES_PASS'),
                 port: int = 5432):
        self.connection = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            port=port
        )
        self.cursor = self.connection.cursor()
        
    def close(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
            
model = load_model(os.path.join('models','NeuralNetwork.h5'))

# Groq client setup
client = OpenAI(
    api_key=os.getenv("GROQ_API"),
    base_url='https://api.groq.com/openai/v1'
)

class ChurnFeatures(BaseModel):
    features: list

@app.post("/predict")
async def predict(data: ChurnFeatures):
    prediction = model.predict(np.array([data.features]))
    return {"prediction": float(prediction[0][0])}

class Query(BaseModel):
    query: str

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': os.getenv('POSTGRES_PASS'),
    'database': 'churn',
    'port': 5432
}

def get_schema() -> Dict:
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT table_name, column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_catalog = %s
                ORDER BY table_name
            """, (DB_CONFIG['database'],))
            
            schema = {}
            for row in cursor.fetchall():
                table = row['table_name']
                if table not in schema:
                    schema[table] = []
                schema[table].append({
                    'column': row['column_name'],
                    'type': row['data_type']
                })
            return schema

@app.post("/generate_sql")
async def generate_sql(query: Query):
    schema = get_schema()
    schema_text = "Database Schema:\n"
    for table, columns in schema.items():
        schema_text += f"\nTable: {table}\n"
        for col in columns:
            schema_text += f"- {col['column']} ({col['type']})\n"

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{
            "role": "system", 
            "content": f"You are a SQL assistant. {schema_text}\nConvert this question to SQL (PostgreSQL syntax): {query.query}\nReturn only the SQL query. Write SQL without using markdown or code fences. Just give me the plain SQL."
        }],
        temperature=0,
        top_p=1,
    )
    
    sql = response.choices[0].message.content.strip()
    
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()  # Fetch all rows as dictionaries
                # Get column names from cursor description
                columns = [column[0] for column in cursor.description] if cursor.description else []
                
                # Convert RealDictRow objects to regular dictionaries
                results_list = [dict(row) for row in results]
                
        return {
            "sql": sql, 
            "status": "success",
            "results": results_list,  # List of dictionary rows
            "columns": columns   # List of column names
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e), "sql": sql})

class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[DocumentResponse]
    confidence: Optional[float] = None

@app.post("/rag_query")
async def process_rag_query(request: QueryRequest):
    try:
        # Format the query using the skeleton
        formatted_query = skeleton_query.format(request.query)
        
        # Get relevant documents using the retriever
        docs = qa.get_relevant_documents(formatted_query)
        
        # Process the results
        sources = []
        for doc in docs:
            # Convert Document objects to digestible format
            source = DocumentResponse(
                content=doc.page_content,
                metadata=doc.metadata
            )
            sources.append(source)
            
        # Calculate a simple confidence score based on number of relevant documents
        confidence = min(len(sources) / 5, 1.0)  # Normalize to 0-1 range, max at 5 docs
        
        # Combine the documents into a coherent answer
        combined_answer = "Based on the query, here are the relevant customers:\n\n"
        for idx, source in enumerate(sources, 1):
            combined_answer += f"Customer {idx}:\n"
            combined_answer += f"- ID: {source.metadata.get('customer_id', 'N/A')}\n"
            combined_answer += f"- Monthly Charges: ${source.metadata.get('monthly_charges', 'N/A')}\n"
            combined_answer += f"- Total Charges: ${source.metadata.get('total_charges', 'N/A')}\n"
            combined_answer += f"- Tenure: {source.metadata.get('tenure', 'N/A')} months\n"
            combined_answer += f"- Contract Type: {source.metadata.get('contract_type', 'N/A')}\n\n"
        
        return QueryResponse(
            answer=combined_answer,
            sources=sources,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Add this to run the server directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)