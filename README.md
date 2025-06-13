# 🔮 Customer Churn Intelligence Platform

An end-to-end AI-powered web application for customer churn prediction, semantic document retrieval, and natural language SQL generation.

## 🧠 Key Features

- **Churn Prediction (TensorFlow)**  
  Predicts the probability of customer churn using a trained neural network model, deployed via FastAPI.

- **Retrieval-Augmented Generation (LangChain + Qdrant)**  
  Retrieves semantically relevant customer records from a vector store using OpenAI embeddings and LangChain's Self-QueryRetriever.

- **Text2SQL (LLaMA-3 via Groq API)**  
  Converts natural language questions to PostgreSQL queries using Groq-hosted LLaMA-3, with dynamic schema introspection.

- **Streamlit Frontend**  
  Stylish and interactive dashboard for entering customer data, asking SQL questions, and exploring vector-based RAG results.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, TensorFlow, LangChain, Qdrant, PostgreSQL  
- **Frontend**: Streamlit, Plotly, HTML/CSS customization  
- **LLM APIs**: OpenAI, Groq (LLaMA-3)  
- **Vector DB**: Qdrant (OpenAI Embeddings)

---

## ⚙️ Project Structure

📁 models/ # Trained churn prediction models (Keras format)
📄 app.py # Streamlit frontend (3-tab layout: Prediction, Text2SQL, RAG)
📄 backend.py # FastAPI endpoints for prediction, SQL generation, RAG
📄 RAG_BONUS.py # LangChain SelfQueryRetriever setup
📄 load_embeddings_BONUS.py # Loads vector store with chunked embeddings + metadata


---

##  How to Run

### 1. Start Backend
```bash
uvicorn backend:app --reload --port 8000
```

## Launch frontend 
```bash 
streamlit run app.py
```
### 3.Setup your env\

```bash
POSTGRES_PASS=your_password
QDRANT_API_KEY=your_qdrant_key
QDRANT_HOST=http://localhost:6333
OPENAI_API_KEY=your_openai_key
GROQ_API=your_groq_key
```



