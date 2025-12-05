# RAG System - Quick Start Guide

## Overview

The Water Network Anomaly Detection RAG system consists of:
1. **LangGraph RAG Agent API** - Backend API server (port 8001)
2. **Streamlit Dashboard** - Web UI (port 8501)

---

## 🚀 Quick Start (Recommended)

### Start Everything at Once
```powershell
python -m src.rag.api.startup
```

This will automatically:
- Start the API server on http://localhost:8001
- Start the Streamlit dashboard on http://localhost:8501

Then open your browser and go to: **http://localhost:8501**

---

## 🔧 Individual Startup

### Option 1: Start Only API Server
```powershell
python -m src.rag.api.launcher
```
- API server runs on: http://localhost:8001
- API Docs: http://localhost:8001/docs

### Option 2: Start Only Dashboard
First ensure API server is already running, then:
```powershell
streamlit run src/rag/pages/streamlit_app.py
```
- Dashboard runs on: http://localhost:8501

---

## 📊 What You Can Do

### Via Dashboard (UI)
1. **Ingest Documents**
   - Upload single documents
   - Ingest entire directories
   - Load data from SQLite databases

2. **Ask Questions**
   - Query the knowledge base
   - Get contextual answers
   - Choose response mode (concise/verbose/internal)

3. **Manage Collections**
   - View all collections in ChromaDB
   - Browse documents by RBAC tags
   - Check vector database statistics

4. **Optimize System**
   - Adjust configuration parameters
   - Monitor performance

### Via API (Programmatic)
```python
import requests

# Health check
response = requests.get("http://localhost:8001/health")

# Ingest document
response = requests.post("http://localhost:8001/ingest", json={
    "text": "Anomaly detected: Water leak in Zone A",
    "doc_id": "incident_001",
    "company_id": 1
})

# Ask question
response = requests.post("http://localhost:8001/ask", json={
    "question": "What anomalies were detected?",
    "response_mode": "verbose",
    "top_k": 5
})

# Get statistics
response = requests.get("http://localhost:8001/vectordb/stats")
```

---

## 🔍 API Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check server health |
| `/status` | GET | Get agent status |
| `/docs` | GET | Swagger UI documentation |
| `/ingest` | POST | Ingest a document |
| `/ingest_directory` | POST | Ingest documents from directory |
| `/ingest_sqlite` | POST | Ingest data from SQLite table |
| `/ask` | POST | Ask question to RAG system |
| `/optimize` | POST | Optimize system configuration |
| `/vectordb/stats` | GET | Get vector database statistics |
| `/vectordb/collection/{name}` | GET | Get collection details |

---

## ⚙️ Advanced Options

### API Server Options
```powershell
# With auto-reload (development)
python -m src.rag.api.launcher --reload

# Custom port
python -m src.rag.api.launcher --port 8000

# Debug logging
python -m src.rag.api.launcher --log-level debug

# Specific host and port
python -m src.rag.api.launcher --host 127.0.0.1 --port 8001
```

### Startup Script Options
```powershell
# Start both with custom ports
python -m src.rag.api.startup --api-port 8000 --ui-port 8502

# Start API only
python -m src.rag.api.startup --api-only

# Start UI only (API must be running separately)
python -m src.rag.api.startup --ui-only

# Start API with auto-reload
python -m src.rag.api.startup --reload
```

---

## 🗂️ Directory Structure

```
src/rag/
├── api/                           # API server launchers
│   ├── launcher.py               # Main LangGraph API launcher
│   ├── startup.py                # Both API + Streamlit launcher
│   └── README.md
│
├── pages/
│   └── streamlit_app.py          # Streamlit dashboard UI
│
├── agents/
│   └── langgraph_agent/
│       ├── api.py                # FastAPI endpoints
│       └── langgraph_rag_agent.py # RAG agent logic
│
├── tools/
│   ├── services/
│   │   ├── vectordb_service.py   # ChromaDB client
│   │   └── llm_service.py        # LLM provider abstraction
│   ├── ingestion_tools.py        # Document ingestion
│   └── retrieval_tools.py        # Query & retrieval
│
├── config/
│   └── env_config.py             # Environment configuration
│
└── RAG_CONFIGURATION.md          # Full documentation
```

---

## 📦 Database Locations

- **SQLite Metadata**: `src/data/RAG/rag_metadata.db`
- **ChromaDB Vectors**: `src/data/RAG/chroma_db/`

Both are automatically created on first use.

---

## 🐛 Troubleshooting

### API Server Won't Start
```powershell
# Check if port 8001 is in use
netstat -ano | findstr :8001

# Use different port
python -m src.rag.api.launcher --port 8000
```

### Dashboard Can't Connect to API
1. Ensure API server is running: `http://localhost:8001/health`
2. Check firewall/network settings
3. Try custom API URL in dashboard sidebar

### Missing Dependencies
```powershell
pip install -r requirements.txt
```

### Module Import Errors
```powershell
# Set Python path
$env:PYTHONPATH="${env:PYTHONPATH};$(pwd)"
python -m src.rag.api.launcher
```

---

## 🔐 Integration with Water Anomaly Detection

### Typical Workflow
1. **Model detects anomaly** in sensor data
2. **Anomaly is ingested** into RAG: `POST /ingest`
3. **User queries** for explanation: `POST /ask`
4. **RAG retrieves context** from ChromaDB
5. **LLM generates explanation** based on historical data

### Example Code
```python
from src.model.model import predict_anomaly
import requests

# Step 1: Detect anomaly
sensor_data = {...}  # Your sensor data
anomaly = predict_anomaly(sensor_data)

# Step 2: Ingest into RAG
if anomaly['class'] != 'Normal':
    requests.post("http://localhost:8001/ingest", json={
        "text": f"Anomaly: {anomaly['class']} in Zone {anomaly['zone_id']}",
        "doc_id": f"alert_{anomaly['timestamp']}",
        "dept_id": anomaly['zone_id']
    })

# Step 3: Generate explanation
explanation = requests.post("http://localhost:8001/ask", json={
    "question": f"Why was {anomaly['class']} detected in this zone?",
    "dept_id": anomaly['zone_id'],
    "response_mode": "verbose"
})

print(explanation.json()['answer'])
```

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8001/docs (when server is running)
- **RAG Configuration**: `src/rag/RAG_CONFIGURATION.md`
- **Enhanced Architecture**: `src/rag/ENHANCED_RAG_ARCHITECTURE.md`
- **LangGraph Integration**: `src/rag/LANGGRAPH_INTEGRATION_COMPLETE.md`

---

## ✅ Verification Checklist

- [ ] API server starts without errors
- [ ] Dashboard loads at http://localhost:8501
- [ ] API health check passes: `curl http://localhost:8001/health`
- [ ] Swagger UI accessible at http://localhost:8001/docs
- [ ] Can ingest a test document
- [ ] Can ask a question and get response
- [ ] ChromaDB database files exist in `src/data/RAG/`

---

## 🎯 Summary

| Task | Command |
|------|---------|
| Start everything | `python -m src.rag.api.startup` |
| Start API only | `python -m src.rag.api.launcher` |
| View API docs | `http://localhost:8001/docs` |
| Access dashboard | `http://localhost:8501` |
| Check API health | `curl http://localhost:8001/health` |
| Ingest document | `POST http://localhost:8001/ingest` |
| Ask question | `POST http://localhost:8001/ask` |

---

For more information, see `RAG_CONFIGURATION.md` in the `src/rag/` directory.
