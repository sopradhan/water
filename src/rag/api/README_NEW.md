# RAG API Server & Streamlit Dashboard

Complete RAG system with unified launchers for API server and web dashboard.

## 🚀 Quick Start (Recommended)

Start everything at once:
```powershell
python -m src.rag.api.startup
```

Then open your browser: **http://localhost:8501**

This automatically starts:
- **API Server**: http://localhost:8001 (port 8001)
- **Dashboard**: http://localhost:8501 (port 8501)

---

## 🔧 Individual Components

### Option 1: API Server Only
```powershell
python -m src.rag.api.launcher
```
- Runs on: http://localhost:8001
- API Docs: http://localhost:8001/docs
- Health: http://localhost:8001/health

### Option 2: Dashboard Only
```powershell
streamlit run src/rag/pages/streamlit_app.py
```
- Runs on: http://localhost:8501
- Requires API server to be running separately

---

## 📊 Files in This Folder

| File | Purpose |
|------|---------|
| `launcher.py` | Start LangGraph API server only |
| `startup.py` | Start both API server + Streamlit dashboard |
| `__init__.py` | Package initialization |
| `README.md` | This file |
| `QUICKSTART.md` | Complete feature guide & examples |

---

## ⚙️ Configuration

### Environment Variables
```env
RAG_API_URL=http://localhost:8001    # API server URL (for dashboard)
RAG_HOST=0.0.0.0                     # API server host (default: 0.0.0.0)
RAG_PORT=8001                        # API server port (default: 8001)
RAG_LOG_LEVEL=info                   # Logging level (default: info)
APP_ENV=development                  # Environment (default: development)
STREAMLIT_PORT=8501                  # Dashboard port (default: 8501)
```

### API Server Options
```powershell
# Development with auto-reload
python -m src.rag.api.launcher --reload

# Custom port
python -m src.rag.api.launcher --port 8000

# Debug logging
python -m src.rag.api.launcher --log-level debug
```

### Startup Script Options
```powershell
# Start with custom ports
python -m src.rag.api.startup --api-port 8000 --ui-port 8502

# Start API only
python -m src.rag.api.startup --api-only

# Start UI only (requires separate API)
python -m src.rag.api.startup --ui-only

# Start with auto-reload
python -m src.rag.api.startup --reload
```

---

## 📡 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check server health |
| `/status` | GET | Get agent status |
| `/docs` | GET | Swagger UI |
| `/ingest` | POST | Ingest document |
| `/ingest_directory` | POST | Ingest directory |
| `/ingest_sqlite` | POST | Ingest SQLite table |
| `/ask` | POST | Ask question |
| `/optimize` | POST | Optimize system |
| `/vectordb/stats` | GET | VectorDB stats |
| `/vectordb/collection/{name}` | GET | Collection details |

---

## 🎨 Dashboard Features

1. **Ingest Documents**
   - Single document upload
   - Directory ingestion
   - SQLite database import
   
2. **Ask Questions**
   - Query knowledge base
   - Multiple response modes
   - RBAC filtering

3. **Manage Collections**
   - View ChromaDB collections
   - Browse documents
   - Check statistics

4. **Optimize System**
   - Adjust configuration
   - Monitor performance

---

## 🔍 Example Usage

### Ingest a Document
```python
import requests

requests.post("http://localhost:8001/ingest", json={
    "text": "Water leak detected in Zone A, pressure dropped to 32 PSI",
    "doc_id": "incident_001",
    "company_id": 1,
    "dept_id": 10
})
```

### Ask a Question
```python
import requests

response = requests.post("http://localhost:8001/ask", json={
    "question": "What anomalies were detected?",
    "response_mode": "verbose",
    "top_k": 5,
    "company_id": 1
})

print(response.json()['answer'])
```

### Check Statistics
```python
import requests

stats = requests.get("http://localhost:8001/vectordb/stats")
print(stats.json())
```

---

## 📂 Database Storage

- **SQLite Metadata**: `src/data/RAG/rag_metadata.db`
- **ChromaDB Vectors**: `src/data/RAG/chroma_db/`

Automatically created on first use.

---

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Use different port
python -m src.rag.api.launcher --port 8000
python -m src.rag.api.startup --api-port 8000 --ui-port 8502
```

### API Server Won't Start
```powershell
# Check dependencies
pip install -r requirements.txt

# Check if port is available
netstat -ano | findstr :8001
```

### Dashboard Can't Connect to API
1. Ensure API is running: `http://localhost:8001/health`
2. Check firewall settings
3. Use custom API URL in dashboard sidebar

### Missing Packages
```powershell
pip install chromadb langchain langchain-core langgraph uvicorn fastapi
```

---

## 📚 More Information

- **Quick Start Guide**: See `QUICKSTART.md` in this folder
- **Full Configuration**: See `src/rag/RAG_CONFIGURATION.md`
- **Architecture Details**: See `src/rag/ENHANCED_RAG_ARCHITECTURE.md`
- **LangGraph Integration**: See `src/rag/LANGGRAPH_INTEGRATION_COMPLETE.md`

---

## ✅ Verification

```powershell
# Test API health
curl http://localhost:8001/health

# Test API docs
# Visit: http://localhost:8001/docs

# Test dashboard
# Visit: http://localhost:8501
```

---

## 🎯 Summary

| Goal | Command |
|------|---------|
| Start everything | `python -m src.rag.api.startup` |
| Start API only | `python -m src.rag.api.launcher` |
| View API docs | `http://localhost:8001/docs` |
| Access dashboard | `http://localhost:8501` |
| Check API | `curl http://localhost:8001/health` |
| With auto-reload | `python -m src.rag.api.launcher --reload` |

---

## 📖 API Documentation

When the API server is running, interactive API documentation is available at:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI Schema**: http://localhost:8001/openapi.json
