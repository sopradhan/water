# RAG Backend API Launcher

Unified API server launcher for the LangGraph RAG Agent system.

## Quick Start

### 1. Run API Server
```powershell
python -m src.rag.api.launcher
```
Server starts at: `http://localhost:8001`

### 2. Access API Documentation
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI Schema**: http://localhost:8001/openapi.json

### 3. Test Health Check
```powershell
curl http://localhost:8001/health
```

---

## Server Configuration

### Environment Variables
```env
RAG_HOST=0.0.0.0           # Server host (default: 0.0.0.0)
RAG_PORT=8001              # Server port (default: 8001)
RAG_LOG_LEVEL=info         # Logging level (default: info)
APP_ENV=development        # Environment type (default: development)
```

### Command Line Options
```bash
# Development with auto-reload
python -m src.rag.api.launcher --reload

# Custom port
python -m src.rag.api.launcher --port 8000

# Debug logging
python -m src.rag.api.launcher --log-level debug

# Custom host and port
python -m src.rag.api.launcher --host 127.0.0.1 --port 8001
```

---

## API Endpoints

### Health & Status
- **GET /health** - Check server health
- **GET /status** - Get agent status
- **GET /** - Root endpoint with API info

### Document Ingestion
- **POST /ingest** - Ingest single document
- **POST /ingest_directory** - Ingest all documents from directory
- **POST /ingest_sqlite** - Ingest data from SQLite table

### Query & Retrieval
- **POST /ask** - Ask question to RAG system
- **POST /optimize** - Optimize system configuration

### Vector Database
- **GET /vectordb/stats** - Get vector database statistics
- **GET /vectordb/collection/{collection_name}** - Get collection details

### Documentation
- **GET /docs** - Swagger UI
- **GET /redoc** - ReDoc documentation
- **GET /openapi.json** - OpenAPI schema

---

## Example Requests

### 1. Ingest a Document
```bash
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Water leakage detected in Zone A at 14:30 UTC",
    "doc_id": "incident_001",
    "company_id": 1,
    "dept_id": 10
  }'
```

### 2. Ask a Question
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are recent anomalies in the water network?",
    "response_mode": "verbose",
    "top_k": 5,
    "company_id": 1
  }'
```

### 3. Get Vector Database Statistics
```bash
curl http://localhost:8001/vectordb/stats
```

### 4. Ingest from Directory
```bash
curl -X POST http://localhost:8001/ingest_directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "./src/data/documents",
    "file_extensions": ["txt", "md", "pdf"],
    "recursive": true,
    "company_id": 1
  }'
```

---

## Architecture

```
src/rag/
├── api/                                  # ← API Server (this folder)
│   ├── __init__.py
│   ├── run_server.py                    # Uvicorn server trigger
│   └── README.md                        # This file
│
├── agents/
│   └── langgraph_agent/
│       ├── api.py                       # FastAPI app & routes
│       └── langgraph_rag_agent.py       # RAG agent logic
│
├── tools/
│   ├── services/
│   │   ├── vectordb_service.py          # ChromaDB client
│   │   └── llm_service.py               # LLM provider abstraction
│   ├── ingestion_tools.py               # Document ingestion
│   └── retrieval_tools.py               # Query & retrieval
│
├── config/
│   └── env_config.py                    # Environment configuration
│
└── RAG_CONFIGURATION.md                 # Full documentation
```

---

## Deployment

### Local Development
```bash
python -m src.rag.api.launcher --reload --log-level debug
```

### Docker (Optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8001
CMD ["python", "-m", "src.rag.api.launcher", "--host", "0.0.0.0"]
```

### Production
```bash
python -m src.rag.api.launcher \
  --host 0.0.0.0 \
  --port 8001 \
  --log-level warning
```

---

## Troubleshooting

### Port Already in Use
```bash
# Use different port
python -m src.rag.api.run_server --port 8002
```

### ChromaDB Connection Issues
- Ensure `src/data/RAG/` directory exists
- Check `.env` file has correct `CHROMA_DB_PATH`
- Run: `python -c "from src.rag.tools.services.vectordb_service import VectorDBService; VectorDBService()"`

### Missing Dependencies
```bash
pip install -r requirements.txt
# Or just RAG packages:
pip install chromadb langchain langchain-core langgraph uvicorn fastapi
```

### Module Import Errors
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src.rag.api.run_server
```

---

## Monitoring

### Access logs are printed to console
```
2025-12-06 10:30:45 - INFO - Started server process [1234]
2025-12-06 10:30:45 - INFO - Uvicorn running on http://0.0.0.0:8001
2025-12-06 10:31:02 - INFO - POST /ingest - "200 OK"
```

### Check server health
```bash
curl http://localhost:8001/health | python -m json.tool
```

---

## Integration with Water Anomaly Detection

The RAG API works in conjunction with the water network anomaly detection system:

1. **Data Flow**:
   - Anomaly detection model generates alerts
   - Alerts ingested into RAG via `/ingest` endpoint
   - Users query via `/ask` endpoint
   - RAG retrieves context from ChromaDB
   - LLM synthesizes explanations

2. **Example Workflow**:
   ```python
   # 1. Detect anomaly
   anomaly = model.predict(sensor_data)  # e.g., Leak in Zone A
   
   # 2. Ingest into RAG
   requests.post("http://localhost:8001/ingest", json={
       "text": f"Anomaly detected: {anomaly}",
       "doc_id": f"alert_{timestamp}",
       "company_id": zone_id
   })
   
   # 3. Query for explanation
   response = requests.post("http://localhost:8001/ask", json={
       "question": "Why was this anomaly detected?",
       "company_id": zone_id
   })
   ```

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `launcher.py` | Main API server launcher |
| `__init__.py` | Package initialization |
| `README.md` | This documentation |

---

## Support

For issues or questions:
1. Check `RAG_CONFIGURATION.md` in parent folder
2. Review API response codes and error messages
3. Check server logs with `--log-level debug`
4. Verify `.env` configuration
