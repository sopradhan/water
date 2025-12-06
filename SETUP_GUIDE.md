# Water Anomaly Detection RAG System - Setup Guide

## Quick Start

### 1. Prerequisites
- Python 3.8+
- pip or conda
- Ollama (for LLM & embeddings) - [Download](https://ollama.ai)

### 2. Environment Setup

```bash
# Clone repository (if not already done)
git clone <repo-url>
cd water

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Ollama Service

```bash
# Start Ollama in a separate terminal/tab
ollama serve

# In another terminal, pull required models
ollama pull mistral
ollama pull nomic-embed-text
```

Ollama will run on `http://localhost:11434` by default.

### 4. Initialize Database

```bash
python check_db.py
```

This creates SQLite tables for:
- `document_metadata` - Document information
- `rag_history_and_optimization` - Query history and RL feedback
- `chunk_embedding_data` - Vector embeddings
- `agent_memory` - RL agent state

### 5. Start RAG API Server

```bash
python src/rag/api/launcher.py
```

Server will run on `http://localhost:8001`

### 6. (Optional) Start Model API Server

```bash
python src/model/api.py
```

Model API will run on `http://localhost:8002`

---

## Configuration

All paths and configurations are in `src/config/paths_config.json`:

```json
{
  "database": {
    "rag_metadata_db": "src/data/RAG/rag_metadata.db",
    "collection_name": "water_anomaly_detection"
  },
  "vectordb": {
    "chroma_db_path": "src/data/RAG/chroma_db"
  },
  "api": {
    "rag_port": 8001,
    "model_port": 8002
  }
}
```

### Environment Overrides

Set these environment variables to override config:

```bash
# Database path
export RAG_DB_PATH="custom/path/database.db"

# Vector DB path
export CHROMA_DB_PATH="custom/path/chroma_db"

# LLM provider (default: http://localhost:11434)
export LLM_BASE_URL="http://custom-llm:11434"

# Embedding provider
export EMBEDDING_BASE_URL="http://custom-embedding:11434"
```

---

## RAG API Usage

Base URL: `http://localhost:8001`

### 1. Health Check

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "operational",
  "rag_agent": "ready",
  "vectordb": "ready"
}
```

---

### 2. Ingest Documents

```bash
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "water_manual_001",
    "doc_title": "Water System Manual",
    "doc_type": "manual",
    "content": "Water pressure should be maintained between 40-80 PSI...",
    "company_id": 1,
    "dept_id": 101
  }'
```

Response:
```json
{
  "success": true,
  "doc_id": "water_manual_001",
  "chunks_created": 5,
  "message": "Document ingested successfully"
}
```

---

### 3. Ask Questions - CONCISE MODE (Default - User Friendly)

**Best for:** End users who want quick, clean answers

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is normal water pressure?",
    "response_mode": "concise",
    "company_id": 1,
    "dept_id": 101,
    "top_k": 5
  }'
```

Response:
```json
{
  "success": true,
  "question": "What is normal water pressure?",
  "answer": "Normal water pressure is typically maintained between 40-80 PSI.",
  "session_id": "sess_abc123",
  "context_chunks": 2,
  "guardrails_applied": true,
  "errors": []
}
```

**Features:**
- ✅ Clean, concise answers
- ✅ Guardrails applied (PII redaction, safety checks)
- ✅ Smart responses when content blocked
- ✅ RBAC filtering by company/dept
- ❌ No debug information

---

### 4. Ask Questions - INTERNAL MODE (System Integration)

**Best for:** Backend systems, database updates

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is normal water pressure?",
    "response_mode": "internal",
    "company_id": 1,
    "dept_id": 101
  }'
```

Response:
```json
{
  "success": true,
  "answer": "Normal water pressure is typically maintained between 40-80 PSI.",
  "quality_score": 0.92,
  "sources_count": 2,
  "source_docs": [
    {"doc_id": "water_manual_001", "chunk_id": "chunk_0"},
    {"doc_id": "water_manual_001", "chunk_id": "chunk_1"}
  ],
  "metadata": {
    "session_id": "sess_abc123",
    "timestamp": 1702000000.123,
    "model": "langgraph_rag_agent",
    "execution_time_ms": 245
  },
  "guardrails_applied": false,
  "errors": []
}
```

**Features:**
- ✅ Structured data for database storage
- ✅ Quality scores and metadata
- ✅ Source document tracking
- ✅ Execution metrics
- ❌ No guardrails
- ❌ No debug info

---

### 5. Ask Questions - VERBOSE MODE (Engineering/Debug)

**Best for:** Engineers, RAG admins, debugging

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is normal water pressure?",
    "response_mode": "verbose",
    "company_id": 1,
    "dept_id": 101,
    "top_k": 5
  }'
```

Response:
```json
{
  "success": true,
  "question": "What is normal water pressure?",
  "answer": "Normal water pressure is typically maintained between 40-80 PSI.",
  "sources": [
    {
      "content": "Water pressure should be maintained between 40-80 PSI...",
      "metadata": {
        "doc_id": "water_manual_001",
        "chunk_id": "chunk_0",
        "relevance_score": 0.95
      }
    }
  ],
  "sources_count": 2,
  "traceability": {
    "retrieval_method": "hybrid",
    "reranking_used": true,
    "reranked_context": [...]
  },
  "retrieval_quality": 0.92,
  "optimization_applied": false,
  "rl_action": "SKIP",
  "rl_recommendation": {
    "action": "MONITOR",
    "confidence": 0.87,
    "expected_improvement": 0.05,
    "learning_stats": {...}
  },
  "execution_time_ms": 245,
  "session_id": "sess_abc123",
  "visualization_data": {...},
  "guardrails_applied": false,
  "errors": []
}
```

**Features:**
- ✅ Full source documents
- ✅ Traceability information
- ✅ RL recommendations
- ✅ Execution metrics
- ✅ Visualization data
- ✅ All debug information
- ❌ No guardrails

---

### 6. Submit User Feedback (for RL training)

```bash
curl -X POST http://localhost:8001/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sess_abc123",
    "rating": 4,
    "feedback_text": "Good answer, but could be more detailed"
  }'
```

Response:
```json
{
  "success": true,
  "session_id": "sess_abc123",
  "rating": 4,
  "reward_signal": 0.75,
  "message": "Feedback recorded and RL agent updated"
}
```

---

## Model API Usage

Base URL: `http://localhost:8002`

### 1. Health Check

```bash
curl http://localhost:8002/health
```

Response:
```json
{
  "status": "operational",
  "models": {
    "knn": "loaded",
    "lstm": "loaded"
  },
  "version": "1.0"
}
```

---

### 2. Predict Water Anomaly

```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pressure": 65,
    "temperature": 22,
    "ph_level": 7.2,
    "dissolved_oxygen": 8.5,
    "turbidity": 0.3,
    "flow_rate": 150,
    "location": "valve_a",
    "sensor_type": "digital"
  }'
```

Response:
```json
{
  "success": true,
  "prediction": {
    "knn_prediction": "normal",
    "knn_confidence": 0.92,
    "lstm_prediction": "normal",
    "lstm_confidence": 0.87,
    "ensemble_prediction": "normal",
    "ensemble_confidence": 0.89
  },
  "analysis": {
    "anomaly_detected": false,
    "risk_level": "low",
    "features_used": [
      "pressure", "temperature", "ph_level", "dissolved_oxygen",
      "turbidity", "flow_rate"
    ]
  },
  "execution_time_ms": 45,
  "model_version": "v1.0"
}
```

---

### 3. Batch Predict

```bash
curl -X POST http://localhost:8002/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "pressure": 65,
        "temperature": 22,
        "ph_level": 7.2,
        "dissolved_oxygen": 8.5,
        "turbidity": 0.3,
        "flow_rate": 150,
        "location": "valve_a",
        "sensor_type": "digital"
      },
      {
        "pressure": 120,
        "temperature": 28,
        "ph_level": 6.5,
        "dissolved_oxygen": 4.2,
        "turbidity": 2.1,
        "flow_rate": 250,
        "location": "valve_b",
        "sensor_type": "digital"
      }
    ]
  }'
```

Response:
```json
{
  "success": true,
  "predictions": [
    {
      "knn_prediction": "normal",
      "lstm_prediction": "normal",
      "ensemble_prediction": "normal",
      "ensemble_confidence": 0.89,
      "anomaly_detected": false
    },
    {
      "knn_prediction": "anomaly",
      "lstm_prediction": "anomaly",
      "ensemble_prediction": "anomaly",
      "ensemble_confidence": 0.94,
      "anomaly_detected": true,
      "risk_level": "high"
    }
  ],
  "total_processed": 2,
  "anomalies_found": 1,
  "execution_time_ms": 85
}
```

---

## Combined Workflow Example

### Complete Integration Flow

```bash
#!/bin/bash

# 1. Start services
echo "Starting Ollama..."
ollama serve &

echo "Starting RAG API..."
python src/rag/api/launcher.py &

echo "Starting Model API..."
python src/model/api.py &

# Wait for services to be ready
sleep 10

# 2. Ingest knowledge base
echo "Ingesting documents..."
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "water_guide_001",
    "doc_title": "Water System Guide",
    "content": "Water anomalies include pressure spikes, pH changes, and turbidity increase..."
  }'

# 3. Ask question in different modes
echo "Getting concise answer..."
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are water anomalies?", "response_mode": "concise"}'

echo "Getting detailed answer..."
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are water anomalies?", "response_mode": "verbose"}'

# 4. Predict on sensor data
echo "Checking sensor anomaly..."
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pressure": 150,
    "temperature": 25,
    "ph_level": 6.5,
    "dissolved_oxygen": 3.0,
    "turbidity": 1.5,
    "flow_rate": 200
  }'
```

---

## Troubleshooting

### RAG API Won't Start
```bash
# Check if port 8001 is in use
netstat -an | grep 8001

# Kill existing process on port 8001
lsof -ti:8001 | xargs kill -9  # macOS/Linux
Get-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess | Stop-Process  # Windows
```

### Model API Won't Start
```bash
# Check model files exist
ls -la src/model/weights/

# Retrain models if needed
python src/model/model.py
```

### RAGAS Warning
```
[WARNING] RAGAS not available: cannot import name 'answer_semantic_similarity'
```
This is a compatibility warning. The system continues to work without RAGAS evaluation.

### Ollama Connection Failed
```bash
# Ensure Ollama is running
ollama serve

# Check connectivity
curl http://localhost:11434/api/tags
```

---

## Performance Tips

1. **Adjust top_k** - Fewer results = faster queries
   ```bash
   "top_k": 3  # Instead of default 5
   ```

2. **Use concise mode** for end users (no extra processing)

3. **Batch predictions** for multiple sensor readings

4. **Enable caching** - Repeated questions return cached results

---

## Architecture

```
┌─────────────────────┐
│   Client/Frontend   │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │  RAG API    │  (Port 8001)
    │ LangGraph   │
    └──────┬──────┘
           │
    ┌──────▼──────────┐
    │ ChromaDB + FAISS│
    │ (Vector Search) │
    └──────┬──────────┘
           │
    ┌──────▼──────┐
    │   Ollama    │  (Port 11434)
    │  LLM Engine │
    └─────────────┘

    ┌─────────────────────┐
    │   Model API         │  (Port 8002)
    │  (KNN + LSTM)       │
    └────────┬────────────┘
             │
        ┌────▼─────┐
        │  SQLite   │
        │  Database │
        └───────────┘
```

---

## Next Steps

1. ✅ Ingest your water system documentation
2. ✅ Train your anomaly detection model
3. ✅ Test all three response modes
4. ✅ Collect user feedback for RL improvement
5. ✅ Deploy to production

For more help, check API documentation at `http://localhost:8001/docs`
