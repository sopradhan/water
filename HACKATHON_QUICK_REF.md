# 🚀 Hackathon Quick Reference

## ⚡ 30-Second Setup

```bash
# 1. Start Ollama (separate terminal)
ollama serve

# 2. Initialize database (one-time)
python check_db.py

# 3. Start RAG API (background)
python src/rag/api/launcher.py

# 4. Start Model API (separate terminal)
python src/model/api.py
```

That's it! Both APIs ready on `localhost:8001` and `localhost:8002`

---

## 📚 RAG API Examples

### Concise Mode (User Friendly)
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is normal water pressure?", "response_mode": "concise"}'
```

### Verbose Mode (Debug/Full Info)
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is normal water pressure?", "response_mode": "verbose"}'
```

### Internal Mode (System Integration)
```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is normal water pressure?", "response_mode": "internal"}'
```

---

## 🤖 Model API Examples

### Single Prediction
```bash
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pressure": 65,
    "temperature": 22,
    "ph_level": 7.2,
    "dissolved_oxygen": 8.5,
    "turbidity": 0.3,
    "flow_rate": 150
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8002/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"pressure": 65, "temperature": 22, ...},
      {"pressure": 120, "temperature": 28, ...}
    ]
  }'
```

---

## 📝 Ingest Documents

```bash
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "water_doc_001",
    "doc_title": "Water System Manual",
    "content": "Your documentation here..."
  }'
```

---

## 🔗 API Documentation

- **RAG Swagger UI**: http://localhost:8001/docs
- **Model Swagger UI**: http://localhost:8002/docs
- **Full Guides**:
  - `SETUP_GUIDE.md` - Complete setup and usage
  - `MODEL_API_GUIDE.md` - Model API detailed guide

---

## 🎯 Key Features Ready

### RAG System
- ✅ LangGraph orchestration
- ✅ ChromaDB vector search
- ✅ FAISS reranking
- ✅ 3 response modes (concise/verbose/internal)
- ✅ Intelligent guardrails with LLM responses
- ✅ RBAC filtering
- ✅ RL feedback collection
- ✅ PII detection and redaction

### Model System
- ✅ KNN classifier (instant)
- ✅ LSTM neural network (temporal)
- ✅ Ensemble predictions
- ✅ Batch processing
- ✅ Confidence scores
- ✅ Risk levels (low/medium/high/critical)

---

## 🔧 Configuration

**All paths in**: `src/config/paths_config.json`

**Default locations:**
- Database: `src/data/RAG/rag_metadata.db`
- Vector DB: `src/data/RAG/chroma_db`
- Models: `src/model/weights/`

**Override with env vars:**
```bash
export RAG_DB_PATH="custom/db/path.db"
export CHROMA_DB_PATH="custom/vectordb/path"
export LLM_BASE_URL="http://custom-llm:11434"
```

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 8001 in use | `Get-Process -Id (Get-NetTCPConnection -LocalPort 8001).OwningProcess \| Stop-Process` |
| Port 8002 in use | `Get-Process -Id (Get-NetTCPConnection -LocalPort 8002).OwningProcess \| Stop-Process` |
| No models found | `python src/model/model.py` to train |
| Ollama connection | Ensure `ollama serve` running on port 11434 |
| RAGAS warning | Normal - system works fine without it |

---

## 📊 Response Modes Comparison

| Mode | Use Case | Guardrails | Debug Info | Sources |
|------|----------|-----------|-----------|---------|
| **Concise** | End users | ✅ Yes | ❌ No | ❌ No |
| **Internal** | DB updates | ❌ No | ❌ No | ✅ Yes |
| **Verbose** | Engineers | ❌ No | ✅ Yes | ✅ Yes |

---

## 💡 Pro Tips

1. **Use concise mode** for your UI/frontend
2. **Use verbose mode** for debugging issues
3. **Use internal mode** for automatic database updates
4. **Batch predictions** 3+ samples for better performance
5. **Check `/health`** endpoints to verify services ready
6. **Use Swagger UI** at `/docs` for interactive testing

---

## 📞 Support

- Check `SETUP_GUIDE.md` for detailed setup
- Check `MODEL_API_GUIDE.md` for model details
- API docs available at `http://localhost:8001/docs` and `http://localhost:8002/docs`
- Check logs for errors: watch terminal output

---

## 🎉 You're Ready!

Everything is configured and ready for the hackathon. All hardcoded paths are now in config, documentation is complete, and both APIs are fully functional.

**Good luck! 🚀**
