# Configuration Changes Guide - Hackathon Setup

## Overview
This document details all configuration files, path changes, and LLM settings needed for the hackathon setup with Azure.

---

## 1. LLM Configuration Changes

### File: `src/rag/config/llm_config.json`

**Status:** ✅ Already configured for Azure

#### Current Configuration:
```json
{
  "llm_providers": {
    "azure": {
      "type": "azure",
      "api_key": "sk-",
      "api_key_env": "AZURE_API_KEY",
      "api_endpoint": "https://genailab.tcs.in/v1",
      "model": "azure/genailab-maas-gpt-35-turbo",
      "temperature": 0.3,
      "max_tokens": 2000,
      "enabled": true              ← ENABLED FOR HACKATHON
    },
    "ollama": {
      "enabled": false             ← DISABLED FOR HACKATHON
    }
  },
  "default_provider": "azure",     ← USES AZURE BY DEFAULT
  
  "embedding_providers": {
    "azure_embedding": {
      "type": "azure",
      "api_key": "sk-",
      "api_key_env": "AZURE_API_KEY",
      "api_endpoint": "https://genailab.tcs.in/v1",
      "model": "azure/genailab-maas-text-embedding-3-large",
      "enabled": true              ← ENABLED FOR HACKATHON
    },
    "ollama": {
      "enabled": false             ← DISABLED FOR HACKATHON
    }
  },
  "default_embedding_provider": "azure_embedding"  ← USES AZURE BY DEFAULT
}
```

#### Required Changes:

| Field | Current Value | Required for Hackathon | Notes |
|-------|---------------|------------------------|-------|
| `azure.enabled` | `true` | ✅ KEEP TRUE | Azure LLM is primary provider |
| `azure.api_key_env` | `AZURE_API_KEY` | ✅ KEEP | Environment variable name |
| `azure.api_endpoint` | `https://genailab.tcs.in/v1` | ✅ KEEP | TCS GenAI Lab endpoint |
| `azure.model` | `azure/genailab-maas-gpt-35-turbo` | ✅ VERIFY | Confirm with Azure admin |
| `default_provider` | `azure` | ✅ KEEP | Use Azure for LLM |
| `azure_embedding.enabled` | `true` | ✅ KEEP TRUE | Azure Embedding is primary |
| `azure_embedding.api_endpoint` | `https://genailab.tcs.in/v1` | ✅ KEEP | Same as LLM endpoint |
| `azure_embedding.model` | `azure/genailab-maas-text-embedding-3-large` | ✅ VERIFY | Confirm with Azure admin |
| `default_embedding_provider` | `azure_embedding` | ✅ KEEP | Use Azure for embeddings |
| `ollama.enabled` | `false` | ✅ KEEP FALSE | Fallback only if needed |

---

## 2. Path Configuration Changes

### File: `src/config/paths_config.json`

**Location:** `c:\Users\PRADHAN\OneDrive\Desktop\water\src\config\paths_config.json`

#### Current Configuration:
```json
{
  "database": {
    "rag_metadata_db": "src/data/RAG/rag_metadata.db",
    "collection_name": "water_documents"
  },
  "vector_db": {
    "chroma_db_path": "src/data/RAG/chroma_db",
    "chroma_collection_name": "water_documents"
  },
  "models": {
    "knn_model_file": "src/model/weights/knn_model.pkl",
    "lstm_model_file": "src/model/weights/lstm_model.h5",
    "scaler_file": "src/model/weights/scaler.pkl",
    "label_encoders_file": "src/model/weights/label_encoders.pkl",
    "target_encoder_file": "src/model/weights/target_encoder.pkl"
  },
  "api": {
    "rag_host": "0.0.0.0",
    "rag_port": 8001,
    "model_host": "0.0.0.0",
    "model_port": 8002
  },
  "llm": {
    "provider": "azure",
    "base_url": "https://genailab.tcs.in/v1",
    "model": "azure/genailab-maas-gpt-35-turbo"
  },
  "environment_overrides": {
    "RAG_DB_PATH": "src/data/RAG/rag_metadata.db",
    "CHROMA_DB_PATH": "src/data/RAG/chroma_db",
    "LLM_BASE_URL": "https://genailab.tcs.in/v1"
  }
}
```

#### Path Change Matrix:

| Section | Field | Default Path | Hackathon Path | Status | Notes |
|---------|-------|--------------|-----------------|--------|-------|
| database | rag_metadata_db | `src/data/RAG/rag_metadata.db` | Same | ✅ OK | SQLite database for metadata |
| vector_db | chroma_db_path | `src/data/RAG/chroma_db` | Same | ✅ OK | Chroma vector database |
| models | knn_model_file | `src/model/weights/knn_model.pkl` | Same | ✅ OK | KNN model pickle |
| models | lstm_model_file | `src/model/weights/lstm_model.h5` | Same | ✅ OK | LSTM Keras model |
| models | scaler_file | `src/model/weights/scaler.pkl` | Same | ✅ OK | Data scaler for features |
| models | label_encoders_file | `src/model/weights/label_encoders.pkl` | Same | ✅ OK | Categorical encoders |
| models | target_encoder_file | `src/model/weights/target_encoder.pkl` | Same | ✅ OK | Target variable encoder |
| api | rag_port | `8001` | Same | ✅ OK | RAG API port |
| api | model_port | `8002` | Same | ✅ OK | Model API port |

#### Environment Variable Overrides:

```bash
# Optional: Override paths via environment variables
$env:RAG_DB_PATH = "src/data/RAG/rag_metadata.db"
$env:CHROMA_DB_PATH = "src/data/RAG/chroma_db"
$env:LLM_BASE_URL = "https://genailab.tcs.in/v1"
$env:AZURE_API_KEY = "your-azure-key"
$env:LLM_CONFIG_PATH = "src/rag/config/llm_config.json"
```

---

## 3. RAG Configuration Changes

### File: `src/rag/config/env_config.py`

**Location:** `c:\Users\PRADHAN\OneDrive\Desktop\water\src\rag\config\env_config.py`

**Status:** ✅ Reads from llm_config.json automatically

**No changes needed** - This file loads configuration from `llm_config.json`

---

## 4. Database Configuration

### File: `src/rag/config/llm_config.json` (database section)

```json
{
  "database": {
    "db_path": "src/data/RAG/rag_metadata.db",
    "chroma_persist_dir": "./data/chroma_index"
  }
}
```

#### Status: ✅ Paths are correct

| Setting | Value | Purpose |
|---------|-------|---------|
| db_path | `src/data/RAG/rag_metadata.db` | SQLite database for RAG metadata |
| chroma_persist_dir | `./data/chroma_index` | ChromaDB persistent storage |

---

## 5. Data Configuration

### File: `src/rag/config/data_sources.json`

**Location:** `c:\Users\PRADHAN\OneDrive\Desktop\water\src\rag\config\data_sources.json`

**Status:** ✅ Auto-configured

This file is auto-generated and contains ingested document metadata. No manual changes needed.

---

## 6. Prompt Configuration

### File: `src/rag/config/prompts.json`

**Location:** `c:\Users\PRADHAN\OneDrive\Desktop\water\src\rag\config\prompts.json`

**Status:** ✅ Ready to use

Contains system and user prompts for the RAG agent. No changes needed for hackathon.

---

## 7. Model Configuration

### File: `src/model/api.py`

**Status:** ✅ Uses paths from paths_config.json

The model API automatically reads paths from `src/config/paths_config.json`:

```python
# From paths_config.json
KNN_MODEL_FILE = config["models"]["knn_model_file"]
LSTM_MODEL_FILE = config["models"]["lstm_model_file"]
SCALER_FILE = config["models"]["scaler_file"]
LABEL_ENCODERS_FILE = config["models"]["label_encoders_file"]
TARGET_ENCODER_FILE = config["models"]["target_encoder_file"]
```

**No changes needed.**

---

## 8. API Configuration

### File: `src/rag/api/startup.py`

**Status:** ✅ Uses paths from paths_config.json

```python
# Reads from paths_config.json
RAG_API_HOST = config["api"]["rag_host"]      # 0.0.0.0
RAG_API_PORT = config["api"]["rag_port"]      # 8001
```

**No changes needed.**

---

## 9. Azure Credentials Setup

### Required Environment Variables:

```bash
# CRITICAL FOR HACKATHON - SET BEFORE RUNNING
$env:AZURE_API_KEY = "your-tcs-genai-lab-api-key"

# Optional but recommended
$env:AZURE_ENDPOINT = "https://genailab.tcs.in/v1"
$env:LLM_CONFIG_PATH = "src/rag/config/llm_config.json"
```

### How Credentials are Read:

1. **First Priority:** Direct in config: `"api_key": "sk-..."`
2. **Second Priority:** Environment variable: `os.getenv('AZURE_API_KEY')`
3. **Fallback:** Raises error if not found

**Current setting in llm_config.json:**
```json
"api_key": "sk-",                        // Placeholder
"api_key_env": "AZURE_API_KEY"           // Will read from env var
```

---

## 10. Complete Setup Checklist

### Before Hackathon Launch:

```bash
# 1. Set Azure credentials
$env:AZURE_API_KEY = "your-key"

# 2. Verify paths exist
Test-Path "src/data/RAG"
Test-Path "src/model/weights"
Test-Path "src/rag/config"

# 3. Verify config files exist
Test-Path "src/config/paths_config.json"
Test-Path "src/rag/config/llm_config.json"
Test-Path "src/rag/config/env_config.py"

# 4. Test imports
python -c "from src.rag.tools.services.llm_service import LLMService; print('LLM Service OK')"
python -c "from src.rag.agents.langgraph_agent.langgraph_rag_agent import LangGraphRAGAgent; print('RAG Agent OK')"

# 5. Start RAG API
python src/rag/api/startup.py

# 6. Start Model API (in new terminal)
python -m src.model.api

# 7. Test with client
python src/client.py
```

---

## 11. File Structure Reference

```
water/
├── src/
│   ├── config/
│   │   └── paths_config.json              ← Path configuration
│   │
│   ├── rag/
│   │   ├── config/
│   │   │   ├── llm_config.json            ← LLM & Embedding config
│   │   │   ├── env_config.py              ← Environment loader
│   │   │   ├── data_sources.json          ← Auto-generated
│   │   │   └── prompts.json               ← System prompts
│   │   │
│   │   ├── tools/
│   │   │   └── services/
│   │   │       └── llm_service.py         ← Uses llm_config.json
│   │   │
│   │   ├── agents/
│   │   │   └── langgraph_agent/
│   │   │       ├── langgraph_rag_agent.py ← Loads llm_config
│   │   │       └── api.py                 ← FastAPI server
│   │   │
│   │   └── api/
│   │       ├── startup.py                 ← Starts RAG API
│   │       └── launcher.py                ← Alternative launcher
│   │
│   ├── model/
│   │   ├── weights/
│   │   │   ├── knn_model.pkl              ← ML model
│   │   │   ├── lstm_model.h5              ← Deep learning model
│   │   │   ├── scaler.pkl                 ← Feature scaler
│   │   │   ├── label_encoders.pkl         ← Category encoders
│   │   │   └── target_encoder.pkl         ← Target encoder
│   │   │
│   │   ├── api.py                         ← Model serving API
│   │   └── dashboard.py                   ← Streamlit dashboard
│   │
│   └── data/
│       ├── RAG/
│       │   ├── rag_metadata.db            ← SQLite database
│       │   └── chroma_db/                 ← Vector database
│       │
│       └── training_dataset/              ← Training data
│
└── AZURE_HACKATHON_SETUP.md              ← Azure setup guide
└── CONFIGURATION_CHANGES_GUIDE.md        ← This file
```

---

## 12. Configuration Dependency Map

```
┌─────────────────────────────────────────┐
│     AZURE_API_KEY (Environment)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   src/rag/config/llm_config.json       │
│  - Azure LLM settings                   │
│  - Azure Embedding settings             │
│  - Default provider: azure              │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        ▼              ▼
┌────────────────┐  ┌──────────────────┐
│  LLMService    │  │  LangGraphAgent  │
│  (llm_service) │  │  (rag_agent)     │
└────────────────┘  └──────────────────┘
        │              │
        ▼              ▼
┌─────────────────────────────────────────┐
│   RAG API (src/rag/api/startup.py)     │
└────────────────┬────────────────────────┘
                 │
                 ▼
          ┌──────────────┐
          │ Port: 8001   │
          └──────────────┘
```

---

## 13. LLM Model Options for Azure

### Available Models in Azure GenAI Lab:

| Model | Type | Purpose | Config Name |
|-------|------|---------|-------------|
| `genailab-maas-gpt-35-turbo` | LLM | General purpose | `azure.model` |
| `genailab-maas-gpt-4o` | LLM | Advanced reasoning | Alternative |
| `genailab-maas-text-embedding-3-large` | Embedding | Vector generation | `azure_embedding.model` |
| `genailab-maas-text-embedding-3-small` | Embedding | Lightweight embedding | Alternative |

**To switch models, edit `src/rag/config/llm_config.json`:**

```json
{
  "llm_providers": {
    "azure": {
      "model": "azure/genailab-maas-gpt-4o"  // Change here
    }
  },
  "embedding_providers": {
    "azure_embedding": {
      "model": "azure/genailab-maas-text-embedding-3-small"  // Change here
    }
  }
}
```

---

## 14. Path Change Summary Table

| Component | Config File | Path Setting | Default Value | Change Required |
|-----------|------------|--------------|---------------|-----------------|
| RAG Database | `llm_config.json` | `database.db_path` | `src/data/RAG/rag_metadata.db` | ❌ None |
| Vector DB | `llm_config.json` | `database.chroma_persist_dir` | `./data/chroma_index` | ❌ None |
| KNN Model | `paths_config.json` | `models.knn_model_file` | `src/model/weights/knn_model.pkl` | ❌ None |
| LSTM Model | `paths_config.json` | `models.lstm_model_file` | `src/model/weights/lstm_model.h5` | ❌ None |
| Scaler | `paths_config.json` | `models.scaler_file` | `src/model/weights/scaler.pkl` | ❌ None |
| Label Encoders | `paths_config.json` | `models.label_encoders_file` | `src/model/weights/label_encoders.pkl` | ❌ None |
| Target Encoder | `paths_config.json` | `models.target_encoder_file` | `src/model/weights/target_encoder.pkl` | ❌ None |
| RAG API Port | `paths_config.json` | `api.rag_port` | `8001` | ❌ None |
| Model API Port | `paths_config.json` | `api.model_port` | `8002` | ❌ None |

---

## 15. LLM Config Change Summary Table

| Component | Config File | Setting | Default | Hackathon | Action |
|-----------|------------|---------|---------|-----------|--------|
| LLM Provider | `llm_config.json` | `default_provider` | `azure` | `azure` | ✅ Keep |
| LLM Type | `llm_config.json` | `azure.type` | `azure` | `azure` | ✅ Keep |
| LLM Enabled | `llm_config.json` | `azure.enabled` | `true` | `true` | ✅ Keep |
| LLM API Key Env | `llm_config.json` | `azure.api_key_env` | `AZURE_API_KEY` | `AZURE_API_KEY` | ✅ Keep |
| LLM Endpoint | `llm_config.json` | `azure.api_endpoint` | `https://genailab.tcs.in/v1` | Same | ✅ Keep |
| LLM Model | `llm_config.json` | `azure.model` | `azure/genailab-maas-gpt-35-turbo` | Verify | ⚠️ Confirm |
| Embedding Provider | `llm_config.json` | `default_embedding_provider` | `azure_embedding` | `azure_embedding` | ✅ Keep |
| Embedding Type | `llm_config.json` | `azure_embedding.type` | `azure` | `azure` | ✅ Keep |
| Embedding Enabled | `llm_config.json` | `azure_embedding.enabled` | `true` | `true` | ✅ Keep |
| Embedding Model | `llm_config.json` | `azure_embedding.model` | `azure/genailab-maas-text-embedding-3-large` | Verify | ⚠️ Confirm |

---

## 16. Quick Reference: What to Change

### ❌ DO NOT CHANGE:
- Database paths in `llm_config.json`
- API ports in `paths_config.json`
- Model file paths in `paths_config.json`
- Default provider settings in `llm_config.json`

### ✅ MUST CHANGE:
- Set `AZURE_API_KEY` environment variable before running

### ⚠️ VERIFY:
- Azure endpoint URL: `https://genailab.tcs.in/v1`
- Azure LLM model name: `azure/genailab-maas-gpt-35-turbo`
- Azure Embedding model name: `azure/genailab-maas-text-embedding-3-large`

---

## 17. Troubleshooting: Path & Config Issues

### Issue: "Config file not found"
```bash
# Check paths exist
Test-Path "src/config/paths_config.json"
Test-Path "src/rag/config/llm_config.json"

# If missing, system creates defaults
```

### Issue: "Database path error"
```bash
# Verify directory exists
Test-Path "src/data/RAG"

# Create if missing
New-Item -ItemType Directory -Path "src/data/RAG" -Force
```

### Issue: "Model files not found"
```bash
# Check model weights directory
Test-Path "src/model/weights"

# List contents
Get-ChildItem "src/model/weights"
```

### Issue: "Azure API key not found"
```bash
# Set environment variable
$env:AZURE_API_KEY = "your-key"

# Verify it's set
Write-Host $env:AZURE_API_KEY
```

---

## Summary

**Key Takeaways:**
1. ✅ **Paths:** All default paths are correct for hackathon
2. ✅ **LLM Config:** Already configured for Azure
3. ✅ **Embedding:** Already configured for Azure
4. ⚠️ **Action Required:** Set `AZURE_API_KEY` environment variable
5. ✅ **Verification:** Run health check via `/health` endpoint

**For Hackathon:**
```bash
# 1. Set Azure key
$env:AZURE_API_KEY = "your-key"

# 2. Start RAG API
python src/rag/api/startup.py

# 3. Test
curl http://localhost:8001/health
```

---

**Last Updated:** December 6, 2025  
**Status:** ✅ Ready for Hackathon with Azure
