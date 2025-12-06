# Azure Setup Guide for Hackathon

## Location of Configuration Files

### 1. **LLM Configuration** 
**Path:** `src/rag/config/llm_config.json`

### 2. **Paths Configuration**
**Path:** `src/config/paths_config.json`

---

## Azure LLM & Embedding Setup

### Configuration in `llm_config.json`

The system is now configured to use Azure for both LLM and embeddings:

```json
{
  "llm_providers": {
    "azure": {
      "type": "azure",
      "api_key": "sk-",                    // Replace or use env var
      "api_key_env": "AZURE_API_KEY",     // Environment variable to read
      "api_endpoint": "https://genailab.tcs.in/v1",
      "model": "azure/genailab-maas-gpt-35-turbo",
      "temperature": 0.3,
      "max_tokens": 2000,
      "enabled": true                     // ✅ ENABLED
    }
  },
  "default_provider": "azure",            // ✅ Set as default
  
  "embedding_providers": {
    "azure_embedding": {
      "type": "azure",
      "api_key": "sk-",
      "api_key_env": "AZURE_API_KEY",
      "api_endpoint": "https://genailab.tcs.in/v1",
      "model": "azure/genailab-maas-text-embedding-3-large",
      "enabled": true                     // ✅ ENABLED
    }
  },
  "default_embedding_provider": "azure_embedding"  // ✅ Set as default
}
```

---

## Setting Up Azure Credentials

### Option 1: Environment Variable (Recommended)

```bash
# Set the environment variable before running the application
$env:AZURE_API_KEY = "your-azure-api-key-here"

# Verify it's set
Write-Host $env:AZURE_API_KEY
```

### Option 2: Direct Configuration

Edit `src/rag/config/llm_config.json` and replace `"api_key": "sk-"` with your actual key:

```json
"api_key": "sk-your-actual-azure-key",
```

---

## Testing Azure Configuration

### Test LLM Connection

```bash
# Navigate to project root
cd c:\Users\PRADHAN\OneDrive\Desktop\water

# Run test
python -c "
from src.rag.tools.services.llm_service import LLMService
import json

# Load config
with open('src/rag/config/llm_config.json') as f:
    config = json.load(f)

# Create service
llm = LLMService(config)

# Test
print('LLM Service initialized successfully!')
print(f'Provider: {llm.provider}')
print(f'Model: {llm.model}')
"
```

### Test Embedding Connection

```bash
python -c "
from src.rag.tools.services.llm_service import LLMService
import json

with open('src/rag/config/llm_config.json') as f:
    config = json.load(f)

llm = LLMService(config)
embedder = llm.get_embeddings()

print('Embedding Service initialized successfully!')
print(f'Embedder: {type(embedder).__name__}')

# Test embedding
test_text = 'Test water anomaly detection'
embedding = embedder.embed_query(test_text)
print(f'Embedding dimension: {len(embedding)}')
print('Sample values:', embedding[:5])
"
```

---

## How LLMService Uses Azure Configuration

### File: `src/rag/tools/services/llm_service.py`

The service automatically handles Azure setup:

```python
class LLMService:
    def _create_azure(self, config: dict) -> BaseChatModel:
        """Create Azure OpenAI-compatible chat model"""
        api_key = config.get('api_key') or os.getenv(config.get('api_key_env', 'AZURE_API_KEY'))
        
        if not api_key:
            raise ValueError("Azure API key not found")
        
        from langchain_community.chat_models import ChatOpenAI
        
        return ChatOpenAI(
            openai_api_base=config.get('api_endpoint'),
            openai_api_key=api_key,
            model=config.get('model', 'azure/genailab-maas-gpt-4o'),
            temperature=config.get('temperature', 0.3),
            max_tokens=config.get('max_tokens', 2000),
            request_timeout=300
        )
    
    def _create_azure_embedding(self, config: dict):
        """Create Azure embedding model"""
        api_key = config.get('api_key') or os.getenv(config.get('api_key_env', 'AZURE_API_KEY'))
        
        if not api_key:
            raise ValueError("Azure API key not found")
        
        from langchain_community.embeddings import OpenAIEmbeddings
        
        return OpenAIEmbeddings(
            openai_api_base=config.get('api_endpoint'),
            openai_api_key=api_key,
            model=config.get('model', 'azure/genailab-maas-text-embedding-3-large'),
            request_timeout=300
        )
```

---

## Starting RAG API with Azure

### Quick Start

```bash
# Method 1: Run the startup script
python src/rag/api/startup.py

# Method 2: Use the launcher
python src/rag/api/launcher.py

# Method 3: Run as module
python -m src.rag.api.startup
```

### Verify API is Running

```bash
# Check health endpoint
curl http://localhost:8001/health

# Should return:
# {"status": "healthy", "services": {"llm": "ok", "embeddings": "ok"}}
```

---

## Configuration Files Reference

### 1. `llm_config.json` (LLM & Embedding Settings)
**Location:** `src/rag/config/llm_config.json`
- LLM provider configuration (Azure, OpenAI, Ollama, etc.)
- Embedding provider configuration
- Default provider selection
- Database paths

### 2. `paths_config.json` (System Paths)
**Location:** `src/config/paths_config.json`
- Database paths
- Model paths
- API ports
- Vector DB paths

### 3. `env_config.py` (Environment Variables)
**Location:** `src/rag/config/env_config.py`
- Environment variable loading
- Default configurations
- Path resolution

---

## Common Issues & Solutions

### Issue: "Azure API key not found"

**Solution:** Set environment variable
```bash
$env:AZURE_API_KEY = "your-key"
python src/rag/api/startup.py
```

### Issue: Connection timeout to Azure

**Solution:** Check endpoint
```bash
# Verify Azure endpoint in llm_config.json
# Should be: https://genailab.tcs.in/v1
```

### Issue: Model not found

**Solution:** Verify model name
```bash
# Check llm_config.json for correct model name
# Example: "azure/genailab-maas-gpt-35-turbo"
```

---

## Azure Integration in RAG Agent

### The agent automatically uses Azure configuration:

```python
# src/rag/agents/langgraph_agent/langgraph_rag_agent.py

# Load LLM config
with open(llm_config_path, "r") as f:
    llm_config = json.load(f)

# Create LLM service with Azure
llm_service = LLMService(llm_config)

# Uses Azure by default
llm = llm_service.get_llm()           # Uses Azure LLM
embeddings = llm_service.get_embeddings()  # Uses Azure Embedding
```

---

## Switching Between Providers

### To switch from Azure back to Ollama:

Edit `src/rag/config/llm_config.json`:

```json
{
  "default_provider": "ollama",              // Change from "azure"
  "default_embedding_provider": "ollama",     // Change from "azure_embedding"
  
  "llm_providers": {
    "ollama": {
      "enabled": true                        // Set to true
    },
    "azure": {
      "enabled": false                       // Set to false
    }
  }
}
```

---

## Testing the Complete Pipeline

```bash
# 1. Start RAG API
python src/rag/api/startup.py &

# 2. In another terminal, test with Python client
python -c "
from src.client import WaterAnomalyClient

client = WaterAnomalyClient()

# Test with Azure
answer = client.ask('What is water leakage?', mode='concise')
print('Q: What is water leakage?')
print(f'A: {answer.answer}')
"
```

---

## Environment Variables for Hackathon

```bash
# Set these before running the application

# Azure API Key
$env:AZURE_API_KEY = "your-azure-key"

# Optional: Override config paths
$env:LLM_CONFIG_PATH = "src/rag/config/llm_config.json"
$env:RAG_DB_PATH = "src/data/RAG/rag_metadata.db"

# Optional: RAG API URL (if running on different machine)
$env:RAG_API_URL = "http://localhost:8001"
```

---

## Quick Checklist for Hackathon

- [ ] Set `AZURE_API_KEY` environment variable
- [ ] Verify `llm_config.json` has Azure enabled
- [ ] Test LLM connection: `python -c "from src.rag.tools.services.llm_service import LLMService; print('OK')"`
- [ ] Test Embedding connection: `python -c "from src.rag.tools.services.llm_service import LLMService; print('OK')"`
- [ ] Start RAG API: `python src/rag/api/startup.py`
- [ ] Test API health: `curl http://localhost:8001/health`
- [ ] Test with client: `python src/client.py`
- [ ] Ready for hackathon! 🚀

---

## Support

For issues, check:
1. `src/rag/tools/services/llm_service.py` - LLM service implementation
2. `src/rag/config/llm_config.json` - Configuration values
3. `src/rag/agents/langgraph_agent/api.py` - RAG API implementation
