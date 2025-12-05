"""
Configuration Management Tool

This tool allows the Orchestrator or Healing Agent to dynamically adjust 
RAG and LLM parameters, closing the self-optimization loop.
"""
import json
from typing import Dict, Any
from langchain_core.tools import tool

# NOTE: This implementation is conceptual. In a real system, 'config_service'
# would manage a persistent configuration store (e.g., Redis, Consul, YAML file).

@tool
def adjust_config_tool(config_service: Any, updates: Dict[str, Any]) -> str:
    """
    Dynamically adjusts system configuration parameters based on optimization signals.
    This includes RAG parameters (chunk_size, k_final) and LLM parameters (model_name, temperature).

    The Healing Agent (via optimize_chunk_size_tool) suggests updates, and the 
    Orchestrator Agent uses this tool to apply them.

    Args:
        config_service: Service object providing .get_config() and .update_config(updates) methods.
        updates (Dict[str, Any]): Dictionary of configuration keys and new values to set.
                                  Example: {"RAG_K_FINAL": 4, "LLM_MODEL_NAME": "gemini-2.5-flash-preview-09-2025"}

    Returns:
        JSON string with 'success' status and the 'applied_updates'.
    """
    try:
        # Load current config to validate keys (omitted for brevity)
        current_config = config_service.get_config()
        
        # Apply updates
        config_service.update_config(updates)
        
        # Log the change for audit/RL tracking
        # logging_service.log_decision("Config_Update", updates)

        return json.dumps({
            "success": True,
            "applied_updates": updates,
            "message": "System configuration successfully updated."
        })

    except Exception as e:
        return json.dumps({"success": False, "error": f"Failed to adjust configuration: {str(e)}"})
