"""LangGraph Visualizer Module - Graph visualization utilities."""

import json
from pathlib import Path
from typing import Optional, Any, Dict


class GraphVisualization:
    """Represents a graph visualization."""
    
    def __init__(self, session_id: str):
        """Initialize graph visualization."""
        self.session_id = session_id
        self.data = {}
        self.nodes = {}
        self.edges = []
    
    def record_node_start(self, node_id: str, node_data: Dict[str, Any] = None) -> None:
        """
        Record node start event.
        
        Args:
            node_id: Node identifier
            node_data: Optional node data
        """
        self.nodes[node_id] = {
            "id": node_id,
            "status": "started",
            "data": node_data or {}
        }
    
    def record_node_end(self, node_id: str, result: Dict[str, Any] = None, status: str = "success") -> None:
        """
        Record node end event.
        
        Args:
            node_id: Node identifier
            result: Optional result data
            status: Node status (success/error/warning)
        """
        if node_id in self.nodes:
            self.nodes[node_id]["status"] = status
            self.nodes[node_id]["result"] = result or {}
    
    def record_edge_traversal(self, source: str, target: str, data: Any = None) -> None:
        """
        Record edge traversal.
        
        Args:
            source: Source node ID
            target: Target node ID
            data: Optional data passed through edge
        """
        self.edges.append({
            "source": source,
            "target": target,
            "data": data
        })
    
    def record_error(self, node_id: str, error_msg: str) -> None:
        """
        Record error event.
        
        Args:
            node_id: Node where error occurred
            error_msg: Error message
        """
        if node_id in self.nodes:
            self.nodes[node_id]["status"] = "error"
            self.nodes[node_id]["error"] = error_msg
    
    def get_trace_data(self) -> Dict[str, Any]:
        """
        Get complete trace data for visualization.
        
        Returns:
            Dictionary containing full trace information
        """
        return {
            "session_id": self.session_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "data": self.data
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "data": self.data
        }


def create_visualization(session_id: str) -> GraphVisualization:
    """
    Create a graph visualization for a session.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        GraphVisualization object
    """
    return GraphVisualization(session_id)


def save_visualization(viz: GraphVisualization, output_dir: str = "logs", graph: Optional[Any] = None) -> Dict[str, str]:
    """
    Save visualization to disk with proper JSON serialization.
    
    Args:
        viz: GraphVisualization object
        output_dir: Directory to save visualizations
        graph: Optional LangGraph graph object
    
    Returns:
        Dictionary with file paths of saved visualizations
    """
    import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get viz data and ensure all values are JSON serializable
    viz_dict = viz.to_dict()
    
    # Recursively clean non-serializable objects with strict limits
    def make_serializable(obj, depth=0, max_depth=5):
        """Convert non-JSON-serializable objects to strings. Max depth 5 to prevent recursion issues."""
        # Hard limit on depth
        if depth > max_depth:
            return "[Max depth exceeded]"
        
        try:
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    try:
                        result[str(k)] = make_serializable(v, depth+1, max_depth)
                    except (RecursionError, TypeError, AttributeError):
                        result[str(k)] = "[Non-serializable value]"
                return result
            elif isinstance(obj, (list, tuple)):
                try:
                    return [make_serializable(item, depth+1, max_depth) for item in obj]
                except (RecursionError, TypeError, AttributeError):
                    return "[Non-serializable list]"
            elif isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # For everything else, convert to string
                return str(obj)
        except RecursionError:
            return "[Recursion error]"
        except Exception:
            return "[Serialization failed]"
    
    viz_dict = make_serializable(viz_dict)
    
    # Save visualization data as JSON
    viz_file = output_path / f"visualization_{viz.session_id}.json"
    with open(viz_file, 'w') as f:
        json.dump(viz_dict, f, indent=2, default=str)
    
    return {
        "visualization": str(viz_file),
        "status": "saved"
    }
