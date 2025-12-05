"""Animated Graph Visualizer Module - Workflow tracking and visualization."""

from typing import Optional, Any, Dict
from datetime import datetime


class AnimatedGraphTracker:
    """Tracks workflow execution for animation and visualization."""
    
    def __init__(self, workflow_type: str, workflow_id: str):
        """
        Initialize workflow tracker.
        
        Args:
            workflow_type: Type of workflow (e.g., 'ingestion', 'retrieval')
            workflow_id: Unique workflow identifier
        """
        self.workflow_type = workflow_type
        self.workflow_id = workflow_id
        self.created_at = datetime.now().isoformat()
        self.events = []
        self.status = "running"
    
    def record_event(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Record a workflow event.
        
        Args:
            event_type: Type of event (e.g., 'node_start', 'node_end', 'edge_traverse')
            data: Optional event data
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data or {}
        }
        self.events.append(event)
    
    def node_start(self, node_id: str, node_label: str, context: Dict[str, Any] = None) -> None:
        """
        Record node start event.
        
        Args:
            node_id: Unique node identifier
            node_label: Display label for node
            context: Optional context data
        """
        self.record_event("node_start", {
            "node_id": node_id,
            "node_label": node_label,
            "context": context or {}
        })
    
    def node_end(self, node_id: str, result: Dict[str, Any] = None, status: str = "success") -> None:
        """
        Record node end event.
        
        Args:
            node_id: Unique node identifier
            result: Optional result data
            status: Node completion status (success/error/warning)
        """
        self.record_event("node_end", {
            "node_id": node_id,
            "result": result or {},
            "status": status
        })
    
    def edge_traversal(self, source_node: str, target_node: str, data_passed: list = None) -> None:
        """
        Record edge traversal between nodes.
        
        Args:
            source_node: Source node ID
            target_node: Target node ID
            data_passed: List of data items passed between nodes
        """
        self.record_event("edge_traversal", {
            "source": source_node,
            "target": target_node,
            "data_passed": data_passed or []
        })
    
    def get_graph_data(self) -> Dict[str, Any]:
        """
        Get graph data for visualization.
        
        Returns:
            Dictionary containing graph structure and events
        """
        nodes = {}
        edges = []
        
        for event in self.events:
            if event["type"] == "node_start":
                node_id = event["data"]["node_id"]
                nodes[node_id] = {
                    "id": node_id,
                    "label": event["data"]["node_label"],
                    "status": "started",
                    "context": event["data"]["context"]
                }
            elif event["type"] == "node_end":
                node_id = event["data"]["node_id"]
                if node_id in nodes:
                    nodes[node_id]["status"] = event["data"]["status"]
                    nodes[node_id]["result"] = event["data"]["result"]
            elif event["type"] == "edge_traversal":
                edges.append({
                    "source": event["data"]["source"],
                    "target": event["data"]["target"],
                    "data": event["data"]["data_passed"]
                })
        
        return {
            "workflow_type": self.workflow_type,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "nodes": nodes,
            "edges": edges,
            "events": self.events
        }
    
    def complete(self) -> None:
        """Mark workflow as complete."""
        self.status = "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker to dictionary."""
        return {
            "workflow_type": self.workflow_type,
            "workflow_id": self.workflow_id,
            "created_at": self.created_at,
            "status": self.status,
            "events": self.events
        }


def create_ingestion_tracker(tracker_id: str) -> AnimatedGraphTracker:
    """
    Create an ingestion workflow tracker.
    
    Args:
        tracker_id: Unique tracker identifier
    
    Returns:
        AnimatedGraphTracker configured for ingestion
    """
    return AnimatedGraphTracker(workflow_type="ingestion", workflow_id=tracker_id)


def create_retrieval_tracker(tracker_id: str) -> AnimatedGraphTracker:
    """
    Create a retrieval workflow tracker.
    
    Args:
        tracker_id: Unique tracker identifier
    
    Returns:
        AnimatedGraphTracker configured for retrieval
    """
    return AnimatedGraphTracker(workflow_type="retrieval", workflow_id=tracker_id)
