"""
Water Anomaly Detection Client Library
======================================

Easy-to-use Python client for both RAG and Model APIs.

Usage:
    from src.client import WaterAnomalyClient
    
    client = WaterAnomalyClient()
    
    # Ask question
    answer = client.ask("What is normal water pressure?", mode="concise")
    
    # Predict anomaly
    prediction = client.predict_anomaly(
        pressure=65, temperature=22, ph_level=7.2, ...
    )
"""

import requests
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ResponseMode(Enum):
    """RAG response modes"""
    CONCISE = "concise"      # User-friendly, clean answers
    VERBOSE = "verbose"      # Full debug information
    INTERNAL = "internal"    # System integration


@dataclass
class PredictionResult:
    """Model prediction result"""
    ensemble_prediction: str
    ensemble_confidence: float
    anomaly_detected: bool
    risk_level: str
    knn_prediction: str
    lstm_prediction: str
    execution_time_ms: float


@dataclass
class RagAnswer:
    """RAG answer result"""
    answer: str
    session_id: str
    sources_count: int
    guardrails_applied: bool
    execution_time_ms: float


class WaterAnomalyClient:
    """Client for Water Anomaly Detection System"""
    
    def __init__(
        self,
        rag_url: str = "http://localhost:8001",
        model_url: str = "http://localhost:8002"
    ):
        """
        Initialize client.
        
        Args:
            rag_url: Base URL for RAG API
            model_url: Base URL for Model API
        """
        self.rag_url = rag_url
        self.model_url = model_url
        self.session = requests.Session()
        
        logger.info(f"WaterAnomalyClient initialized")
        logger.info(f"  RAG API: {rag_url}")
        logger.info(f"  Model API: {model_url}")
    
    def check_health(self) -> Dict[str, Any]:
        """Check if both APIs are healthy"""
        try:
            rag_health = self.session.get(f"{self.rag_url}/health").json()
            model_health = self.session.get(f"{self.model_url}/health").json()
            
            return {
                "rag": rag_health.get("status") == "operational",
                "model": model_health.get("status") == "operational",
                "details": {"rag": rag_health, "model": model_health}
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"rag": False, "model": False, "error": str(e)}
    
    # ========================================================================
    # RAG API Methods
    # ========================================================================
    
    def ask(
        self,
        question: str,
        mode: str = "concise",
        company_id: int = 1,
        dept_id: int = 101,
        top_k: int = 5
    ) -> RagAnswer:
        """
        Ask a question using RAG system.
        
        Args:
            question: The question to ask
            mode: Response mode - "concise", "verbose", or "internal"
            company_id: Company ID for RBAC filtering
            dept_id: Department ID for RBAC filtering
            top_k: Number of context chunks to use
        
        Returns:
            RagAnswer with answer and metadata
        
        Example:
            >>> client = WaterAnomalyClient()
            >>> answer = client.ask("What is normal water pressure?")
            >>> print(answer.answer)
            "Normal water pressure is 40-80 PSI"
        """
        try:
            response = self.session.post(
                f"{self.rag_url}/ask",
                json={
                    "question": question,
                    "response_mode": mode,
                    "company_id": company_id,
                    "dept_id": dept_id,
                    "top_k": top_k
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return RagAnswer(
                answer=data.get("answer", ""),
                session_id=data.get("session_id", ""),
                sources_count=data.get("context_chunks", 0),
                guardrails_applied=data.get("guardrails_applied", False),
                execution_time_ms=0  # Not in response
            )
        except Exception as e:
            logger.error(f"Ask failed: {e}")
            raise
    
    def ask_verbose(
        self,
        question: str,
        company_id: int = 1,
        dept_id: int = 101
    ) -> Dict[str, Any]:
        """
        Ask question in verbose mode (full debug info).
        
        Args:
            question: The question
            company_id: Company ID
            dept_id: Department ID
        
        Returns:
            Full response with all metadata
        """
        response = self.session.post(
            f"{self.rag_url}/ask",
            json={
                "question": question,
                "response_mode": "verbose",
                "company_id": company_id,
                "dept_id": dept_id
            }
        )
        response.raise_for_status()
        return response.json()
    
    def ask_internal(
        self,
        question: str,
        company_id: int = 1,
        dept_id: int = 101
    ) -> Dict[str, Any]:
        """
        Ask question in internal mode (system integration).
        
        Args:
            question: The question
            company_id: Company ID
            dept_id: Department ID
        
        Returns:
            Structured response for database storage
        """
        response = self.session.post(
            f"{self.rag_url}/ask",
            json={
                "question": question,
                "response_mode": "internal",
                "company_id": company_id,
                "dept_id": dept_id
            }
        )
        response.raise_for_status()
        return response.json()
    
    def ingest_document(
        self,
        doc_id: str,
        content: str,
        doc_title: str = "",
        doc_type: str = "text",
        company_id: int = 1,
        dept_id: int = 101
    ) -> Dict[str, Any]:
        """
        Ingest a document into the RAG system.
        
        Args:
            doc_id: Unique document ID
            content: Document content
            doc_title: Document title
            doc_type: Document type (text, pdf, etc)
            company_id: Company ID
            dept_id: Department ID
        
        Returns:
            Ingestion result
        """
        response = self.session.post(
            f"{self.rag_url}/ingest",
            json={
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_type": doc_type,
                "content": content,
                "company_id": company_id,
                "dept_id": dept_id
            }
        )
        response.raise_for_status()
        return response.json()
    
    def submit_feedback(
        self,
        session_id: str,
        rating: int,
        feedback_text: str = ""
    ) -> Dict[str, Any]:
        """
        Submit user feedback for RL training.
        
        Args:
            session_id: Session ID from ask response
            rating: Rating 1-5
            feedback_text: Optional feedback text
        
        Returns:
            Feedback submission result
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        response = self.session.post(
            f"{self.rag_url}/feedback",
            json={
                "session_id": session_id,
                "rating": rating,
                "feedback_text": feedback_text
            }
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================================================
    # Model API Methods
    # ========================================================================
    
    def predict_anomaly(
        self,
        pressure: float,
        temperature: float,
        ph_level: float,
        dissolved_oxygen: float,
        turbidity: float,
        flow_rate: float,
        location: Optional[str] = None,
        sensor_type: Optional[str] = None
    ) -> PredictionResult:
        """
        Predict water anomaly for single sensor reading.
        
        Args:
            pressure: Pressure in PSI
            temperature: Temperature in Celsius
            ph_level: pH level (0-14)
            dissolved_oxygen: Dissolved oxygen in mg/L
            turbidity: Turbidity in NTU
            flow_rate: Flow rate in L/min
            location: Sensor location
            sensor_type: Sensor type
        
        Returns:
            PredictionResult with ensemble prediction
        
        Example:
            >>> prediction = client.predict_anomaly(
            ...     pressure=65, temperature=22, ph_level=7.2,
            ...     dissolved_oxygen=8.5, turbidity=0.3, flow_rate=150
            ... )
            >>> if prediction.anomaly_detected:
            ...     print(f"[ALERT] Anomaly detected! Risk: {prediction.risk_level}")
            ... else:
            ...     print("[OK] Normal reading")
        """
        try:
            response = self.session.post(
                f"{self.model_url}/predict",
                json={
                    "pressure": pressure,
                    "temperature": temperature,
                    "ph_level": ph_level,
                    "dissolved_oxygen": dissolved_oxygen,
                    "turbidity": turbidity,
                    "flow_rate": flow_rate,
                    "location": location,
                    "sensor_type": sensor_type
                }
            )
            response.raise_for_status()
            data = response.json()
            
            pred = data["prediction"]
            return PredictionResult(
                ensemble_prediction=pred["ensemble_prediction"],
                ensemble_confidence=pred["ensemble_confidence"],
                anomaly_detected=pred["anomaly_detected"],
                risk_level=data["analysis"]["risk_level"],
                knn_prediction=pred["knn_prediction"],
                lstm_prediction=pred["lstm_prediction"],
                execution_time_ms=data["execution_time_ms"]
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict anomalies for multiple sensor readings.
        
        Args:
            samples: List of sensor reading dicts
        
        Returns:
            Batch prediction results
        
        Example:
            >>> samples = [
            ...     {"pressure": 65, "temperature": 22, ...},
            ...     {"pressure": 120, "temperature": 28, ...}
            ... ]
            >>> results = client.predict_batch(samples)
            >>> print(f"Anomalies: {results['anomalies_found']}/{results['total_processed']}")
        """
        response = self.session.post(
            f"{self.model_url}/predict/batch",
            json={"samples": samples}
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================================================
    # Combined Workflows
    # ========================================================================
    
    def handle_sensor_anomaly(
        self,
        pressure: float,
        temperature: float,
        ph_level: float,
        dissolved_oxygen: float,
        turbidity: float,
        flow_rate: float
    ) -> Dict[str, Any]:
        """
        Complete workflow: detect anomaly, then ask RAG for advice.
        
        Args:
            pressure: Sensor readings...
            temperature: ...
            ph_level: ...
            dissolved_oxygen: ...
            turbidity: ...
            flow_rate: ...
        
        Returns:
            Combined result with prediction and advice
        """
        # 1. Predict anomaly
        prediction = self.predict_anomaly(
            pressure, temperature, ph_level,
            dissolved_oxygen, turbidity, flow_rate
        )
        
        result = {
            "prediction": prediction,
            "advice": None
        }
        
        # 2. If anomaly, ask for advice
        if prediction.anomaly_detected:
            question = f"""
            I detected anomalies: pressure={pressure} PSI, 
            temperature={temperature}°C, dissolved_oxygen={dissolved_oxygen} mg/L.
            What should I do?
            """
            
            try:
                answer = self.ask(question, mode="concise")
                result["advice"] = answer.answer
            except Exception as e:
                logger.error(f"Failed to get advice: {e}")
        
        return result


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize client
    client = WaterAnomalyClient()
    
    print("[CHECK] Checking API health...")
    health = client.check_health()
    print(f"Health: {health}\n")
    
    # Example 1: Ask question (concise mode)
    print("[QUERY] Asking question (concise mode)...")
    try:
        answer = client.ask("What is normal water pressure?", mode="concise")
        print(f"Q: What is normal water pressure?")
        print(f"A: {answer.answer}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Predict anomaly
    print("[PREDICT] Predicting water anomaly...")
    try:
        prediction = client.predict_anomaly(
            pressure=65,
            temperature=22,
            ph_level=7.2,
            dissolved_oxygen=8.5,
            turbidity=0.3,
            flow_rate=150
        )
        
        print(f"Prediction: {prediction.ensemble_prediction}")
        print(f"Confidence: {prediction.ensemble_confidence:.2%}")
        print(f"Anomaly: {prediction.anomaly_detected}")
        if prediction.anomaly_detected:
            print(f"Risk Level: {prediction.risk_level}\n")
        else:
            print("")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Batch prediction
    print("[BATCH] Batch prediction...")
    try:
        samples = [
            {"pressure": 65, "temperature": 22, "ph_level": 7.2, 
             "dissolved_oxygen": 8.5, "turbidity": 0.3, "flow_rate": 150},
            {"pressure": 120, "temperature": 28, "ph_level": 6.5, 
             "dissolved_oxygen": 4.2, "turbidity": 2.1, "flow_rate": 250}
        ]
        
        results = client.predict_batch(samples)
        print(f"Processed: {results['total_processed']}")
        print(f"Anomalies found: {results['anomalies_found']}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 4: Combined workflow
    print("[WORKFLOW] Complete workflow (anomaly detection + advice)...")
    try:
        result = client.handle_sensor_anomaly(
            pressure=65, temperature=22, ph_level=7.2,
            dissolved_oxygen=8.5, turbidity=0.3, flow_rate=150
        )
        
        print(f"Anomaly detected: {result['prediction'].anomaly_detected}")
        if result['advice']:
            print(f"Advice: {result['advice']}")
        print()
    except Exception as e:
        print(f"Error: {e}")
