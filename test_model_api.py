#!/usr/bin/env python
"""
Model API Testing Script

Tests all Model API endpoints with real water anomaly detection data.
Demonstrates single prediction, batch prediction, and health checks.
"""

import requests
import json
import time
from typing import Dict, List, Any
import sys

# Configuration
MODEL_API_URL = "http://localhost:8002"
TIMEOUT = 30

# Test data - realistic water sensor readings
TEST_SAMPLES = {
    "normal": {
        "pressure": 65.0,
        "temperature": 22.0,
        "ph_level": 7.2,
        "dissolved_oxygen": 8.5,
        "turbidity": 0.3,
        "flow_rate": 150.0,
        "location": "Plant A",
        "sensor_type": "Type1"
    },
    "anomaly_high_turbidity": {
        "pressure": 70.0,
        "temperature": 25.0,
        "ph_level": 6.8,
        "dissolved_oxygen": 4.2,
        "turbidity": 5.5,  # HIGH - indicates particles
        "flow_rate": 120.0,
        "location": "Plant B",
        "sensor_type": "Type2"
    },
    "anomaly_low_oxygen": {
        "pressure": 68.0,
        "temperature": 28.0,
        "ph_level": 7.5,
        "dissolved_oxygen": 2.1,  # LOW - indicates contamination
        "turbidity": 1.2,
        "flow_rate": 180.0,
        "location": "Plant C",
        "sensor_type": "Type3"
    },
    "anomaly_pressure_spike": {
        "pressure": 120.0,  # HIGH - possible leak
        "temperature": 20.0,
        "ph_level": 7.0,
        "dissolved_oxygen": 7.5,
        "turbidity": 0.5,
        "flow_rate": 250.0,  # Increased flow
        "location": "Plant D",
        "sensor_type": "Type1"
    }
}


class ModelAPITester:
    """Test suite for Model API"""
    
    def __init__(self, base_url: str = MODEL_API_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def print_section(self, title: str):
        """Print formatted section header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    
    def print_test(self, name: str, status: str, message: str = ""):
        """Print test result"""
        status_color = "✅" if status == "PASS" else "❌"
        print(f"{status_color} {name}")
        if message:
            print(f"   └─ {message}")
        self.results.append({"test": name, "status": status})
    
    def print_json(self, data: Dict[str, Any], indent: int = 2):
        """Pretty print JSON"""
        print(json.dumps(data, indent=indent))
    
    # =====================================================================
    # Test 1: Health Check
    # =====================================================================
    
    def test_health_check(self) -> bool:
        """Test /health endpoint"""
        self.print_section("TEST 1: Health Check")
        
        try:
            url = f"{self.base_url}/health"
            print(f"\nEndpoint: GET {url}")
            
            response = self.session.get(url, timeout=TIMEOUT)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse:")
                self.print_json(data)
                
                # Verify response structure
                if "status" in data:
                    self.print_test(
                        "Health Check",
                        "PASS",
                        f"Status: {data.get('status')}"
                    )
                    return True
                else:
                    self.print_test("Health Check", "FAIL", "Missing 'status' field")
                    return False
            else:
                self.print_test("Health Check", "FAIL", f"Status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            self.print_test(
                "Health Check",
                "FAIL",
                f"Cannot connect to {self.base_url}. Is the Model API running?"
            )
            return False
        except Exception as e:
            self.print_test("Health Check", "FAIL", str(e))
            return False
    
    # =====================================================================
    # Test 2: Single Prediction - Normal Reading
    # =====================================================================
    
    def test_single_prediction_normal(self) -> bool:
        """Test single prediction with normal data"""
        self.print_section("TEST 2: Single Prediction - Normal Reading")
        
        try:
            url = f"{self.base_url}/predict"
            print(f"\nEndpoint: POST {url}")
            
            payload = TEST_SAMPLES["normal"]
            print(f"\nPayload (Normal Water Reading):")
            self.print_json(payload)
            
            response = self.session.post(url, json=payload, timeout=TIMEOUT)
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse:")
                self.print_json(data)
                
                # Verify structure
                if "prediction" in data and "analysis" in data:
                    pred = data["prediction"]
                    analysis = data["analysis"]
                    
                    print(f"\nAnalysis:")
                    print(f"  Ensemble Prediction: {pred.get('ensemble_prediction')}")
                    print(f"  Anomaly Detected: {pred.get('anomaly_detected')}")
                    print(f"  Risk Level: {analysis.get('risk_level')}")
                    print(f"  Confidence: {pred.get('ensemble_confidence'):.2%}")
                    
                    self.print_test(
                        "Single Prediction (Normal)",
                        "PASS",
                        f"Anomaly: {pred.get('anomaly_detected')}, Risk: {analysis.get('risk_level')}"
                    )
                    return True
                else:
                    self.print_test("Single Prediction (Normal)", "FAIL", "Missing fields")
                    return False
            else:
                print(f"Error: {response.text}")
                self.print_test("Single Prediction (Normal)", "FAIL", f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.print_test("Single Prediction (Normal)", "FAIL", str(e))
            return False
    
    # =====================================================================
    # Test 3: Single Prediction - Anomaly (High Turbidity)
    # =====================================================================
    
    def test_single_prediction_anomaly_turbidity(self) -> bool:
        """Test single prediction with high turbidity anomaly"""
        self.print_section("TEST 3: Single Prediction - Anomaly (High Turbidity)")
        
        try:
            url = f"{self.base_url}/predict"
            print(f"\nEndpoint: POST {url}")
            
            payload = TEST_SAMPLES["anomaly_high_turbidity"]
            print(f"\nPayload (High Turbidity Anomaly):")
            self.print_json(payload)
            
            response = self.session.post(url, json=payload, timeout=TIMEOUT)
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse:")
                self.print_json(data)
                
                pred = data.get("prediction", {})
                analysis = data.get("analysis", {})
                
                print(f"\nAnalysis:")
                print(f"  Ensemble Prediction: {pred.get('ensemble_prediction')}")
                print(f"  Anomaly Detected: {pred.get('anomaly_detected')}")
                print(f"  Risk Level: {analysis.get('risk_level')}")
                print(f"  Confidence: {pred.get('ensemble_confidence'):.2%}")
                
                # For anomaly test, we expect anomaly_detected = True
                if pred.get('anomaly_detected'):
                    self.print_test(
                        "Single Prediction (Anomaly - Turbidity)",
                        "PASS",
                        f"Correctly detected! Risk: {analysis.get('risk_level')}"
                    )
                    return True
                else:
                    self.print_test(
                        "Single Prediction (Anomaly - Turbidity)",
                        "FAIL",
                        "Did not detect anomaly"
                    )
                    return False
            else:
                print(f"Error: {response.text}")
                self.print_test("Single Prediction (Anomaly - Turbidity)", "FAIL", f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.print_test("Single Prediction (Anomaly - Turbidity)", "FAIL", str(e))
            return False
    
    # =====================================================================
    # Test 4: Single Prediction - Anomaly (Low Oxygen)
    # =====================================================================
    
    def test_single_prediction_anomaly_oxygen(self) -> bool:
        """Test single prediction with low oxygen anomaly"""
        self.print_section("TEST 4: Single Prediction - Anomaly (Low Oxygen)")
        
        try:
            url = f"{self.base_url}/predict"
            print(f"\nEndpoint: POST {url}")
            
            payload = TEST_SAMPLES["anomaly_low_oxygen"]
            print(f"\nPayload (Low Dissolved Oxygen Anomaly):")
            self.print_json(payload)
            
            response = self.session.post(url, json=payload, timeout=TIMEOUT)
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse:")
                self.print_json(data)
                
                pred = data.get("prediction", {})
                analysis = data.get("analysis", {})
                
                print(f"\nAnalysis:")
                print(f"  Ensemble Prediction: {pred.get('ensemble_prediction')}")
                print(f"  Anomaly Detected: {pred.get('anomaly_detected')}")
                print(f"  Risk Level: {analysis.get('risk_level')}")
                print(f"  Confidence: {pred.get('ensemble_confidence'):.2%}")
                
                if pred.get('anomaly_detected'):
                    self.print_test(
                        "Single Prediction (Anomaly - Oxygen)",
                        "PASS",
                        f"Correctly detected! Risk: {analysis.get('risk_level')}"
                    )
                    return True
                else:
                    self.print_test(
                        "Single Prediction (Anomaly - Oxygen)",
                        "FAIL",
                        "Did not detect anomaly"
                    )
                    return False
            else:
                print(f"Error: {response.text}")
                self.print_test("Single Prediction (Anomaly - Oxygen)", "FAIL", f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.print_test("Single Prediction (Anomaly - Oxygen)", "FAIL", str(e))
            return False
    
    # =====================================================================
    # Test 5: Batch Prediction
    # =====================================================================
    
    def test_batch_prediction(self) -> bool:
        """Test batch prediction endpoint"""
        self.print_section("TEST 5: Batch Prediction")
        
        try:
            url = f"{self.base_url}/predict/batch"
            print(f"\nEndpoint: POST {url}")
            
            # Create batch with mixed samples
            samples = [
                TEST_SAMPLES["normal"],
                TEST_SAMPLES["anomaly_high_turbidity"],
                TEST_SAMPLES["anomaly_low_oxygen"],
                TEST_SAMPLES["anomaly_pressure_spike"]
            ]
            
            payload = {"samples": samples}
            print(f"\nBatch Payload: {len(samples)} samples")
            print(f"  Sample 1: Normal")
            print(f"  Sample 2: High Turbidity")
            print(f"  Sample 3: Low Oxygen")
            print(f"  Sample 4: Pressure Spike")
            
            response = self.session.post(url, json=payload, timeout=TIMEOUT)
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\nResponse Summary:")
                print(f"  Total Processed: {data.get('total_processed', 0)}")
                print(f"  Anomalies Found: {data.get('anomalies_found', 0)}")
                print(f"  Processing Time: {data.get('total_time_ms', 0):.2f}ms")
                
                # Detailed results
                if "predictions" in data:
                    print(f"\nDetailed Results:")
                    for i, pred in enumerate(data["predictions"], 1):
                        print(f"  Sample {i}:")
                        print(f"    └─ Anomaly: {pred.get('anomaly_detected')}, Risk: {pred.get('risk_level')}")
                
                self.print_test(
                    "Batch Prediction",
                    "PASS",
                    f"Processed {data.get('total_processed')} samples, Found {data.get('anomalies_found')} anomalies"
                )
                return True
            else:
                print(f"Error: {response.text}")
                self.print_test("Batch Prediction", "FAIL", f"Status {response.status_code}")
                return False
                
        except Exception as e:
            self.print_test("Batch Prediction", "FAIL", str(e))
            return False
    
    # =====================================================================
    # Test 6: Error Handling - Missing Field
    # =====================================================================
    
    def test_error_handling_missing_field(self) -> bool:
        """Test error handling with missing required field"""
        self.print_section("TEST 6: Error Handling - Missing Field")
        
        try:
            url = f"{self.base_url}/predict"
            print(f"\nEndpoint: POST {url}")
            
            # Remove required field
            payload = TEST_SAMPLES["normal"].copy()
            del payload["pressure"]  # Remove required field
            
            print(f"\nPayload (Missing 'pressure' field):")
            self.print_json(payload)
            
            response = self.session.post(url, json=payload, timeout=TIMEOUT)
            
            print(f"\nStatus Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error Response:")
                print(response.text)
                
                self.print_test(
                    "Error Handling (Missing Field)",
                    "PASS",
                    f"Correctly returned error code {response.status_code}"
                )
                return True
            else:
                self.print_test(
                    "Error Handling (Missing Field)",
                    "FAIL",
                    "Should have returned error"
                )
                return False
                
        except Exception as e:
            self.print_test("Error Handling (Missing Field)", "FAIL", str(e))
            return False
    
    # =====================================================================
    # Test 7: Performance - Execution Time
    # =====================================================================
    
    def test_performance(self) -> bool:
        """Test prediction performance"""
        self.print_section("TEST 7: Performance - Execution Time")
        
        try:
            url = f"{self.base_url}/predict"
            print(f"\nEndpoint: POST {url}")
            
            payload = TEST_SAMPLES["normal"]
            
            # Warm up
            self.session.post(url, json=payload, timeout=TIMEOUT)
            
            # Multiple runs
            times = []
            num_runs = 5
            print(f"\nRunning {num_runs} predictions...")
            
            for i in range(num_runs):
                start = time.time()
                response = self.session.post(url, json=payload, timeout=TIMEOUT)
                elapsed = (time.time() - start) * 1000  # Convert to ms
                times.append(elapsed)
                print(f"  Run {i+1}: {elapsed:.2f}ms")
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"\nPerformance Statistics:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Min: {min_time:.2f}ms")
            print(f"  Max: {max_time:.2f}ms")
            
            # Check if within reasonable bounds (< 1000ms)
            if avg_time < 1000:
                self.print_test(
                    "Performance",
                    "PASS",
                    f"Average prediction time: {avg_time:.2f}ms"
                )
                return True
            else:
                self.print_test(
                    "Performance",
                    "FAIL",
                    f"Performance degraded: {avg_time:.2f}ms"
                )
                return False
                
        except Exception as e:
            self.print_test("Performance", "FAIL", str(e))
            return False
    
    # =====================================================================
    # Test Summary
    # =====================================================================
    
    def print_summary(self):
        """Print test summary"""
        self.print_section("TEST SUMMARY")
        
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        total = len(self.results)
        
        print(f"\nResults:")
        print(f"  Total Tests: {total}")
        print(f"  Passed: {passed} ✅")
        print(f"  Failed: {failed} ❌")
        print(f"  Success Rate: {(passed/total*100):.1f}%")
        
        if failed > 0:
            print(f"\nFailed Tests:")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"  • {result['test']}")
        
        return failed == 0
    
    # =====================================================================
    # Run All Tests
    # =====================================================================
    
    def run_all(self) -> bool:
        """Run all tests"""
        print("\n" + "=" * 80)
        print("  MODEL API TEST SUITE")
        print("=" * 80)
        print(f"Target: {self.base_url}")
        print(f"Timeout: {TIMEOUT}s")
        
        # Run all tests
        all_passed = True
        
        # Test 1: Health Check
        if not self.test_health_check():
            print("\n⚠️  Health check failed! Other tests may not work.")
            all_passed = False
        
        time.sleep(0.5)
        
        # Test 2: Single Prediction - Normal
        if not self.test_single_prediction_normal():
            all_passed = False
        
        time.sleep(0.5)
        
        # Test 3: Single Prediction - Anomaly (Turbidity)
        if not self.test_single_prediction_anomaly_turbidity():
            all_passed = False
        
        time.sleep(0.5)
        
        # Test 4: Single Prediction - Anomaly (Oxygen)
        if not self.test_single_prediction_anomaly_oxygen():
            all_passed = False
        
        time.sleep(0.5)
        
        # Test 5: Batch Prediction
        if not self.test_batch_prediction():
            all_passed = False
        
        time.sleep(0.5)
        
        # Test 6: Error Handling
        if not self.test_error_handling_missing_field():
            all_passed = False
        
        time.sleep(0.5)
        
        # Test 7: Performance
        if not self.test_performance():
            all_passed = False
        
        # Print summary
        self.print_summary()
        
        return all_passed


def main():
    """Main entry point"""
    
    # Check if running with custom URL
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = MODEL_API_URL
    
    # Create tester and run
    tester = ModelAPITester(base_url)
    success = tester.run_all()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
