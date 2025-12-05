#!/usr/bin/env python3
"""
Test script to verify:
1. SQLite threading fixes
2. Feedback endpoint integration
3. RL agent feedback processing
4. Visualization serialization
"""
import sys
import time
import json
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_rl_agent_feedback():
    """Test RL agent feedback processing"""
    print("\n" + "="*70)
    print("TEST 1: RL Agent Feedback Processing")
    print("="*70)
    
    try:
        from src.rag.agents.healing_agent.rl_healing_agent import RLHealingAgent
        from src.rag.rag_db_models.config.env_config import EnvConfig
        
        # Create agent
        db_path = EnvConfig.get_rag_db_path()
        agent = RLHealingAgent(db_path)
        
        # Test feedback processing
        feedback = {
            'rating': 5,
            'question': 'What is water management?',
            'answer': 'Water management is the practice of managing water resources...',
            'user_id': 'user_123',
            'timestamp': time.time(),
            'feedback_text': 'Great answer!'
        }
        
        result = agent.process_feedback(feedback)
        
        print(f"✓ Feedback processed successfully")
        print(f"  - Rating: {result.get('rating')}/5")
        print(f"  - Reward signal: {result.get('reward_signal')}")
        print(f"  - Learning applied: {result.get('rl_learning_applied')}")
        print(f"  - Feedback ID: {result.get('feedback_id')}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_threading_safety():
    """Test SQLite thread-safe access"""
    print("\n" + "="*70)
    print("TEST 2: SQLite Thread Safety")
    print("="*70)
    
    try:
        from src.rag.rag_db_models.db_models.rag_history_model import RAGHistoryModel
        from concurrent.futures import ThreadPoolExecutor
        import json
        
        model = RAGHistoryModel()
        
        def log_in_thread(thread_id):
            """Log from a thread"""
            metrics = json.dumps({'thread': thread_id})
            context = json.dumps({'timestamp': time.time()})
            
            result = model.log_query(
                query_text=f"Query from thread {thread_id}",
                target_doc_id=f"doc_{thread_id}",
                metrics_json=metrics,
                context_json=context,
                session_id=f"session_{thread_id}"
            )
            return result
        
        # Test with multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(log_in_thread, i) for i in range(4)]
            results = [f.result() for f in futures]
        
        if all(r is not None for r in results):
            print(f"✓ Thread-safe logging successful")
            print(f"  - Logged {len(results)} records from {len(results)} threads")
            return True
        else:
            print(f"✗ Some logging operations failed")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_serialization():
    """Test visualization serialization with depth limits"""
    print("\n" + "="*70)
    print("TEST 3: Visualization Serialization")
    print("="*70)
    
    try:
        from src.rag.visualization.langgraph_visualizer import save_visualization
        from src.rag.visualization.langgraph_visualizer import GraphVisualization
        
        # Create a mock visualization with nested data
        viz = GraphVisualization(
            session_id="test_session",
            states=[{'node': 'retrieve', 'data': {'nested': {'deep': {'value': 'test'}}}}],
            transitions=[('retrieve', 'answer')],
            durations={'retrieve': 1.5, 'answer': 2.3}
        )
        
        # Try to save (this used to cause recursion errors)
        output_dir = project_root / "test_output"
        result = save_visualization(viz, str(output_dir))
        
        print(f"✓ Visualization serialized successfully")
        print(f"  - File: {result.get('visualization')}")
        print(f"  - Status: {result.get('status')}")
        
        # Verify the file can be loaded
        viz_file = Path(result.get('visualization'))
        with open(viz_file, 'r') as f:
            data = json.load(f)
        
        print(f"  - JSON valid: Yes")
        print(f"  - Keys: {list(data.keys())[:3]}...")
        
        # Cleanup
        viz_file.unlink()
        output_dir.rmdir()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_feedback_endpoint():
    """Test the feedback endpoint structure"""
    print("\n" + "="*70)
    print("TEST 4: Feedback API Endpoint")
    print("="*70)
    
    try:
        from src.rag.agents.langgraph_agent.api import FeedbackRequest, FeedbackResponse
        
        # Create a feedback request
        req = FeedbackRequest(
            session_id="test_session",
            question="What is water?",
            answer="Water is H2O...",
            rating=4,
            user_id=1,
            company_id=1,
            feedback_text="Good answer"
        )
        
        print(f"✓ FeedbackRequest model valid")
        print(f"  - Rating: {req.rating}/5")
        print(f"  - Session: {req.session_id}")
        
        # Validate response model
        resp = FeedbackResponse(
            success=True,
            feedback_id="fb_12345",
            message="Feedback received",
            rl_learning_applied=True,
            rating=4
        )
        
        print(f"✓ FeedbackResponse model valid")
        print(f"  - Success: {resp.success}")
        print(f"  - RL Learning: {resp.rl_learning_applied}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("RUNNING SYSTEM FIX VERIFICATION TESTS")
    print("="*70)
    
    tests = [
        ("RL Agent Feedback", test_rl_agent_feedback),
        ("SQLite Thread Safety", test_threading_safety),
        ("Visualization Serialization", test_visualization_serialization),
        ("Feedback API Endpoint", test_api_feedback_endpoint),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! System is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
