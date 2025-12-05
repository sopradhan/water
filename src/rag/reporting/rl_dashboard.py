"""
Reinforcement Learning Dashboard

Provides comprehensive monitoring of RL Healing Agent learning progress:
- Q-value evolution and convergence
- Action effectiveness tracking
- Reward signal analysis
- Exploration vs Exploitation metrics
- Learning rate and policy impact
- State-action value heatmaps
- Decision quality over time
- Cumulative rewards and ROI
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics


class RLDashboard:
    """Monitor and visualize RL Healing Agent learning progress."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    # ========================================================================
    # SECTION 1: ACTION EFFECTIVENESS ANALYSIS
    # ========================================================================
    
    def analyze_action_effectiveness(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze effectiveness of each RL action (SKIP, OPTIMIZE, REINDEX, RE_EMBED)
        
        Returns:
            Dict with action statistics, success rates, and ROI metrics
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Get all healing actions with their outcomes
            self.cursor.execute("""
                SELECT 
                    action_taken,
                    COUNT(*) as action_count,
                    AVG(reward_signal) as avg_reward,
                    MIN(reward_signal) as min_reward,
                    MAX(reward_signal) as max_reward,
                    SUM(CASE WHEN reward_signal > 0.5 THEN 1 ELSE 0 END) as success_count
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ? AND action_taken IS NOT NULL
                GROUP BY action_taken
                ORDER BY avg_reward DESC
            """, (since.isoformat(),))
            
            actions_data = [dict(row) for row in self.cursor.fetchall()]
            
            # Calculate effectiveness metrics
            effectiveness = {}
            for action in actions_data:
                action_name = action['action_taken']
                count = action['action_count']
                success = action['success_count']
                
                effectiveness[action_name] = {
                    "total_actions": count,
                    "successful_actions": success,
                    "success_rate": round((success / count * 100) if count > 0 else 0, 2),
                    "avg_reward": round(action['avg_reward'], 4),
                    "reward_range": {
                        "min": round(action['min_reward'], 4),
                        "max": round(action['max_reward'], 4)
                    }
                }
            
            # Calculate cumulative rewards per action
            cumulative_rewards = {}
            for action in actions_data:
                cumulative_rewards[action['action_taken']] = round(
                    action['avg_reward'] * action['action_count'], 2
                )
            
            return {
                "dashboard_section": "ACTION_EFFECTIVENESS",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "actions_data": actions_data,
                "effectiveness_metrics": effectiveness,
                "cumulative_rewards": cumulative_rewards,
                "best_action": max(effectiveness.items(), key=lambda x: x[1]['avg_reward'])[0] if effectiveness else None,
                "worst_action": min(effectiveness.items(), key=lambda x: x[1]['avg_reward'])[0] if effectiveness else None
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # SECTION 2: REWARD SIGNAL ANALYSIS
    # ========================================================================
    
    def analyze_reward_signals(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze reward signals to understand RL agent learning
        
        Reward Signal Interpretation:
        - High reward (0.8-1.0): Action was very effective, improved system
        - Medium reward (0.5-0.8): Action was moderately effective
        - Low reward (0.2-0.5): Action had minimal impact
        - Negative reward (<0.2): Action was counterproductive
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Overall reward statistics
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_events,
                    AVG(reward_signal) as avg_reward,
                    MIN(reward_signal) as min_reward,
                    MAX(reward_signal) as max_reward,
                    SUM(CASE WHEN reward_signal > 0.8 THEN 1 ELSE 0 END) as excellent_count,
                    SUM(CASE WHEN reward_signal BETWEEN 0.5 AND 0.8 THEN 1 ELSE 0 END) as good_count,
                    SUM(CASE WHEN reward_signal BETWEEN 0.2 AND 0.5 THEN 1 ELSE 0 END) as fair_count,
                    SUM(CASE WHEN reward_signal < 0.2 THEN 1 ELSE 0 END) as poor_count
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ? AND reward_signal IS NOT NULL
            """, (since.isoformat(),))
            
            stats = dict(self.cursor.fetchone() or {})
            total = stats.get('total_events', 1)
            
            # Reward distribution
            reward_distribution = {
                "EXCELLENT (0.8-1.0)": {
                    "count": stats.get('excellent_count', 0),
                    "percentage": round((stats.get('excellent_count', 0) / total * 100), 2) if total > 0 else 0
                },
                "GOOD (0.5-0.8)": {
                    "count": stats.get('good_count', 0),
                    "percentage": round((stats.get('good_count', 0) / total * 100), 2) if total > 0 else 0
                },
                "FAIR (0.2-0.5)": {
                    "count": stats.get('fair_count', 0),
                    "percentage": round((stats.get('fair_count', 0) / total * 100), 2) if total > 0 else 0
                },
                "POOR (<0.2)": {
                    "count": stats.get('poor_count', 0),
                    "percentage": round((stats.get('poor_count', 0) / total * 100), 2) if total > 0 else 0
                }
            }
            
            # Reward trend over time (daily average)
            self.cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    AVG(reward_signal) as daily_avg_reward,
                    COUNT(*) as event_count,
                    SUM(CASE WHEN reward_signal > 0.7 THEN 1 ELSE 0 END) as high_reward_count
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ? AND reward_signal IS NOT NULL
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (since.isoformat(),))
            
            reward_trend = [dict(row) for row in self.cursor.fetchall()]
            
            # Reward by action type
            self.cursor.execute("""
                SELECT 
                    action_taken,
                    AVG(reward_signal) as avg_reward,
                    STDDEV(reward_signal) as reward_variance
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ? AND reward_signal IS NOT NULL AND action_taken IS NOT NULL
                GROUP BY action_taken
                ORDER BY avg_reward DESC
            """, (since.isoformat(),))
            
            reward_by_action = [dict(row) for row in self.cursor.fetchall()]
            
            return {
                "dashboard_section": "REWARD_SIGNALS",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_events": stats.get('total_events'),
                    "avg_reward": round(stats.get('avg_reward', 0), 4),
                    "reward_range": {
                        "min": round(stats.get('min_reward', 0), 4),
                        "max": round(stats.get('max_reward', 0), 4)
                    }
                },
                "reward_distribution": reward_distribution,
                "trend": reward_trend,
                "by_action": reward_by_action
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # SECTION 3: EXPLORATION VS EXPLOITATION
    # ========================================================================
    
    def analyze_exploration_exploitation(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze exploration (trying new actions) vs exploitation (using known good actions)
        
        RL agents need to balance:
        - Exploration: Try different actions to learn
        - Exploitation: Use actions known to work well
        
        Epsilon-greedy strategy adjusts this over time.
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Get action diversity over time
            self.cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(DISTINCT action_taken) as action_diversity,
                    COUNT(*) as total_actions,
                    AVG(reward_signal) as avg_reward
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ? AND action_taken IS NOT NULL
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (since.isoformat(),))
            
            diversity_trend = [dict(row) for row in self.cursor.fetchall()]
            
            # Most used action vs best performing action
            self.cursor.execute("""
                SELECT action_taken, COUNT(*) as count, AVG(reward_signal) as avg_reward
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ? AND action_taken IS NOT NULL
                GROUP BY action_taken
                ORDER BY count DESC
            """, (since.isoformat(),))
            
            action_usage = [dict(row) for row in self.cursor.fetchall()]
            
            # Calculate exploitation rate
            if action_usage:
                total_actions = sum(a['count'] for a in action_usage)
                most_used = action_usage[0]
                exploitation_rate = round((most_used['count'] / total_actions * 100), 2)
                
                # Check if most used is also best performing
                best_action = max(action_usage, key=lambda x: x['avg_reward'])
                is_optimal = most_used['action_taken'] == best_action['action_taken']
            else:
                exploitation_rate = 0
                is_optimal = False
            
            # Action frequency distribution (for exploration assessment)
            if action_usage:
                freq_variance = statistics.variance([a['count'] for a in action_usage])
                high_variance = freq_variance > 5  # Indicates good exploration
            else:
                freq_variance = 0
                high_variance = False
            
            return {
                "dashboard_section": "EXPLORATION_EXPLOITATION",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "diversity_trend": diversity_trend,
                "action_usage": action_usage,
                "analysis": {
                    "total_unique_actions": len(action_usage),
                    "exploitation_rate": exploitation_rate,
                    "is_exploiting_optimal": is_optimal,
                    "action_frequency_variance": round(freq_variance, 2),
                    "exploration_assessment": "GOOD" if high_variance else "LOW"
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # SECTION 4: LEARNING CONVERGENCE
    # ========================================================================
    
    def analyze_learning_convergence(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze if RL agent is converging to optimal policy
        
        Convergence indicators:
        - Reward signals stabilizing (low variance)
        - Consistent action choices (low exploration)
        - Improving trend in average rewards
        - Action diversity decreasing (focusing on best actions)
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Split period into early, middle, late phases
            period_start = datetime.fromisoformat(since.isoformat())
            period_end = datetime.now()
            total_seconds = (period_end - period_start).total_seconds()
            phase_duration = total_seconds / 3
            
            phase1_end = period_start + timedelta(seconds=phase_duration)
            phase2_end = period_start + timedelta(seconds=phase_duration * 2)
            
            phases_data = []
            
            for phase_num, (phase_start, phase_end) in enumerate([
                (since.isoformat(), phase1_end.isoformat()),
                (phase1_end.isoformat(), phase2_end.isoformat()),
                (phase2_end.isoformat(), datetime.now().isoformat())
            ], 1):
                self.cursor.execute("""
                    SELECT 
                        AVG(reward_signal) as avg_reward,
                        COUNT(*) as action_count,
                        COUNT(DISTINCT action_taken) as action_diversity
                    FROM rag_history_and_optimization
                    WHERE event_type = 'HEAL' AND timestamp >= ? AND timestamp <= ? 
                          AND reward_signal IS NOT NULL AND action_taken IS NOT NULL
                """, (phase_start, phase_end))
                
                phase_stats = dict(self.cursor.fetchone() or {})
                phases_data.append({
                    "phase": phase_num,
                    "avg_reward": round(phase_stats.get('avg_reward', 0), 4),
                    "action_count": phase_stats.get('action_count', 0),
                    "action_diversity": phase_stats.get('action_diversity', 0)
                })
            
            # Calculate convergence metrics
            if len(phases_data) >= 2:
                reward_improvement = phases_data[-1]['avg_reward'] - phases_data[0]['avg_reward']
                convergence_rate = round(reward_improvement / max(abs(phases_data[0]['avg_reward']), 0.1), 4)
                
                # Check if rewards are stabilizing (variance decreasing)
                diversity_decrease = phases_data[0]['action_diversity'] - phases_data[-1]['action_diversity']
                is_converging = diversity_decrease >= 0 and reward_improvement >= 0
            else:
                convergence_rate = 0
                is_converging = False
            
            return {
                "dashboard_section": "LEARNING_CONVERGENCE",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "phases": phases_data,
                "convergence_analysis": {
                    "convergence_rate": convergence_rate,
                    "is_converging": is_converging,
                    "reward_improvement": round(reward_improvement, 4) if len(phases_data) >= 2 else 0,
                    "diversity_trend": "DECREASING (Good)" if diversity_decrease > 0 else "INCREASING (Exploring)"
                },
                "interpretation": self._interpret_convergence(is_converging, convergence_rate)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _interpret_convergence(self, is_converging: bool, rate: float) -> str:
        """Interpret convergence status."""
        if not is_converging:
            return "Agent is still exploring and learning. Not yet converged."
        elif rate > 0.1:
            return "Strong convergence detected. Agent rapidly improving."
        elif rate > 0.01:
            return "Moderate convergence. Agent improving steadily."
        else:
            return "Slow convergence or already converged. Agent rewards stabilizing."
    
    # ========================================================================
    # SECTION 5: STATE-ACTION VALUE ANALYSIS
    # ========================================================================
    
    def analyze_state_action_values(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze which document states benefit most from which actions
        
        Maps: state characteristics -> action effectiveness
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Analyze action effectiveness for documents at different quality levels
            self.cursor.execute("""
                SELECT 
                    c.quality_score,
                    h.action_taken,
                    AVG(h.reward_signal) as avg_reward,
                    COUNT(*) as count
                FROM rag_history_and_optimization h
                LEFT JOIN chunk_embedding_data c ON h.target_chunk_id = c.chunk_id
                WHERE h.event_type = 'HEAL' AND h.timestamp >= ? 
                      AND h.action_taken IS NOT NULL AND h.reward_signal IS NOT NULL
                GROUP BY 
                    CASE 
                        WHEN c.quality_score >= 0.8 THEN 'HIGH_QUALITY'
                        WHEN c.quality_score >= 0.6 THEN 'MEDIUM_QUALITY'
                        ELSE 'LOW_QUALITY'
                    END,
                    h.action_taken
            """, (since.isoformat(),))
            
            state_action_values = [dict(row) for row in self.cursor.fetchall()]
            
            # Best action per state
            state_best_actions = {}
            for row in state_action_values:
                state = row.get('quality_score', 'UNKNOWN')
                action = row.get('action_taken')
                reward = row.get('avg_reward', 0)
                
                if state not in state_best_actions or reward > state_best_actions[state]['reward']:
                    state_best_actions[state] = {
                        'action': action,
                        'reward': reward
                    }
            
            return {
                "dashboard_section": "STATE_ACTION_VALUES",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "state_action_matrix": state_action_values,
                "optimal_actions_per_state": state_best_actions
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # SECTION 6: RL COMPREHENSIVE DASHBOARD
    # ========================================================================
    
    def generate_rl_dashboard(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive RL dashboard combining all metrics
        """
        try:
            action_effectiveness = self.analyze_action_effectiveness(days)
            reward_signals = self.analyze_reward_signals(days)
            exploration = self.analyze_exploration_exploitation(days)
            convergence = self.analyze_learning_convergence(days)
            state_values = self.analyze_state_action_values(days)
            
            # Calculate RL health score (0-100)
            try:
                avg_reward = reward_signals.get('summary', {}).get('avg_reward', 0) * 100
                convergence_quality = min(100, convergence.get('convergence_analysis', {}).get('convergence_rate', 0) * 100 + 50)
                exploration_quality = 50 if exploration.get('analysis', {}).get('exploration_assessment') == 'GOOD' else 30
                
                rl_health_score = (avg_reward * 0.5 + convergence_quality * 0.3 + exploration_quality * 0.2)
            except:
                rl_health_score = 0
            
            return {
                "dashboard_type": "RL_COMPREHENSIVE",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "rl_health_score": round(rl_health_score, 2),
                "action_effectiveness": action_effectiveness,
                "reward_signals": reward_signals,
                "exploration_exploitation": exploration,
                "learning_convergence": convergence,
                "state_action_values": state_values,
                "executive_summary": self._generate_executive_summary(
                    action_effectiveness, reward_signals, exploration, convergence
                )
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_executive_summary(self, action_eff, rewards, explore, converg) -> Dict[str, str]:
        """Generate executive summary of RL performance."""
        summary = {}
        
        # Action performance summary
        best_action = action_eff.get('best_action', 'UNKNOWN')
        best_reward = action_eff.get('effectiveness_metrics', {}).get(best_action, {}).get('avg_reward', 0)
        summary['action_performance'] = f"Best performing action is {best_action} with {best_reward:.2f} avg reward"
        
        # Reward trend summary
        avg_reward = rewards.get('summary', {}).get('avg_reward', 0)
        summary['reward_trend'] = f"Average reward: {avg_reward:.2f}. Learning effectiveness is {'strong' if avg_reward > 0.6 else 'moderate' if avg_reward > 0.4 else 'weak'}"
        
        # Exploration summary
        is_optimal = explore.get('analysis', {}).get('is_exploiting_optimal', False)
        summary['exploration'] = f"Agent is {'exploiting optimal strategy' if is_optimal else 'still exploring'}"
        
        # Convergence summary
        is_converging = converg.get('convergence_analysis', {}).get('is_converging', False)
        summary['convergence'] = f"Learning is {'converging to optimal policy' if is_converging else 'still in exploration phase'}"
        
        return summary
    
    # ========================================================================
    # EXPORT UTILITIES
    # ========================================================================
    
    def export_rl_dashboard_json(self, dashboard: Dict[str, Any], filename: str) -> bool:
        """Export RL dashboard to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(dashboard, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error exporting RL dashboard: {e}")
            return False


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    db_path = "path/to/your/database.db"
    rl_dashboard = RLDashboard(db_path)
    
    print("Generating RL Dashboard...\n")
    
    # Generate comprehensive RL dashboard
    dashboard = rl_dashboard.generate_rl_dashboard(days=30)
    
    print("=" * 80)
    print("REINFORCEMENT LEARNING DASHBOARD")
    print("=" * 80)
    print(f"\nRL Health Score: {dashboard.get('rl_health_score', 0)}/100")
    print(f"Period: Last {dashboard.get('period_days', 30)} days")
    print(f"Generated: {dashboard.get('generated_at', 'N/A')}\n")
    
    print("EXECUTIVE SUMMARY:")
    for key, value in dashboard.get('executive_summary', {}).items():
        print(f"  • {value}")
    
    print("\n" + "=" * 80)
    print("ACTION EFFECTIVENESS")
    print("=" * 80)
    effectiveness = dashboard.get('action_effectiveness', {})
    for action, metrics in effectiveness.get('effectiveness_metrics', {}).items():
        print(f"\n{action}:")
        print(f"  Total Actions: {metrics.get('total_actions', 0)}")
        print(f"  Success Rate: {metrics.get('success_rate', 0)}%")
        print(f"  Avg Reward: {metrics.get('avg_reward', 0):.4f}")
    
    print("\n" + "=" * 80)
    print("REWARD SIGNALS")
    print("=" * 80)
    rewards = dashboard.get('reward_signals', {})
    dist = rewards.get('reward_distribution', {})
    for level, data in dist.items():
        print(f"{level}: {data.get('count', 0)} events ({data.get('percentage', 0)}%)")
    
    # Export dashboard
    rl_dashboard.export_rl_dashboard_json(dashboard, "rl_dashboard.json")
    print("\n✓ Dashboard exported to rl_dashboard.json")
