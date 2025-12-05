"""
Water Network Synthetic Dataset Generator - HYDRAULIC PHYSICS VERSION
---------------------------------------------------------------------
Realistic water distribution network simulation with:
- Booster pump system (pump-driven pressure control)
- Elevation-based system load
- Hazen-Williams friction losses
- Time-varying demand model (hourly + seasonal)
- Zone-specific consumer types
- Leak propagation effects
"""

import json
import numpy as np
import math
from datetime import datetime, timedelta
from pathlib import Path
from src.config import config


# ========================================
# HYDRAULIC SYSTEM CONFIGURATION
# ========================================

# Elevation map (meters above mean sea level)
DEFAULT_ELEVATIONS = {
    "Master": 48.0,
    "Zone0": 44.0,
    "Zone1": 58.0,
    "Zone2": 33.0,
    "Zone3": 22.0,
    "Zone4": 5.0,
}

# Per-branch pipe specifications (diameter mm, length m, Hazen-Williams C coefficient)
BRANCH_SPECS = {
    ("Master", "Zone0"): {"diameter_mm": 350, "length_m": 600, "C": 120},
    ("Master", "Zone1"): {"diameter_mm": 300, "length_m": 1200, "C": 120},
    ("Master", "Zone2"): {"diameter_mm": 250, "length_m": 1800, "C": 110},
    ("Master", "Zone3"): {"diameter_mm": 200, "length_m": 3000, "C": 105},
}

# Zone characteristics
ZONE_CHARACTERISTICS = {
    "Zone0": {"type": "local_distribution", "base_demand_lpm": 800, "population": 2000},
    "Zone1": {"type": "residential", "base_demand_lpm": 1200, "population": 5000},
    "Zone2": {"type": "commercial", "base_demand_lpm": 1500, "population": 3000},
    "Zone3": {"type": "industrial", "base_demand_lpm": 2000, "population": 1000},
}

# Pump system parameters
PUMP_SETPOINT_PSI = 120.0  # Booster maintains this pressure
PUMP_MIN_PSI = 100.0
PUMP_MAX_PSI = 140.0


# ========================================
# HYDRAULIC CALCULATIONS
# ========================================

def head_m_from_hazen_williams(L_m, Q_m3s, D_m, C=120):
    """
    Hazen-Williams head loss in meters.
    L_m: pipe length (meters)
    Q_m3s: flow rate (m³/s)
    D_m: pipe diameter (meters)
    C: Hazen-Williams coefficient (typically 100-150)
    """
    if Q_m3s <= 0 or D_m <= 0:
        return 0.0
    try:
        hf = 10.67 * L_m * (Q_m3s ** 1.852) / ((C ** 1.852) * (D_m ** 4.87))
        return max(0.0, hf)
    except:
        return 0.0


def compute_zone_pressure_with_booster(
    pump_setpoint_psi,
    master_elev_m,
    zone_elev_m,
    branch_spec,
    branch_flow_lpm,
    valve_open=1.0,
):
    """
    Compute zone pressure (psi) from booster pump system.
    
    Physics:
    - Pump maintains setpoint pressure at master
    - Elevation difference affects pump workload (internal)
    - Friction loss reduces pressure from master to zone
    - Final zone pressure = pump_setpoint - friction_loss
    """
    D_m = branch_spec["diameter_mm"] / 1000.0
    L_m = branch_spec["length_m"]
    C = branch_spec["C"]
    
    # Effective flow after valve
    Q_lpm = max(0.0, branch_flow_lpm * valve_open)
    Q_m3s = (Q_lpm / 60.0) / 1000.0
    
    # Friction head loss
    hf_m = head_m_from_hazen_williams(L_m, Q_m3s, D_m, C)
    hf_psi = hf_m * 0.433  # Convert meters to psi
    
    # Elevation difference affects pump workload (but not zone pressure directly)
    elev_diff_m = zone_elev_m - master_elev_m
    pump_workload_psi = abs(elev_diff_m * 0.433)
    
    # Zone pressure = pump setpoint - friction loss + sensor noise
    P_zone = pump_setpoint_psi - hf_psi
    P_zone += np.random.normal(0, 0.15)  # Sensor noise
    
    # Ensure pressure stays within realistic bounds
    P_zone = np.clip(P_zone, PUMP_MIN_PSI, PUMP_MAX_PSI)
    
    return float(P_zone), {
        "friction_head_m": hf_m,
        "friction_psi": hf_psi,
        "pump_workload_psi": pump_workload_psi,
        "elevation_diff_m": elev_diff_m,
    }


def hourly_demand_factor(hour):
    """
    Realistic 24-hour demand curve.
    Peak at 7am and 7pm, low at night.
    """
    # Two peaks: morning (7am) and evening (7pm)
    morning_peak = np.exp(-((hour - 7) ** 2) / (2 * 2.5 ** 2))
    evening_peak = np.exp(-((hour - 19) ** 2) / (2 * 2.5 ** 2))
    base = 0.4  # Night baseline
    return base + 0.8 * (morning_peak + 0.7 * evening_peak)


def seasonal_demand_multiplier(month):
    """
    Seasonal variation in water demand.
    Peak in summer, lower in monsoon/winter.
    """
    if month in [6, 7, 8, 9]:  # Monsoon
        return 0.85
    elif month in [3, 4, 5]:  # Summer (peak)
        return 1.2
    else:  # Winter/Spring
        return 1.0


def zone_demand_factor(zone_name):
    """Zone-specific demand based on consumer type."""
    factors = {
        "Zone0": 0.6,   # Local distribution hub (lower demand)
        "Zone1": 1.0,   # Residential (baseline)
        "Zone2": 1.3,   # Commercial (higher daytime demand)
        "Zone3": 1.5,   # Industrial (high constant demand)
    }
    return factors.get(zone_name, 1.0)


def compute_zone_demand(base_demand_lpm, hour, month, zone_name, leak_factor=1.0):
    """
    Compute dynamic zone demand based on time and characteristics.
    
    FinalDemand = BaseDemand × HourlyFactor × SeasonalFactor × ZoneFactor × LeakFactor
    """
    hour_factor = hourly_demand_factor(hour)
    seasonal_factor = seasonal_demand_multiplier(month)
    zone_factor = zone_demand_factor(zone_name)
    
    demand = base_demand_lpm * hour_factor * seasonal_factor * zone_factor * leak_factor
    return max(0.0, demand)


def compute_flow_distribution(master_flow_lpm, zones_demand, total_demand):
    """
    Distribute master flow to zones proportional to their demand.
    If total demand > master flow, apply pressure reduction system-wide.
    """
    if total_demand <= 0:
        return {zone: 0.0 for zone in zones_demand}
    
    # Flow distribution proportional to demand
    distribution = {}
    for zone, demand in zones_demand.items():
        distribution[zone] = master_flow_lpm * (demand / total_demand)
    
    return distribution


# ========================================
# DATA GENERATOR CLASS
# ========================================

class WaterDatasetGenerator:
    """
    Generates realistic water distribution network data with hydraulic physics.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else config.TRAINING_DATA_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.soil_types = ["Sandy", "Rocky", "Clay", "Mixed"]
        self.materials = ["PVC", "HDPE", "DI", "GI", "CI"]
        self.zones = list(ZONE_CHARACTERISTICS.keys())

    def correlated_features(self, pressure, pipe_age, flow_rate):
        """
        Generates correlated sensor features based on hydraulic parameters.
        """
        # RPM correlates with flow rate and pressure
        rpm = (flow_rate / 1000) * np.random.uniform(1.8, 2.5) + (pressure / 20) * 50

        # Vibration increases with pipe age, RPM, and flow
        vibration = (pipe_age * 0.04) + (rpm / 1200) + (flow_rate / 2000) + np.random.uniform(0, 0.5)

        # Acoustic correlates with vibration and flow turbulence
        acoustic = vibration * np.random.uniform(8, 14) + (flow_rate / 1000)

        # Ultrasonic increases for turbulence/cavitation (high flow in small pipes)
        ultrasonic_base = vibration / 10 + (flow_rate / 5000)

        return rpm, vibration, acoustic, ultrasonic_base

    def generate_reading(self, zone_name, timestamp, master_flow_lpm, zones_demand, leak_factor=1.0):
        """
        Generate a sensor reading for a zone at a specific timestamp.
        Incorporates hydraulic physics, demand model, and leak effects.
        """
        hour = timestamp.hour
        month = timestamp.month
        
        # Get zone characteristics
        zone_char = ZONE_CHARACTERISTICS[zone_name]
        
        # Compute demand for this zone
        zone_demand = compute_zone_demand(
            zone_char["base_demand_lpm"],
            hour,
            month,
            zone_name,
            leak_factor=leak_factor
        )
        
        # Get flow distribution
        flow_dist = compute_flow_distribution(master_flow_lpm, zones_demand, sum(zones_demand.values()))
        zone_flow = flow_dist.get(zone_name, 0.0)
        
        # Get branch specs and elevations
        branch_key = ("Master", zone_name)
        branch_spec = BRANCH_SPECS.get(branch_key, {"diameter_mm": 250, "length_m": 1500, "C": 120})
        master_elev = DEFAULT_ELEVATIONS["Master"]
        zone_elev = DEFAULT_ELEVATIONS.get(zone_name, 44.0) + np.random.uniform(-1, 1)
        
        # Compute zone pressure using booster pump model
        zone_pressure, loss_info = compute_zone_pressure_with_booster(
            PUMP_SETPOINT_PSI,
            master_elev,
            zone_elev,
            branch_spec,
            zone_flow,
            valve_open=np.random.uniform(0.85, 1.0),  # Normal valve opening with slight variation
        )
        
        # Compute correlated features
        pipe_age = np.random.randint(5, 40)
        rpm, vibration, acoustic, ultrasonic = self.correlated_features(zone_pressure, pipe_age, zone_flow)
        
        # Seasonal temperature
        temp_base = {
            12: np.random.uniform(18, 24), 1: np.random.uniform(18, 24),
            2: np.random.uniform(20, 26), 3: np.random.uniform(22, 32),
            4: np.random.uniform(25, 34), 5: np.random.uniform(26, 36),
            6: np.random.uniform(30, 40), 7: np.random.uniform(30, 40),
            8: np.random.uniform(28, 36), 9: np.random.uniform(24, 32),
            10: np.random.uniform(22, 30), 11: np.random.uniform(20, 28),
        }.get(month, 28)
        
        # Determine anomaly class based on pressure/flow
        if zone_pressure < 35 or zone_flow > zone_demand * 1.8:
            class_label = "Leak"
        elif zone_pressure > 140 or vibration > 8:
            class_label = "Defect"
        elif zone_flow < zone_demand * 0.4:
            class_label = "IllegalConnection"
        elif pipe_age > 35 and vibration > 6:
            class_label = "MaintenanceRequired"
        else:
            class_label = "Normal"

        record = {
            "ZoneName": zone_name,
            "ZoneType": zone_char["type"],
            "Timestamp": timestamp.isoformat(),
            "Hour": hour,
            "Month": month,
            # Elevation & location data
            "Master_Elevation_m": {"value": master_elev, "unit": "meters"},
            "Zone_Elevation_m": {"value": zone_elev, "unit": "meters"},
            "Elevation_Diff_m": {"value": loss_info["elevation_diff_m"], "unit": "meters"},
            # Pipe specifications
            "Branch_Diameter_mm": {"value": branch_spec["diameter_mm"], "unit": "millimeters"},
            "Branch_Length_m": {"value": branch_spec["length_m"], "unit": "meters"},
            # Pump system
            "Pump_Setpoint_PSI": {"value": PUMP_SETPOINT_PSI, "unit": "PSI"},
            "Pump_Workload_PSI": {"value": loss_info["pump_workload_psi"], "unit": "PSI"},
            # Flow & Pressure
            "Master_Flow_LPM": {"value": round(master_flow_lpm, 2), "unit": "L/min"},
            "Zone_Demand_LPM": {"value": round(zone_demand, 2), "unit": "L/min"},
            "Actual_Flow_LPM": {"value": round(zone_flow, 2), "unit": "L/min"},
            "Pressure_PSI": {"value": round(zone_pressure, 2), "unit": "PSI"},
            "Friction_Loss_PSI": {"value": round(loss_info["friction_psi"], 2), "unit": "PSI"},
            # Sensor measurements
            "Temperature_C": {"value": round(temp_base + np.random.normal(0, 1), 2), "unit": "Celsius"},
            "Vibration": {"value": round(vibration, 2), "unit": "mm/s"},
            "RPM": {"value": round(rpm, 2), "unit": "revolutions/minute"},
            "OperationHours": {"value": round(np.random.uniform(1000, 50000), 2), "unit": "hours"},
            "AcousticLevel": {"value": round(acoustic, 2), "unit": "dB"},
            "UltrasonicSignal": {"value": round(ultrasonic, 2), "unit": "V"},
            "PipeAge": {"value": pipe_age, "unit": "years"},
            "SoilType": np.random.choice(self.soil_types),
            "Material": np.random.choice(self.materials),
            # Labels
            "class_label": class_label,
        }

        return record

    def generate_batch(self, name, num_samples, days=365):
        """
        Generate a batch of training data.
        """
        records = []
        start_date = datetime.now() - timedelta(days=days)
        
        # Time resolution: hourly samples
        hours_total = days * 24
        sample_interval = max(1, hours_total // num_samples)

        for i in range(0, hours_total, sample_interval):
            timestamp = start_date + timedelta(hours=i)
            
            # Simulate master source flow (varies with time/demand)
            hour = timestamp.hour
            month = timestamp.month
            hour_factor = hourly_demand_factor(hour)
            seasonal_factor = seasonal_demand_multiplier(month)
            master_flow_lpm = 6000 * hour_factor * seasonal_factor
            
            # Compute demands for all zones
            zones_demand = {}
            for zone in self.zones:
                zones_demand[zone] = compute_zone_demand(
                    ZONE_CHARACTERISTICS[zone]["base_demand_lpm"],
                    hour,
                    month,
                    zone,
                    leak_factor=1.0
                )
            
            # Simulate occasional leaks (5% chance per zone)
            for zone in self.zones:
                leak_factor = np.random.choice([1.0, 1.0, 1.0, 1.0, 1.3], p=[0.8, 0.05, 0.05, 0.05, 0.05])
                
                record = self.generate_reading(zone, timestamp, master_flow_lpm, zones_demand, leak_factor)
                records.append(record)
            
            if len(records) >= num_samples:
                break

        path = self.output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(records[:num_samples], f, indent=2)

        print(f"✓ Generated {name} ({len(records[:num_samples])} rows)")

    def generate_all(self):
        """Generate all training batches."""
        self.generate_batch("water_batch_01", 10000, days=120)
        self.generate_batch("water_batch_02", 12000, days=150)
        self.generate_batch("water_batch_03", 15000, days=180)
        print("\n✔ All training batches generated with hydraulic physics!\n")


# ----------------------------------------
# RUN
# ----------------------------------------
if __name__ == "__main__":
    gen = WaterDatasetGenerator()
    gen.generate_all()
