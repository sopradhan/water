"""
Water Network Production Data Generator - HYDRAULIC PHYSICS VERSION
-------------------------------------------------------------------
Real-time inference data with:
- Booster pump system (pump-driven pressure control)
- Elevation-based system load
- Hazen-Williams friction losses
- Zone-specific demand model
- Valve interconnections
- Realistic anomaly injection
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

DEFAULT_ELEVATIONS = {
    "Master": 48.0,
    "Zone0": 44.0,
    "Zone1": 58.0,
    "Zone2": 33.0,
    "Zone3": 22.0,
    "Zone4": 5.0,
}

BRANCH_SPECS = {
    ("Master", "Zone0"): {"diameter_mm": 350, "length_m": 600, "C": 120},
    ("Master", "Zone1"): {"diameter_mm": 300, "length_m": 1200, "C": 120},
    ("Master", "Zone2"): {"diameter_mm": 250, "length_m": 1800, "C": 110},
    ("Master", "Zone3"): {"diameter_mm": 200, "length_m": 3000, "C": 105},
}

ZONE_CHARACTERISTICS = {
    "Zone0": {
        "name": "Master Distribution",
        "type": "distribution_hub",
        "lat": 28.7041,
        "lon": 77.1025,
        "base_demand_lpm": 800,
        "interconnected": ["Zone1", "Zone2", "Zone3"],
    },
    "Zone1": {
        "name": "Residential Zone 1",
        "type": "residential",
        "lat": 28.6139,
        "lon": 77.2090,
        "base_demand_lpm": 1200,
        "interconnected": ["Zone0", "Zone3"],
    },
    "Zone2": {
        "name": "Commercial Zone 2",
        "type": "commercial",
        "lat": 28.5244,
        "lon": 77.1855,
        "base_demand_lpm": 1500,
        "interconnected": ["Zone0", "Zone3"],
    },
    "Zone3": {
        "name": "Industrial Zone 3",
        "type": "industrial",
        "lat": 28.4595,
        "lon": 77.0266,
        "base_demand_lpm": 2000,
        "interconnected": ["Zone1", "Zone2"],
    },
}

PUMP_SETPOINT_PSI = 120.0


# ========================================
# HYDRAULIC CALCULATIONS (Same as training)
# ========================================

def head_m_from_hazen_williams(L_m, Q_m3s, D_m, C=120):
    """Hazen-Williams head loss calculation."""
    if Q_m3s <= 0 or D_m <= 0:
        return 0.0
    try:
        hf = 10.67 * L_m * (Q_m3s ** 1.852) / ((C ** 1.852) * (D_m ** 4.87))
        return max(0.0, hf)
    except:
        return 0.0


def compute_zone_pressure_with_booster(pump_setpoint_psi, master_elev_m, zone_elev_m,
                                       branch_spec, branch_flow_lpm, valve_open=1.0):
    """Compute zone pressure using booster pump model."""
    D_m = branch_spec["diameter_mm"] / 1000.0
    L_m = branch_spec["length_m"]
    C = branch_spec["C"]
    
    Q_lpm = max(0.0, branch_flow_lpm * valve_open)
    Q_m3s = (Q_lpm / 60.0) / 1000.0
    
    hf_m = head_m_from_hazen_williams(L_m, Q_m3s, D_m, C)
    hf_psi = hf_m * 0.433
    
    elev_diff_m = zone_elev_m - master_elev_m
    pump_workload_psi = abs(elev_diff_m * 0.433)
    
    P_zone = pump_setpoint_psi - hf_psi
    P_zone += np.random.normal(0, 0.15)
    P_zone = np.clip(P_zone, 95.0, 140.0)
    
    return float(P_zone), {"friction_psi": hf_psi, "pump_workload_psi": pump_workload_psi}


def hourly_demand_factor(hour):
    """24-hour demand curve."""
    morning_peak = np.exp(-((hour - 7) ** 2) / (2 * 2.5 ** 2))
    evening_peak = np.exp(-((hour - 19) ** 2) / (2 * 2.5 ** 2))
    base = 0.4
    return base + 0.8 * (morning_peak + 0.7 * evening_peak)


def seasonal_demand_multiplier(month):
    """Seasonal variation."""
    if month in [6, 7, 8, 9]:
        return 0.85
    elif month in [3, 4, 5]:
        return 1.2
    else:
        return 1.0


# ========================================
# PRODUCTION DATA GENERATOR
# ========================================

class ProductionDataGenerator:
    """Generate real-time production/inference data."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else config.PROD_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.soil_types = ["Sandy", "Rocky", "Clay", "Mixed"]
        self.materials = ["PVC", "HDPE", "DI", "GI", "CI"]
        self.zones = list(ZONE_CHARACTERISTICS.keys())

    def correlated_features(self, pressure, pipe_age, flow_rate):
        """Generate correlated sensor features."""
        rpm = (flow_rate / 1000) * np.random.uniform(1.8, 2.5) + (pressure / 20) * 50
        vibration = (pipe_age * 0.04) + (rpm / 1200) + (flow_rate / 2000) + np.random.uniform(0, 0.5)
        acoustic = vibration * np.random.uniform(8, 14) + (flow_rate / 1000)
        ultrasonic = vibration / 10 + (flow_rate / 5000)
        return rpm, vibration, acoustic, ultrasonic

    def generate_reading(self, zone_name, timestamp, master_flow_lpm, is_anomalous=False, anomaly_type=None):
        """Generate a sensor reading for a zone."""
        hour = timestamp.hour
        month = timestamp.month
        
        zone_char = ZONE_CHARACTERISTICS[zone_name]
        
        # Compute demand
        hour_factor = hourly_demand_factor(hour)
        seasonal_factor = seasonal_demand_multiplier(month)
        zone_demand = zone_char["base_demand_lpm"] * hour_factor * seasonal_factor
        
        # Flow distribution
        all_demands = {z: ZONE_CHARACTERISTICS[z]["base_demand_lpm"] * hour_factor * seasonal_factor 
                       for z in self.zones}
        total_demand = sum(all_demands.values())
        zone_flow = master_flow_lpm * (zone_demand / max(total_demand, 1))
        
        # Get branch specs
        branch_key = ("Master", zone_name)
        branch_spec = BRANCH_SPECS.get(branch_key, {"diameter_mm": 250, "length_m": 1500, "C": 120})
        master_elev = DEFAULT_ELEVATIONS["Master"]
        zone_elev = DEFAULT_ELEVATIONS.get(zone_name, 44.0) + np.random.uniform(-1, 1)
        
        # Compute pressure
        zone_pressure, loss_info = compute_zone_pressure_with_booster(
            PUMP_SETPOINT_PSI,
            master_elev,
            zone_elev,
            branch_spec,
            zone_flow,
            valve_open=np.random.uniform(0.85, 1.0),
        )
        
        # Apply anomalies
        if is_anomalous and anomaly_type:
            if anomaly_type == "leak":
                zone_pressure *= np.random.uniform(0.6, 0.9)
                zone_flow *= np.random.uniform(1.3, 1.8)
            elif anomaly_type == "defect":
                zone_pressure = np.clip(zone_pressure + np.random.uniform(-5, 15), 90, 145)
                zone_flow *= np.random.uniform(0.7, 1.1)
            elif anomaly_type == "illegal":
                zone_flow *= np.random.uniform(0.4, 0.7)
                zone_pressure *= 1.05
        
        pipe_age = np.random.randint(5, 40)
        rpm, vibration, acoustic, ultrasonic = self.correlated_features(zone_pressure, pipe_age, zone_flow)
        
        # Temperature
        temp_base = {
            12: np.random.uniform(18, 24), 1: np.random.uniform(18, 24),
            2: np.random.uniform(20, 26), 3: np.random.uniform(22, 32),
            4: np.random.uniform(25, 34), 5: np.random.uniform(26, 36),
            6: np.random.uniform(30, 40), 7: np.random.uniform(30, 40),
            8: np.random.uniform(28, 36), 9: np.random.uniform(24, 32),
            10: np.random.uniform(22, 30), 11: np.random.uniform(20, 28),
        }.get(month, 28)

        record = {
            "ZoneName": zone_name,
            "ZoneType": zone_char["type"],
            "Latitude": {"value": round(zone_char["lat"] + np.random.uniform(-0.005, 0.005), 6), "unit": "degrees"},
            "Longitude": {"value": round(zone_char["lon"] + np.random.uniform(-0.005, 0.005), 6), "unit": "degrees"},
            "InterconnectedZones": zone_char["interconnected"],
            "Timestamp": timestamp.isoformat(),
            "Hour": hour,
            "Month": month,
            # Hydraulic parameters
            "Master_Elevation_m": {"value": master_elev, "unit": "meters"},
            "Zone_Elevation_m": {"value": zone_elev, "unit": "meters"},
            "Branch_Diameter_mm": {"value": branch_spec["diameter_mm"], "unit": "millimeters"},
            "Branch_Length_m": {"value": branch_spec["length_m"], "unit": "meters"},
            "Pump_Setpoint_PSI": {"value": PUMP_SETPOINT_PSI, "unit": "PSI"},
            # Flow and Demand
            "Master_Flow_LPM": {"value": round(master_flow_lpm, 2), "unit": "L/min"},
            "Zone_Demand_LPM": {"value": round(zone_demand, 2), "unit": "L/min"},
            "Actual_Flow_LPM": {"value": round(zone_flow, 2), "unit": "L/min"},
            "Pressure_PSI": {"value": round(zone_pressure, 2), "unit": "PSI"},
            "Friction_Loss_PSI": {"value": round(loss_info["friction_psi"], 2), "unit": "PSI"},
            # Sensors
            "Temperature_C": {"value": round(temp_base + np.random.normal(0, 1), 2), "unit": "Celsius"},
            "Vibration": {"value": round(vibration, 2), "unit": "mm/s"},
            "RPM": {"value": round(rpm, 2), "unit": "revolutions/minute"},
            "OperationHours": {"value": round(np.random.uniform(1000, 50000), 2), "unit": "hours"},
            "AcousticLevel": {"value": round(acoustic, 2), "unit": "dB"},
            "UltrasonicSignal": {"value": round(ultrasonic, 2), "unit": "V"},
            "PipeAge": {"value": pipe_age, "unit": "years"},
            "SoilType": np.random.choice(self.soil_types),
            "Material": np.random.choice(self.materials),
        }

        return record

    def generate_zone_data(self, zone_name, name, num_samples, normal_ratio=0.85):
        """Generate data for a specific zone."""
        records = []
        start_date = datetime.now() - timedelta(days=7)
        
        hours_total = 7 * 24
        sample_interval = max(1, hours_total // num_samples)

        for i in range(0, hours_total, sample_interval):
            timestamp = start_date + timedelta(hours=i)
            
            hour = timestamp.hour
            month = timestamp.month
            hour_factor = hourly_demand_factor(hour)
            seasonal_factor = seasonal_demand_multiplier(month)
            master_flow_lpm = 6000 * hour_factor * seasonal_factor
            
            is_anomalous = np.random.rand() > normal_ratio
            anomaly_type = None
            if is_anomalous:
                anomaly_type = np.random.choice(["leak", "defect", "illegal"])
            
            rec = self.generate_reading(zone_name, timestamp, master_flow_lpm, is_anomalous, anomaly_type)
            records.append(rec)
            
            if len(records) >= num_samples:
                break

        path = self.output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(records[:num_samples], f, indent=2)

        print(f"✓ Generated {name} ({len(records[:num_samples])} rows)")

    def generate_all(self):
        """Generate all production zone data."""
        self.generate_zone_data("Zone0", "prod_zone0_master", 5000, normal_ratio=0.95)
        self.generate_zone_data("Zone1", "prod_zone1_residential", 6000, normal_ratio=0.85)
        self.generate_zone_data("Zone2", "prod_zone2_commercial", 6000, normal_ratio=0.80)
        self.generate_zone_data("Zone3", "prod_zone3_industrial", 6000, normal_ratio=0.90)
        print("\n✔ All production data batches generated with hydraulic physics!\n")


# ----------------------------------------
# RUN
# ----------------------------------------
if __name__ == "__main__":
    gen = ProductionDataGenerator()
    gen.generate_all()
