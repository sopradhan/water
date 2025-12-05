"""
Water Network Synthetic Dataset Generator (CORRELATED VERSION)
--------------------------------------------------------------
Now features strong realistic correlations between:
Pressure, FlowRate, Vibration, RPM, Acoustic, Ultrasonic, PipeAge
Makes the dataset realistic for ML training.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class WaterDatasetGenerator:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.soil_types = ["Sandy", "Rocky", "Clay", "Mixed"]
        self.materials = ["PVC", "HDPE", "DI", "GI", "CI"]

    # ----------------------------------------
    # BASE CORRELATION ENGINE
    # ----------------------------------------
    def correlated_features(self, pressure, pipe_age):
        """Generates correlated features for all 5 classes"""

        # FlowRate inverse to pressure
        flow_rate = (15000 / (pressure + 20)) + np.random.uniform(-50, 50)

        # RPM positively correlated with FlowRate
        rpm = flow_rate * np.random.uniform(2.2, 3.5)

        # Vibration increases with pipe age and rpm
        vibration = (pipe_age * 0.05) + (rpm / 1500) + np.random.uniform(0, 1)

        # Acoustic increases with vibration
        acoustic = vibration * np.random.uniform(7, 12)

        # Ultrasonic increases for leak/defects
        ultrasonic_base = vibration / 12

        return flow_rate, rpm, vibration, acoustic, ultrasonic_base

    # ----------------------------------------
    # CLASS DEFINITIONS
    # ----------------------------------------
    
    def generate_normal(self, temp_base):
        pressure = np.random.uniform(65, 110)
        pipe_age = np.random.randint(1, 40)
        
        flow_rate, rpm, vibration, acoustic, us_base = self.correlated_features(pressure, pipe_age)

        return {
            "Pressure": pressure,
            "FlowRate": flow_rate,
            "Temperature": temp_base + np.random.normal(0, 1),
            "Vibration": vibration,
            "RPM": rpm,
            "OperationHours": np.random.uniform(1000, 30000),
            "AcousticLevel": acoustic,
            "UltrasonicSignal": us_base,
            "PipeAge": pipe_age,
            "SoilType": np.random.choice(self.soil_types),
            "Material": np.random.choice(self.materials),
            "class_label": "Normal"
        }

    def generate_leak(self, temp_base):
        pressure = np.random.uniform(20, 38)  # low
        pipe_age = np.random.randint(5, 60)

        flow_rate, rpm, vibration, acoustic, us_base = self.correlated_features(pressure, pipe_age)

        return {
            "Pressure": pressure,
            "FlowRate": flow_rate * np.random.uniform(1.4, 2.0),  # leak increases flow
            "Temperature": temp_base,
            "Vibration": vibration + np.random.uniform(2, 5),
            "RPM": rpm * np.random.uniform(1.1, 1.3),
            "OperationHours": np.random.uniform(500, 40000),
            "AcousticLevel": acoustic * np.random.uniform(1.5, 2.2),
            "UltrasonicSignal": us_base * np.random.uniform(2.0, 3.5),  # high ultrasonic
            "PipeAge": pipe_age,
            "SoilType": np.random.choice(self.soil_types),
            "Material": np.random.choice(self.materials),
            "class_label": "Leak"
        }

    def generate_illegal_connection(self, temp_base):
        pressure = np.random.uniform(45, 85) - np.random.uniform(5, 15)
        pipe_age = np.random.randint(10, 50)

        flow_rate, rpm, vibration, acoustic, us_base = self.correlated_features(pressure, pipe_age)

        return {
            "Pressure": pressure,
            "FlowRate": flow_rate * np.random.uniform(0.6, 1.1),
            "Temperature": temp_base,
            "Vibration": vibration + np.random.uniform(1, 3),
            "RPM": rpm,
            "OperationHours": np.random.uniform(500, 20000),
            "AcousticLevel": acoustic,
            "UltrasonicSignal": us_base * np.random.uniform(1.0, 1.8),
            "PipeAge": pipe_age,
            "SoilType": np.random.choice(self.soil_types),
            "Material": np.random.choice(self.materials),
            "class_label": "IllegalConnection"
        }

    def generate_defect(self, temp_base):
        pressure = np.random.choice([np.random.uniform(20, 60),
                                     np.random.uniform(100, 120)])
        pipe_age = np.random.randint(20, 80)

        flow_rate, rpm, vibration, acoustic, us_base = self.correlated_features(pressure, pipe_age)

        return {
            "Pressure": pressure,
            "FlowRate": flow_rate,
            "Temperature": temp_base,
            "Vibration": vibration + np.random.uniform(4, 6),
            "RPM": rpm * np.random.uniform(0.7, 1.4),
            "OperationHours": np.random.uniform(500, 30000),
            "AcousticLevel": acoustic * np.random.uniform(1.1, 1.6),
            "UltrasonicSignal": us_base * np.random.uniform(1.0, 2.0),
            "PipeAge": pipe_age,
            "SoilType": np.random.choice(self.soil_types),
            "Material": np.random.choice(self.materials),
            "class_label": "Defect"
        }

    def generate_maintenance(self, temp_base):
        pressure = np.random.uniform(40, 100)
        pipe_age = np.random.randint(50, 100)

        flow_rate, rpm, vibration, acoustic, us_base = self.correlated_features(pressure, pipe_age)

        return {
            "Pressure": pressure,
            "FlowRate": flow_rate * 0.8,
            "Temperature": temp_base,
            "Vibration": vibration + np.random.uniform(3, 7),
            "RPM": rpm,
            "OperationHours": np.random.uniform(30000, 60000),
            "AcousticLevel": acoustic,
            "UltrasonicSignal": us_base,
            "PipeAge": pipe_age,
            "SoilType": np.random.choice(self.soil_types),
            "Material": np.random.choice(self.materials),
            "class_label": "MaintenanceRequired"
        }

    # ----------------------------------------
    # BATCH GENERATOR
    # ----------------------------------------
    def generate_batch(self, name, num_samples):

        records = []
        start_date = datetime.now() - timedelta(days=365)

        for i in range(num_samples):
            timestamp = start_date + timedelta(minutes=i)
            month = timestamp.month

            # Seasonal temperature model
            temp_base = {
                12: np.random.uniform(18, 24),
                1: np.random.uniform(18, 24),
                2: np.random.uniform(20, 26),
                3: np.random.uniform(22, 32),
                4: np.random.uniform(25, 34),
                5: np.random.uniform(26, 36),
                6: np.random.uniform(30, 40),
                7: np.random.uniform(30, 40),
                8: np.random.uniform(28, 36),
                9: np.random.uniform(24, 32),
                10: np.random.uniform(22, 30),
                11: np.random.uniform(20, 28),
            }.get(month, 28)

            r = np.random.rand()
            if r < 0.60:
                rec = self.generate_normal(temp_base)
            elif r < 0.75:
                rec = self.generate_leak(temp_base)
            elif r < 0.85:
                rec = self.generate_illegal_connection(temp_base)
            elif r < 0.93:
                rec = self.generate_defect(temp_base)
            else:
                rec = self.generate_maintenance(temp_base)

            rec["Timestamp"] = timestamp.isoformat()
            records.append(rec)

        path = self.output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"✓ Generated {name} ({num_samples} rows)")

    def generate_all(self):
        self.generate_batch("water_batch_01", 10000)
        self.generate_batch("water_batch_02", 12000)
        self.generate_batch("water_batch_03", 15000)
        print("\n✔ All correlated batches generated!\n")


# ----------------------------------------
# RUN
# ----------------------------------------
if __name__ == "__main__":
    gen = WaterDatasetGenerator(
        r"C:\Users\GENAIKOLGPUSR36\Desktop\water\water_sensor_data\"
    )
    gen.generate_all()
