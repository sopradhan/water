"""
Create zone-specific training datasets from existing training data.
Zone0, Zone1, Zone2 with class balance and 50+ record increase per class.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_training_data():
    """Load all training data batches."""
    training_dir = Path('src/data/training_dataset')
    all_data = []
    
    for batch_file in sorted(training_dir.glob('water_batch_*.json')):
        with open(batch_file) as f:
            batch_data = json.load(f)
            all_data.extend(batch_data)
    
    return all_data

def load_production_data():
    """Load production data to understand zone characteristics."""
    with open('src/data/prod_data/prod_zone0_master.json') as f:
        prod_data = json.load(f)
    return prod_data

def get_class_distribution(data):
    """Get distribution of classes in data."""
    classes = [record.get('class_label', 'Unknown') for record in data]
    return Counter(classes)

def create_zone_variations(record, zone_name, zone_type):
    """Create zone-specific variations of a record."""
    record_copy = record.copy()
    record_copy['ZoneName'] = zone_name
    
    # Assign zone type based on zone
    if zone_name == 'Zone0':
        record_copy['ZoneType'] = 'distribution_hub'
    elif zone_name == 'Zone1':
        record_copy['ZoneType'] = 'primary_distribution'
    else:  # Zone2
        record_copy['ZoneType'] = 'secondary_distribution'
    
    # Add slight variations to numeric features to simulate zone-specific patterns
    # Zone1 typically has higher pressure, Zone2 has lower flow
    
    def get_value(field_value):
        """Extract numeric value from field (handles both direct values and dict format)."""
        if isinstance(field_value, dict) and 'value' in field_value:
            return field_value['value']
        return field_value
    
    if zone_name == 'Zone1':
        # Higher pressure zone
        if 'Pressure_PSI' in record_copy:
            val = get_value(record_copy['Pressure_PSI'])
            record_copy['Pressure_PSI'] = val * 1.05 if isinstance(val, (int, float)) else val
        if 'Master_Flow_LPM' in record_copy:
            val = get_value(record_copy['Master_Flow_LPM'])
            record_copy['Master_Flow_LPM'] = val * 0.95 if isinstance(val, (int, float)) else val
    elif zone_name == 'Zone2':
        # Lower flow zone
        if 'Master_Flow_LPM' in record_copy:
            val = get_value(record_copy['Master_Flow_LPM'])
            record_copy['Master_Flow_LPM'] = val * 0.90 if isinstance(val, (int, float)) else val
        if 'Pressure_PSI' in record_copy:
            val = get_value(record_copy['Pressure_PSI'])
            record_copy['Pressure_PSI'] = val * 0.98 if isinstance(val, (int, float)) else val
    
    return record_copy

def create_augmented_records(original_records, class_label, target_count):
    """
    Create augmented records to reach target count for a class.
    Uses slight variations of existing records.
    """
    augmented = []
    original_count = len(original_records)
    needed = target_count - original_count
    
    if needed <= 0:
        return original_records[:target_count]
    
    # Add all original records
    augmented.extend(original_records)
    
    # Create augmented records with noise
    for i in range(needed):
        template = original_records[i % len(original_records)].copy()
        
        # Add small random noise to numeric features (±2-5%)
        numeric_fields = [
            'Pressure_PSI', 'Master_Flow_LPM', 'Temperature_C', 'Vibration',
            'RPM', 'OperationHours', 'AcousticLevel', 'UltrasonicSignal', 'PipeAge'
        ]
        
        for field in numeric_fields:
            if field in template:
                value = template[field]
                # Handle dict values (e.g., {'value': 123.45, 'unit': 'PSI'})
                if isinstance(value, dict) and 'value' in value:
                    value = value['value']
                
                if isinstance(value, (int, float)):
                    noise = np.random.normal(0, 0.03)  # 3% noise
                    template[field] = value * (1 + noise)
                    # Ensure non-negative values
                    template[field] = max(0, template[field])
        
        augmented.append(template)
    
    return augmented

def create_zone_datasets():
    """Create three zone-specific datasets."""
    print("Loading training data...")
    training_data = load_training_data()
    prod_data = load_production_data()
    
    print(f"Total training records: {len(training_data)}")
    print(f"Class distribution: {get_class_distribution(training_data)}")
    
    # Group training data by class
    class_groups = {}
    for record in training_data:
        class_label = record.get('class_label', 'Unknown')
        if class_label not in class_groups:
            class_groups[class_label] = []
        class_groups[class_label].append(record)
    
    print(f"\nClasses found: {list(class_groups.keys())}")
    
    # Calculate target count per class (50+ increase means ~50-80 extra records per class)
    # If a class has 100 records, target becomes 150-180
    target_per_class = {}
    for class_label, records in class_groups.items():
        original_count = len(records)
        # 50-100 record increase per class
        increase = random.randint(50, 100)
        target_per_class[class_label] = original_count + increase
        print(f"{class_label}: {original_count} -> {target_per_class[class_label]} "
              f"(+{increase} records)")
    
    # Create augmented balanced dataset
    print("\nCreating augmented balanced training data...")
    augmented_balanced = []
    for class_label, target_count in target_per_class.items():
        class_records = class_groups[class_label]
        augmented_class_records = create_augmented_records(class_records, class_label, target_count)
        augmented_balanced.extend(augmented_class_records)
    
    # Shuffle the augmented balanced dataset
    random.shuffle(augmented_balanced)
    
    print(f"Total augmented balanced records: {len(augmented_balanced)}")
    print(f"New class distribution: {get_class_distribution(augmented_balanced)}")
    
    # Create zone-specific datasets
    zones = [
        ('Zone0', 'distribution_hub'),
        ('Zone1', 'primary_distribution'),
        ('Zone2', 'secondary_distribution')
    ]
    
    zone_datasets = {}
    
    for zone_name, zone_type in zones:
        print(f"\nCreating {zone_name} dataset...")
        zone_data = []
        
        for record in augmented_balanced:
            zone_record = create_zone_variations(record, zone_name, zone_type)
            zone_data.append(zone_record)
        
        zone_datasets[zone_name] = zone_data
        
        # Save zone-specific dataset
        output_path = Path(f'src/data/training_dataset/{zone_name}_training_data.json')
        with open(output_path, 'w') as f:
            json.dump(zone_data, f, indent=2)
        
        print(f"  Saved {len(zone_data)} records to {output_path}")
        print(f"  Class distribution: {get_class_distribution(zone_data)}")
        
        # Also create a combined file with all zones
        output_path_combined = Path(f'src/data/training_dataset/{zone_name}_balanced_training.json')
        with open(output_path_combined, 'w') as f:
            json.dump(zone_data, f, indent=2)
    
    # Create a master combined dataset with all zones
    print("\nCreating combined master dataset with all zones...")
    master_data = []
    for zone_name, zone_type in zones:
        master_data.extend(zone_datasets[zone_name])
    
    master_path = Path('src/data/training_dataset/master_balanced_training.json')
    with open(master_path, 'w') as f:
        json.dump(master_data, f, indent=2)
    
    print(f"Master dataset saved: {len(master_data)} records")
    print(f"Master class distribution: {get_class_distribution(master_data)}")
    
    return zone_datasets, master_data

if __name__ == '__main__':
    zone_datasets, master_data = create_zone_datasets()
    
    print("\n" + "="*60)
    print("ZONE-SPECIFIC DATASETS CREATED SUCCESSFULLY")
    print("="*60)
    print(f"\nFiles created:")
    print("  - src/data/training_dataset/Zone0_training_data.json")
    print("  - src/data/training_dataset/Zone1_training_data.json")
    print("  - src/data/training_dataset/Zone2_training_data.json")
    print("  - src/data/training_dataset/master_balanced_training.json")
