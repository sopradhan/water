CRITICAL DISCOVERY: CATEGORICAL FEATURE MISMATCH
================================================

Date: December 6, 2025

## KEY FINDINGS

### 1. Model Training Configuration
**Categorical Features Used:**
- SoilType (Label Encoded: 0, 1, 2, ...)
- Material (Label Encoded: 0, 1, 2, ...)

**Encoding Method:** Label Encoding (NOT One-Hot Encoding)

**Zone Information:** NOT included in training

### 2. Training Data Categorical Values
From analyzing training dataset:
- SoilType values: Various soil types (clay, sandy, silty, rocky, mixed)
- Material values: Various pipe materials (PVC, cast_iron, asbestos_cement, ductile_iron, HDPE, GI)

### 3. Production Data Categorical Values
- SoilType: Mixed, Clay, Sandy, Rocky (matches training data!)
- Material: DI, PVC, CI, GI, HDPE (different format - abbreviations instead of full names!)

**PROBLEM:** Production data uses material abbreviations:
- DI = Ductile Iron
- PVC = PVC
- CI = Cast Iron
- GI = Galvanized Iron n
- HDPE = HDPE

But training data likely used full names like "ductile_iron", "cast_iron", etc.

### 4. Zone Information
**Training Data:** Includes multiple zones (Zone0, Zone1, Zone2, etc.)
**Production Data:** Only Zone0 (distribution_hub)
**Models:** Do NOT use zone information - trained on data-driven patterns only

This is CORRECT - models should learn patterns independent of zone infrastructure.

## IMPACT ANALYSIS

### Why This Matters
1. **Label Encoder Mismatch**: When production sends "DI" as Material, the label encoder 
   may not recognize it if it was trained on "ductile_iron"
   
2. **Categorical Feature Handling**: The preprocess_sample() function in API tries to encode 
   unknown categories as 0 (fallback), which could cause prediction errors

3. **Test Case Design**: Test cases must use categorical values that the label encoders 
   actually learned during training

## SOLUTIONS

### Short-term (Immediate)
1. Check what exact Material values the label encoder knows
2. Create test cases using exact values from training data
3. Add data validation to API to warn about unknown categorical values

### Medium-term (Important)
1. Update production data pipeline to use full material names (not abbreviations)
2. Add mapping layer: DI → ductile_iron, CI → cast_iron, etc.
3. Document all accepted categorical values

### Long-term (Best Practice)
1. Use One-Hot Encoding instead of Label Encoding for categorical features
   - More robust to unknown values
   - Better for ensemble models
   
2. Add explicit categorical value validation
3. Create a data dictionary documenting all acceptable values

## NEXT STEPS

1. Inspect the label encoders to see exact values they recognize
2. Update test cases with correct categorical values
3. Create mapping for production data material abbreviations
4. Retrain models with proper error handling for unknown categories

## CODE SNIPPET: Check Label Encoders

```python
import joblib

label_encoders = joblib.load('src/model/model_weights/label_encoders.pkl')

for feature_name, encoder in label_encoders.items():
    print(f"{feature_name}: {encoder.classes_}")
```

This will show exactly what values each encoder knows about.
