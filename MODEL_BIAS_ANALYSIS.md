MODEL BIAS ANALYSIS REPORT
==========================

Date: December 6, 2025
Subject: Water Anomaly Detection Model - Prediction Bias Analysis

## FINDINGS

### 1. Training Data Class Distribution
- Normal: 26,007 samples (70.29%)
- Leak: 3,713 samples (10.04%)
- Defect: 2,851 samples (7.71%)
- IllegalConnection: 2,557 samples (6.91%)
- MaintenanceRequired: 1,872 samples (5.06%)
- TOTAL: 37,000 samples

### 2. Model Prediction Bias (Test Results)
Out of 10 test cases across 5 classes:
- Predicted as "Leak": 8/10 (80%)
- Predicted correctly: 2/10 (20%)

Results by class:
- Normal (2 tests): 0/2 correct (0%)
- Leak (2 tests): 2/2 correct (100%)
- Defect (2 tests): 1/2 correct (50%)
- MaintenanceRequired (2 tests): 0/2 correct (0%)
- IllegalConnection (2 tests): 0/2 correct (0%)

### 3. Root Causes

**Possible Issues:**
a) **Imbalanced Training Data**: The model was trained on heavily imbalanced data (70% Normal). 
   Models typically learn the majority class better.

b) **Feature Engineering for Test Cases**: The test cases may not represent the actual feature 
   distributions learned by the model during training.

c) **Class Weights**: If class weights were used during training to handle imbalance, the model 
   might still struggle to distinguish minority classes properly.

d) **Threshold Issues**: The decision thresholds or confidence scores might be poorly calibrated 
   for minority classes.

e) **Feature Scaling**: The scaler was fitted on training data with specific distributions. 
   Test case values outside these distributions might cause poor predictions.

## RECOMMENDATIONS

### Short-Term Fixes

1. **Use Real Production Data for Testing**
   - Instead of synthetic test cases, use actual examples from production data
   - This ensures test data has similar feature distributions to training data

2. **Analyze Model Confidence**
   - Check if the model is 100% confident in all "Leak" predictions
   - If confidence is high even for clearly "Normal" cases, there's a systematic bias

3. **Validate Feature Ranges**
   - Compare test case feature values to actual training data ranges
   - Ensure test cases are within realistic bounds

### Medium-Term Fixes

4. **Retrain with Balanced Data**
   - Use techniques like:
     - Undersampling majority class (Normal)
     - Oversampling minority classes
     - SMOTE (Synthetic Minority Over-sampling Technique)
     - Stratified sampling with class weights

5. **Tune Class Weights**
   - Adjust class_weight in KNN and LSTM to better balance predictions
   - Experiment with different weight ratios

6. **Use Calibration Methods**
   - Apply probability calibration (Platt scaling, isotonic regression)
   - Ensure confidence scores are reliable

### Long-Term Solutions

7. **Feature Engineering**
   - Analyze which features are most discriminative for each class
   - Create domain-specific features if needed

8. **Ensemble Methods**
   - Combine multiple models with different architectures
   - Use voting/averaging to reduce individual model bias

9. **Threshold Optimization**
   - Find optimal decision thresholds for each class
   - Use ROC curves and precision-recall analysis

10. **Production Data Collection**
    - Collect more diverse production examples
    - Ensure balanced representation of all anomaly types

## IMMEDIATE ACTION ITEMS

1. Test model with actual production data points (from prod_zone0_master.json)
2. Extract feature statistics from training data and compare with test cases
3. Review model training logs for any warnings about imbalanced classes
4. Check if class_weight was used during training in src/model/model.py
5. Analyze individual model behavior (KNN vs LSTM) to see which is biased

## CONCLUSION

The model exhibits significant prediction bias, predicting "Leak" for most test cases regardless 
of input features. This is likely due to a combination of:
- Imbalanced training data (70% Normal, 10% Leak, 20% others)
- Suboptimal class weight balancing
- Test cases that fall into feature regions dominated by "Leak" class during training

The model needs retraining with proper class balancing techniques and validation on real production data 
before it can be reliably deployed.
