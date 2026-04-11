import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
PROCESSED_DIR = r'e:\MASTERSProject\AMLProject\processed_data'

def run_anomaly_detection():
    print("Loading data for Unsupervised Anomaly Detection...")
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
    
    # We use all features (Hybrid)
    features = X_train.columns.tolist()
    
    print(f"Initializing Isolation Forest on {len(features)} features...")
    # contamination is the expected proportion of outliers (frauds) in the data.
    # We estimate it from the training set if available, or use a heuristic.
    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    
    print("Fitting model (Unsupervised)...")
    iso.fit(X_train[features])
    
    print("Detecting anomalies in test set...")
    # -1 for outliers, 1 for inliers
    preds = iso.predict(X_test[features])
    scores = iso.decision_function(X_test[features])
    
    results = X_test.copy()
    results['anomaly_score'] = scores
    results['is_anomaly'] = [1 if p == -1 else 0 for p in preds]
    
    # Analyze overlap with ground truth labels
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).values.ravel()
    results['is_fraud'] = y_test
    
    overlap = results[(results['is_anomaly'] == 1) & (results['is_fraud'] == 1)]
    print(f"\nAnomaly Detection Results:")
    print(f"Total Anomalies Detected: {sum(results['is_anomaly'])}")
    print(f"Total Actual Frauds: {sum(results['is_fraud'])}")
    print(f"Overlap (Frauds caught by Unsupervised): {len(overlap)}")
    
    # Save results and model
    results_path = os.path.join(PROCESSED_DIR, 'anomaly_detection_results.csv')
    results.to_csv(results_path, index=False)
    
    model_path = os.path.join(PROCESSED_DIR, 'anomaly_detection_model.joblib')
    joblib.dump(iso, model_path)
    
    print(f"Detailed anomaly results saved to {results_path}")
    print(f"Unsupervised model saved to {model_path}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results, x='anomaly_score', hue='is_fraud', bins=50, kde=True)
    plt.title("Distribution of Isolation Forest Anomaly Scores")
    plt.xlabel("Anomaly Score (Lower = More Anomalous)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(PROCESSED_DIR, 'anomaly_score_distribution.png'))
    print("Anomaly score distribution plot saved.")

if __name__ == "__main__":
    run_anomaly_detection()
