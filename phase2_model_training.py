"""
Anti-Gravity AML — Phase 2: Model Training
Orchestrates: Load features → Train all tiers → Save models & comparison table
Run this AFTER phase1_data_prep.py completes.
"""

import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from aml_engine.data_loader   import load_config
from aml_engine.model_trainer import train_all_models
from aml_engine.feature_engineer import ALL_FEATURES


CONFIG_PATH = r'E:\MASTERSProject\AMLProject\aml_config.yaml'


def main():
    print("=" * 70)
    print("  ANTI-GRAVITY AML SYSTEM — Phase 2: Model Training")
    print("=" * 70)

    cfg = load_config()
    paths = cfg['paths']
    processed_dir = paths['processed_dir']
    models_dir    = paths['models_dir']
    os.makedirs(models_dir, exist_ok=True)

    # Load processed features
    feature_path = os.path.join(processed_dir, 'features_full.csv')
    if not os.path.exists(feature_path):
        print(f"[Phase 2] ERROR: Feature file not found at {feature_path}")
        print("[Phase 2] Please run phase1_data_prep.py first.")
        sys.exit(1)

    print(f"\n[Phase 2] Loading features from: {feature_path}")
    df = pd.read_csv(feature_path, low_memory=False)
    print(f"[Phase 2] Loaded: {len(df):,} rows | "
          f"Suspicious: {df['is_suspicious'].sum():,}")

    # Train all models
    print("\n[Phase 2] Starting multi-model training...")
    results_df = train_all_models(df, cfg, models_dir)

    # Print final summary
    print("\n" + "=" * 70)
    print("  Phase 2 Complete! Model Comparison Summary:")
    print("=" * 70)
    summary = results_df[['Model', 'Feature_Set', 'Tier', 'AUPRC', 'F1', 'ROC_AUC']].copy()
    summary = summary.sort_values('AUPRC', ascending=False)
    print(summary.to_string(index=False))
    print(f"\n[Phase 2] ✅ Models saved → {models_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
