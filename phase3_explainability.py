"""
XAI-SNA AML — Phase 3: Explainability (SHAP)
Orchestrates: Load best model → Compute SHAP values → Save charts & JSON
Run this AFTER phase2_model_training.py completes.
"""

import os, sys, pickle, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from aml_engine.data_loader import load_config
from aml_engine.xai_engine  import compute_shap_values


CONFIG_PATH = r'E:\MASTERSProject\AMLProject\aml_config.yaml'


def main():
    print("=" * 70)
    print("  XAI-SNA AML System — Phase 3: Explainability (XAI / SHAP)")
    print("=" * 70)

    cfg = load_config()
    paths = cfg['paths']
    models_dir = paths['models_dir']
    shap_dir   = paths['shap_dir']
    os.makedirs(shap_dir, exist_ok=True)

    # Load best model
    model_path = os.path.join(models_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        print(f"[Phase 3] ERROR: Best model not found at {model_path}")
        print("[Phase 3] Please run phase2_model_training.py first.")
        sys.exit(1)

    print(f"\n[Phase 3] Loading best model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)

    # Load feature names
    feat_path = os.path.join(models_dir, 'feature_names.pkl')
    with open(feat_path, 'rb') as f:
        feat_dict = pickle.load(f)
    feature_names = feat_dict['all']

    # Load test data
    X_test = np.load(os.path.join(models_dir, 'X_test_all.npy'))
    y_test = np.load(os.path.join(models_dir, 'y_test.npy'))
    print(f"[Phase 3] Test set: {len(y_test):,} samples | Suspicious: {y_test.sum():,}")

    # Compute SHAP values
    print("\n[Phase 3] Computing SHAP values (this may take a few minutes)...")
    shap_vals, ev, importance_df = compute_shap_values(
        model_bundle=model_bundle,
        X_test=X_test,
        feature_names=feature_names,
        output_dir=shap_dir,
        max_samples=500,
    )

    print("\n" + "=" * 70)
    print("  Phase 3 Complete! XAI Artifacts saved:")
    print(f"    SHAP values        → {shap_dir}/shap_values.npy")
    print(f"    Feature importance → {shap_dir}/feature_importance.csv")
    print(f"    Force plot data    → {shap_dir}/force_plot_data.json")
    print(f"    Waterfall chart    → {shap_dir}/shap_waterfall.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

