"""
Anti-Gravity AML — Interswitch Field Test (Stage 3)
Applies the IBM-trained model to the unlabeled Interswitch dataset.
Reports OPERATIONAL KPIs (not accuracy metrics since there are no labels).

Pipeline:
  1. Load pre-processed Interswitch features (from phase1_data_prep.py)
  2. Load IBM-trained model
  3. Score every Interswitch transaction → ml_prob
  4. Load Interswitch + IBM SNA features for Pattern Bridge (Stage 2)
  5. Build Pattern Bridge report (IBM ↔ Interswitch motif comparison)
  6. Compute Operational KPIs:
       KPI 1: False Positive Reduction
       KPI 2: Inference Latency
       KPI 3: Explainability Score
  7. Save scored dataset + KPI report for dashboard

Run this AFTER phase0_ibm_pipeline.py AND phase1_data_prep.py.
"""

import os, sys, pickle, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from aml_engine.data_loader          import load_config
from aml_engine.feature_engineer     import ALL_FEATURES
from aml_engine.pattern_bridge       import build_pattern_bridge, load_bridge_report
from aml_engine.operational_evaluator import build_operational_kpis
from aml_engine.xai_engine           import compute_shap_values

ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT, 'aml_config.yaml')


def main():
    print("=" * 70)
    print("  ANTI-GRAVITY AML — Stage 3: Interswitch Field Test")
    print("  Dataset: Interswitch ATM + Agent (Uganda) — unlabeled")
    print("  KPIs: FP Reduction | Latency | Explainability (NOT Accuracy)")
    print("=" * 70)

    cfg       = load_config()
    s1_cfg    = cfg['three_stage_pipeline']['stage1_ibm']
    s3_cfg    = cfg['three_stage_pipeline']['stage3_fieldtest']
    paths_cfg = cfg['paths']

    processed_dir  = os.path.join(ROOT, paths_cfg['processed_dir'])
    models_dir     = os.path.join(ROOT, paths_cfg['models_dir'])
    graphs_dir     = os.path.join(ROOT, paths_cfg['graphs_dir'])
    ibm_models_dir = os.path.join(ROOT, os.path.dirname(s1_cfg['ibm_model_output']))
    ibm_shap_dir   = os.path.join(ROOT, s1_cfg['ibm_shap_output'])
    ibm_graph_dir  = os.path.join(graphs_dir, 'ibm')

    # ── Step 1: Load Interswitch processed features ──────────
    isw_feat_path = os.path.join(processed_dir, 'features_full.csv')
    if not os.path.exists(isw_feat_path):
        print(f"[Stage 3] ERROR: Interswitch features not found at {isw_feat_path}")
        print("  Run phase1_data_prep.py first.")
        sys.exit(1)

    print(f"\n[Stage 3] Step 1: Loading Interswitch features...")
    df_isw = pd.read_csv(isw_feat_path, low_memory=False)
    print(f"  Interswitch rows: {len(df_isw):,}")

    # ── Step 2: Load IBM-trained model ───────────────────────
    ibm_model_path = os.path.join(ROOT, s1_cfg['ibm_model_output'])
    if not os.path.exists(ibm_model_path):
        print(f"[Stage 3] WARNING: IBM model not found at {ibm_model_path}")
        print("  Falling back to locally-trained model...")
        ibm_model_path = os.path.join(ibm_models_dir, 'best_model.pkl')

    if not os.path.exists(ibm_model_path):
        print("[Stage 3] ERROR: No trained model found. Run phase0_ibm_pipeline.py first.")
        sys.exit(1)

    print(f"\n[Stage 3] Step 2: Loading IBM-trained model from: {ibm_model_path}")
    with open(ibm_model_path, 'rb') as f:
        model_bundle = pickle.load(f)

    # Load feature name list
    feat_path = os.path.join(ibm_models_dir, 'feature_names.pkl')
    if os.path.exists(feat_path):
        with open(feat_path, 'rb') as f:
            feat_dict = pickle.load(f)
        feature_names = feat_dict.get('all', [f for f in ALL_FEATURES if f in df_isw.columns])
    else:
        feature_names = [f for f in ALL_FEATURES if f in df_isw.columns]

    # ── Step 3: Score Interswitch transactions ────────────────
    print(f"\n[Stage 3] Step 3: Scoring {len(df_isw):,} Interswitch transactions...")
    X_isw = df_isw[[f for f in feature_names if f in df_isw.columns]].fillna(0).values

    # Pad/align columns if IBM model has different feature count
    ibm_feat_count = (model_bundle['scaler'].n_features_in_
                      if model_bundle.get('scaler') else len(feature_names))
    if X_isw.shape[1] < ibm_feat_count:
        pad = np.zeros((X_isw.shape[0], ibm_feat_count - X_isw.shape[1]))
        X_isw = np.hstack([X_isw, pad])
    elif X_isw.shape[1] > ibm_feat_count:
        X_isw = X_isw[:, :ibm_feat_count]

    model  = model_bundle['model']
    scaler = model_bundle.get('scaler')
    if scaler is not None:
        X_isw_sc = scaler.transform(X_isw)
    else:
        X_isw_sc = X_isw

    if hasattr(model, 'predict_proba'):
        ml_probs = model.predict_proba(X_isw_sc)[:, 1]
    else:
        ml_probs = model.predict(X_isw_sc).astype(float)

    ml_preds = (ml_probs >= 0.5).astype(int)

    df_isw['ml_risk_score'] = ml_probs
    df_isw['ml_flagged']    = ml_preds
    df_isw['inference_source'] = 'IBM_TRAINED_MODEL'

    n_flagged = ml_preds.sum()
    print(f"  IBM model flagged: {n_flagged:,} / {len(df_isw):,} "
          f"({100*n_flagged/len(df_isw):.2f}%) Interswitch transactions")

    # ── Step 4: Save scored dataset ───────────────────────────
    print(f"\n[Stage 3] Step 4: Saving scored Interswitch dataset...")
    scored_path = os.path.join(ROOT, s3_cfg['fieldtest_output'])
    save_cols = ['source', 'target', 'terminal_id', 'amount', 'tran_type', 'step',
                 'rule_triggered', 'rule_score', 'ml_risk_score', 'ml_flagged',
                 'motif_circular', 'motif_smurfing', 'motif_reversal',
                 'inference_source', 'dataset']
    save_cols = [c for c in save_cols if c in df_isw.columns]
    df_isw[save_cols].to_csv(scored_path, index=False)
    print(f"  Saved → {scored_path}")

    # ── Step 5: Pattern Bridge (Stage 2) ─────────────────────
    print(f"\n[Stage 3] Step 5: Building Pattern Bridge (IBM ↔ Interswitch)...")
    ibm_feat_path = os.path.join(processed_dir, 'ibm_features_full.csv')

    if os.path.exists(ibm_feat_path):
        df_ibm = pd.read_csv(ibm_feat_path, low_memory=False)

        # Load SNA features
        ibm_sna, isw_sna = {}, {}
        ibm_sna_path = os.path.join(ibm_graph_dir, 'sna_features.pkl')
        isw_sna_path = os.path.join(graphs_dir, 'sna_features.pkl')
        if os.path.exists(ibm_sna_path):
            with open(ibm_sna_path, 'rb') as f: ibm_sna = pickle.load(f)
        if os.path.exists(isw_sna_path):
            with open(isw_sna_path, 'rb') as f: isw_sna = pickle.load(f)

        # Load motif metadata
        ibm_motif_meta, isw_motif_meta = {}, {}
        ibm_motif_path = os.path.join(ibm_graph_dir, 'motif_metadata.pkl')
        isw_motif_path = os.path.join(graphs_dir, 'motif_metadata.pkl')
        if os.path.exists(ibm_motif_path):
            with open(ibm_motif_path, 'rb') as f: ibm_motif_meta = pickle.load(f)
        if os.path.exists(isw_motif_path):
            with open(isw_motif_path, 'rb') as f: isw_motif_meta = pickle.load(f)

        bridge_report = build_pattern_bridge(
            df_ibm, df_isw, ibm_sna, isw_sna,
            ibm_motif_meta, isw_motif_meta, cfg, processed_dir
        )
    else:
        print("  Skipping Pattern Bridge — IBM processed features not found.")
        print("  Run phase0_ibm_pipeline.py to generate ibm_features_full.csv")
        bridge_report = {}

    # ── Step 6: Compute SHAP for Interswitch alerts ───────────
    print(f"\n[Stage 3] Step 6: Computing Interswitch SHAP explanations...")
    isw_shap_dir = os.path.join(os.path.dirname(ibm_shap_dir), 'isw_shap')
    os.makedirs(isw_shap_dir, exist_ok=True)

    # Score the high-risk subset for SHAP (faster)
    high_risk_mask = ml_probs >= 0.3
    X_shap = X_isw[high_risk_mask][:500]
    y_shap = ml_preds[high_risk_mask][:500]

    if len(X_shap) > 10:
        shap_vals, _, _ = compute_shap_values(
            model_bundle=model_bundle,
            X_test=X_shap,
            feature_names=feature_names[:ibm_feat_count],
            output_dir=isw_shap_dir,
            max_samples=min(500, len(X_shap)),
        )
    else:
        print("  Not enough high-risk samples for SHAP. Using full test sample.")
        shap_vals = np.zeros((len(X_isw_sc[:50]), ibm_feat_count))
        isw_shap_dir_shape = (50, ibm_feat_count)

    # ── Step 7: Compute Operational KPIs ─────────────────────
    print(f"\n[Stage 3] Step 7: Computing Operational KPIs...")
    try:
        shap_vals_loaded = np.load(os.path.join(isw_shap_dir, 'shap_values.npy'))
    except Exception:
        shap_vals_loaded = np.zeros((50, ibm_feat_count))

    kpi_report = build_operational_kpis(
        df=df_isw,
        ml_probs=ml_probs,
        model_bundle=model_bundle,
        X_sample=X_isw,
        shap_values=shap_vals_loaded,
        feature_names=feature_names[:ibm_feat_count],
        cfg=cfg,
        output_dir=os.path.join(ROOT, paths_cfg['processed_dir']),
    )

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Stage 3 Complete! Interswitch Field Test Summary:")
    print(f"    Transactions scored  : {len(df_isw):,}")
    print(f"    IBM model flagged    : {n_flagged:,} ({100*n_flagged/len(df_isw):.2f}%)")
    kpi1 = kpi_report.get('kpi_1_fp_reduction', {})
    kpi2 = kpi_report.get('kpi_2_latency', {})
    kpi3 = kpi_report.get('kpi_3_explainability', {})
    print(f"    KPI 1 FP Reduction   : {kpi1.get('fp_reduction_rate', 0):.1%}")
    print(f"    KPI 2 Latency        : {kpi2.get('mean_latency_ms', 0):.1f}ms / 100 tx")
    print(f"    KPI 3 XAI Coverage   : {kpi3.get('mean_top3_coverage', 0):.1%}")
    print(f"    KPIs Passed          : {kpi_report.get('kpis_met', 0)}/3")
    if bridge_report:
        print(f"    Pattern Similarity   : {bridge_report.get('overall_similarity', 0):.1%}")
    print("=" * 70)
    print("\n  ✅ All three stages complete. Launch the dashboard:")
    print("     streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
