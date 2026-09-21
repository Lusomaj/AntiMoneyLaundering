"""
XAI-SNA AML — Interswitch Field Test (Stage 3)
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

# Force UTF-8 stdout so Unicode chars (→ ✅ ⏳) don't crash on Windows cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from aml_engine.data_loader          import load_config
from aml_engine.feature_engineer     import ALL_FEATURES
from aml_engine.pattern_bridge       import build_pattern_bridge, load_bridge_report
from aml_engine.operational_evaluator import build_operational_kpis
from aml_engine.xai_engine           import compute_shap_values
from aml_engine.rules_engine         import apply_composite_rules
from aml_engine.anomaly_detector     import TopologyIsolationForest

ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT, 'aml_config.yaml')


def main():
    print("=" * 70)
    print("  XAI-SNA AML — Stage 3: Interswitch Field Test")
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

    # ── Domain Adaptation: percentile-clip to IBM training range ───
    # Without this, ISW features (UGX-scale, smaller graph) land outside
    # the IBM decision boundary and the model assigns near-zero probability
    # to every Interswitch transaction regardless of genuine risk level.
    ibm_stats_path = os.path.join(ibm_models_dir, 'ibm_feature_stats.pkl')
    if os.path.exists(ibm_stats_path):
        with open(ibm_stats_path, 'rb') as _f:
            ibm_stats = pickle.load(_f)
        feat_names_clipped = feature_names[:ibm_feat_count]
        for j, feat in enumerate(feat_names_clipped):
            if feat in ibm_stats:
                lo = ibm_stats[feat]['p01']
                hi = ibm_stats[feat]['p99']
                X_isw[:, j] = np.clip(X_isw[:, j], lo, hi)
        print("  [Domain Adapt] Percentile-clipped ISW features to IBM [P1, P99] range.")
    else:
        print("  [Domain Adapt] ibm_feature_stats.pkl not found — run phase0 first for best results.")

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

    # ── Threshold strategy for cross-domain inference ──────────
    # The IBM model's raw probabilities are compressed into a very narrow
    # range for ISW data (distribution shift). Two valid strategies:
    #   1. Absolute: use optimal_threshold from IBM test (unreliable cross-domain)
    #   2. Relative: flag top-K% by risk score, where K = IBM fraud rate
    #      (the methodologically sound approach for zero-label domains)
    # We apply RELATIVE as primary and normalise scores to [0,1] for display.

    _s3_cfg = cfg.get('three_stage_pipeline', {}).get('stage3_fieldtest', {})
    ibm_fraud_rate = cfg.get('ibm_fraud_rate_estimate', 0.027)  # ~2.7% from rebalanced sample

    # Relative threshold: flag top ibm_fraud_rate percentile
    relative_thresh_pct = (1 - ibm_fraud_rate) * 100
    relative_thresh = float(np.percentile(ml_probs, relative_thresh_pct))
    print(f"  Relative threshold ({ibm_fraud_rate:.1%} -> P{relative_thresh_pct:.1f}): {relative_thresh:.6f}")

    ml_preds = (ml_probs >= relative_thresh).astype(int)

    # Normalise scores to [0,1] for dashboard risk gauge
    score_min, score_max = ml_probs.min(), ml_probs.max()
    ml_scores_norm = (ml_probs - score_min) / (score_max - score_min + 1e-12)

    df_isw['ml_risk_score'] = ml_scores_norm   # normalised 0-1 for display
    df_isw['ml_risk_raw']   = ml_probs          # raw probability preserved
    df_isw['ml_flagged']    = ml_preds
    df_isw['alert_threshold'] = relative_thresh
    df_isw['inference_source'] = 'IBM_TRAINED_MODEL (relative-threshold)'

    # Apply Layer 4 Composite Rules (RUL-L4-01 Critical Escalation & RUL-L4-02 FP Auto-Clear)
    df_isw = apply_composite_rules(df_isw, ml_scores_norm, high_risk_cutoff=0.75, auto_clear_cutoff=0.15)

    n_flagged = ml_preds.sum()
    print(f"  IBM model flagged: {n_flagged:,} / {len(df_isw):,} "
          f"({100*n_flagged/len(df_isw):.2f}%) Interswitch transactions")

    # Structuring & cold-start features (guaranteed present for adversarial defense)
    rules = cfg.get('hard_rules', {})
    threshold = rules.get('amount_threshold_ugx', 10_000_000)
    struct_thresh_pct = rules.get('structuring_threshold_percent', 0.85)
    if 'structuring_proximity' not in df_isw.columns and 'amount' in df_isw.columns:
        df_isw['structuring_proximity'] = np.clip(df_isw['amount'] / (threshold + 1e-5), 0.0, 1.0)
    if 'is_structuring_zone' not in df_isw.columns and 'amount' in df_isw.columns:
        df_isw['is_structuring_zone'] = (
            (df_isw['amount'] >= (threshold * struct_thresh_pct)) & 
            (df_isw['amount'] < threshold)
        ).astype(int)
    if 'cold_start_flag' not in df_isw.columns and 'hist_tx_count' in df_isw.columns:
        df_isw['cold_start_flag'] = (df_isw['hist_tx_count'] <= 2).astype(int)
    if 'cold_start_structuring_risk' not in df_isw.columns and 'cold_start_flag' in df_isw.columns:
        df_isw['cold_start_structuring_risk'] = (df_isw['cold_start_flag'] & df_isw['is_structuring_zone']).astype(int)

    # ── Step 3b: Topology Isolation Forest (Unsupervised Defense) ──
    print(f"\n[Stage 3] Step 3b: Fitting Topology Isolation Forest (Unsupervised Defense)...")
    topo_iforest = TopologyIsolationForest(contamination=0.03, random_state=42)
    topo_iforest.fit(df_isw, max_samples=100_000)
    topo_res = topo_iforest.score(df_isw)
    df_isw['topo_anomaly_score'] = topo_res['scores']
    df_isw['is_topo_anomaly']   = topo_res['is_anomaly']

    topo_model_path = os.path.join(models_dir, 'topology_iforest.pkl')
    topo_iforest.save(topo_model_path)
    print(f"  Topology anomalies flagged: {int(topo_res['is_anomaly'].sum()):,} ({100*topo_res['is_anomaly'].mean():.2f}%)")

    # ── Step 4: Save scored dataset ───────────────────────────
    print(f"\n[Stage 3] Step 4: Saving scored Interswitch dataset...")
    scored_path = os.path.join(ROOT, s3_cfg['fieldtest_output'])
    save_cols = ['source', 'target', 'terminal_id', 'amount', 'tran_type', 'step',
                 'rule_triggered', 'rule_score', 'ml_risk_score', 'ml_flagged',
                 'auto_cleared', 'tier1_critical_escalation', 'composite_disposition',
                 'topo_anomaly_score', 'is_topo_anomaly',
                 'structuring_proximity', 'is_structuring_zone',
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

    # Use ml_preds (relative-threshold flags) for SHAP — NOT the absolute 0.3 cutoff
    # which passes zero ISW rows and produces an all-zeros SHAP matrix.
    high_risk_mask = ml_preds == 1
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
        print(f"  Not enough flagged samples for SHAP ({len(X_shap)} found). Falling back.")
        shap_vals = np.zeros((len(X_isw_sc[:50]), ibm_feat_count))

    # ── Step 7: Compute Operational KPIs ─────────────────────
    print(f"\n[Stage 3] Step 7: Computing Operational KPIs...")
    try:
        shap_vals_loaded = np.load(os.path.join(isw_shap_dir, 'shap_values.npy'))
    except Exception:
        shap_vals_loaded = np.zeros((50, ibm_feat_count))

    kpi_report = build_operational_kpis(
        df=df_isw,
        ml_probs=ml_probs,
        ml_threshold=relative_thresh,  # use the actual relative threshold, not 0.5
        model_bundle=model_bundle,
        X_sample=X_isw,
        shap_values=shap_vals_loaded,
        feature_names=feature_names[:ibm_feat_count],
        cfg=cfg,
        output_dir=os.path.join(ROOT, paths_cfg['processed_dir']),
        topo_iforest=topo_iforest,
    )

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Stage 3 Complete! Interswitch Field Test Summary:")
    print(f"    Transactions scored  : {len(df_isw):,}")
    print(f"    IBM model flagged    : {n_flagged:,} ({100*n_flagged/len(df_isw):.2f}%)")
    kpi1 = kpi_report.get('kpi_1_fp_reduction', {})
    kpi2 = kpi_report.get('kpi_2_latency', {})
    kpi3 = kpi_report.get('kpi_3_explainability', {})
    adv_kpi = kpi_report.get('adversarial_robustness', {})
    print(f"    KPI 1 FP Reduction   : {kpi1.get('fp_reduction_rate', 0):.1%}")
    print(f"    KPI 2 Latency        : {kpi2.get('mean_latency_ms', 0):.1f}ms / 100 tx")
    print(f"    KPI 3 XAI Coverage   : {kpi3.get('mean_top3_coverage', 0):.1%}")
    print(f"    Adv. Raw ML Recall   : {adv_kpi.get('adversarial_ml_recall', 0):.1%}")
    print(f"    Adv. Topology Recall : {adv_kpi.get('topology_iforest_recall', 0):.1%}")
    print(f"    Adv. Hybrid Recall   : {adv_kpi.get('hybrid_defense_recall', 0):.1%}")
    print(f"    KPIs Passed          : {kpi_report.get('kpis_met', 0)}/3")
    if bridge_report:
        print(f"    Pattern Similarity   : {bridge_report.get('overall_similarity', 0):.1%}")
    print("=" * 70)
    print("\n  ✅ All three stages complete. Launch the dashboard:")
    print("     streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()

