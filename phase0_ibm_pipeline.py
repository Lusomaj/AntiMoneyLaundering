"""
Anti-Gravity AML — Phase 0: IBM Gold Standard Training (Stage 1)
Orchestrates the full IBM AML pipeline:
  1. Ingest IBM HI-Large dataset (stratified sample from 5GB)
  2. Parse IBM Patterns file (STACK / CYCLE / FAN-IN / FAN-OUT blocks)
  3. Build Super-Node transaction graph + compute SNA features
  4. Detect laundering motifs
  5. Engineer all features (tabular + SNA + temporal)
  6. Apply hard rules
  7. Train full model suite (RF / XGB / MLP / Stacked Ensemble / GAT)
  8. Compute SHAP explainability
  9. Save all artifacts tagged as 'IBM' for use by Stages 2 & 3

Run this BEFORE phase1_data_prep.py (Interswitch pipeline).
"""

import os, sys, pickle, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

# Force UTF-8 stdout so Unicode chars (→ ✅ ⏳) don't crash on Windows cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from aml_engine.data_loader      import load_config
from aml_engine.ibm_loader       import load_ibm_dataset, parse_ibm_patterns
from aml_engine.graph_builder    import build_graph, compute_sna_features, save_graph
from aml_engine.motif_detector   import apply_motif_features
from aml_engine.feature_engineer import engineer_all_features, ALL_FEATURES
from aml_engine.rules_engine     import apply_rules
from aml_engine.model_trainer    import train_all_models
from aml_engine.xai_engine       import compute_shap_values

ROOT = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT, 'aml_config.yaml')


def main():
    print("=" * 70)
    print("  ANTI-GRAVITY AML — Stage 1: IBM Gold Standard Training")
    print("  Dataset: IBM HI-Large AML (labeled, global patterns)")
    print("=" * 70)

    cfg       = load_config()
    s1_cfg    = cfg['three_stage_pipeline']['stage1_ibm']
    paths_cfg = cfg['paths']

    ibm_path      = os.path.join(ROOT, s1_cfg['ibm_dataset_path'])
    patterns_path = os.path.join(ROOT, s1_cfg['ibm_patterns_path'])

    # Output directories
    ibm_models_dir = os.path.dirname(s1_cfg['ibm_model_output'])
    ibm_shap_dir   = s1_cfg['ibm_shap_output']
    graphs_dir     = os.path.join(ROOT, paths_cfg['graphs_dir'])
    processed_dir  = os.path.join(ROOT, paths_cfg['processed_dir'])

    for d in [ibm_models_dir, ibm_shap_dir, graphs_dir, processed_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Step 1: Ingest IBM data ───────────────────────────────
    print("\n[Stage 1] Step 1: Ingesting IBM AML dataset...")
    df_ibm = load_ibm_dataset(ibm_path, s1_cfg, root_dir=ROOT)

    # ── Step 2: Parse IBM Patterns file ──────────────────────
    print("\n[Stage 1] Step 2: Parsing IBM Patterns file...")
    try:
        ibm_patterns = parse_ibm_patterns(patterns_path, root_dir=ROOT)
    except Exception as e:
        print(f"  Warning: Could not parse patterns file: {e}")
        ibm_patterns = {}

    # Save patterns metadata
    with open(os.path.join(processed_dir, 'ibm_patterns_meta.json'), 'w') as f:
        # Truncate for storage (save first 5 examples of each type)
        slim = {k: v[:5] for k, v in ibm_patterns.items()}
        json.dump(slim, f, indent=2, default=str)
    print(f"  Patterns parsed: { {k: len(v) for k, v in ibm_patterns.items()} }")

    # ── Step 3: Build transaction graph ──────────────────────
    print("\n[Stage 1] Step 3: Building IBM transaction graph...")
    # Temporarily set sna_sample_size for IBM graph
    cfg_for_graph = dict(cfg)
    cfg_for_graph['sampling'] = {
        'sna_sample_size': min(s1_cfg.get('ibm_sample_size', 200000), 200000),
        'training_sample_size': s1_cfg.get('ibm_sample_size', 200000),
    }
    G_ibm = build_graph(df_ibm, cfg_for_graph)

    # ── Step 4: Compute SNA features ─────────────────────────
    print("\n[Stage 1] Step 4: Computing IBM SNA features...")
    ibm_sna_features = compute_sna_features(G_ibm, cfg_for_graph)

    # Save IBM-specific graph artifacts
    ibm_graph_dir = os.path.join(graphs_dir, 'ibm')
    os.makedirs(ibm_graph_dir, exist_ok=True)
    save_graph(G_ibm, ibm_sna_features, ibm_graph_dir)

    # ── Step 5: Detect motifs ─────────────────────────────────
    print("\n[Stage 1] Step 5: Detecting laundering motifs in IBM data...")
    df_ibm, ibm_motif_meta = apply_motif_features(df_ibm, G_ibm, cfg)
    with open(os.path.join(ibm_graph_dir, 'motif_metadata.pkl'), 'wb') as f:
        pickle.dump(ibm_motif_meta, f)

    # ── Step 6: Feature engineering ──────────────────────────
    print("\n[Stage 1] Step 6: Engineering IBM features...")
    df_ibm = engineer_all_features(df_ibm, ibm_sna_features, cfg)

    # ── Step 7: Apply rules ───────────────────────────────────
    print("\n[Stage 1] Step 7: Applying hard rules to IBM data...")
    df_ibm = apply_rules(df_ibm, cfg)

    # ── Step 8: Save processed IBM dataset ───────────────────
    feature_cols = [f for f in ALL_FEATURES if f in df_ibm.columns]
    save_cols = feature_cols + ['is_suspicious', 'rule_triggered', 'rule_score',
                                  'source', 'target', 'terminal_id', 'dataset', 'step',
                                  'tran_type', 'amount', 'from_bank', 'to_bank',
                                  'is_cross_bank']
    save_cols = [c for c in save_cols if c in df_ibm.columns]

    ibm_feat_path = os.path.join(processed_dir, 'ibm_features_full.csv')
    df_ibm[save_cols].to_csv(ibm_feat_path, index=False)
    print(f"\n[Stage 1] Step 8: ✅ IBM processed features saved → {ibm_feat_path}")

    # ── Step 9: Train all models on IBM data ─────────────────
    print("\n[Stage 1] Step 9: Training models on IBM labeled data...")
    ibm_results = train_all_models(df_ibm, cfg, ibm_models_dir)

    # Rename IBM model comparison file
    ibm_results_path = s1_cfg['ibm_results_output']
    ibm_results.to_csv(os.path.join(ROOT, ibm_results_path), index=False)

    # Rename best model file to ibm_best_model.pkl
    default_best = os.path.join(ibm_models_dir, 'best_model.pkl')
    ibm_best     = os.path.join(ROOT, s1_cfg['ibm_model_output'])
    if os.path.exists(default_best) and default_best != ibm_best:
        import shutil
        shutil.copy(default_best, ibm_best)
    print(f"[Stage 1] IBM best model saved → {ibm_best}")

    # ── Step 10: SHAP Explainability ─────────────────────────
    print("\n[Stage 1] Step 10: Computing SHAP values for IBM best model...")
    ibm_best_model_path = os.path.join(ROOT, s1_cfg['ibm_model_output'])
    if os.path.exists(ibm_best_model_path):
        with open(ibm_best_model_path, 'rb') as f:
            model_bundle = pickle.load(f)

        feat_path = os.path.join(ibm_models_dir, 'feature_names.pkl')
        if os.path.exists(feat_path):
            with open(feat_path, 'rb') as f:
                feat_dict = pickle.load(f)
            feature_names = feat_dict.get('all', feature_cols)
        else:
            feature_names = feature_cols

        X_test = np.load(os.path.join(ibm_models_dir, 'X_test_all.npy'))
        y_test = np.load(os.path.join(ibm_models_dir, 'y_test.npy'))

        os.makedirs(ibm_shap_dir, exist_ok=True)
        compute_shap_values(
            model_bundle=model_bundle,
            X_test=X_test,
            feature_names=feature_names,
            output_dir=ibm_shap_dir,
            max_samples=500,
        )

    # ── Final Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Stage 1 Complete! IBM Gold Standard Summary:")
    print(f"    Rows processed       : {len(df_ibm):,}")
    print(f"    Fraud (labeled)      : {df_ibm['is_suspicious'].sum():,} "
          f"({100*df_ibm['is_suspicious'].mean():.4f}%)")
    print(f"    Motif-circular       : {df_ibm['motif_circular'].sum():,}")
    print(f"    Motif-smurfing       : {df_ibm['motif_smurfing'].sum():,}")
    print(f"    Motif-reversal       : {df_ibm['motif_reversal'].sum():,}")
    print(f"    IBM model saved      : {ibm_best}")
    print(f"    IBM SHAP saved       : {ibm_shap_dir}")
    print("=" * 70)
    print("\n  ✅ Stage 1 artifacts ready for Pattern Bridge (Stage 2)")
    print("     Next: python phase_interswitch_fieldtest.py")


if __name__ == "__main__":
    main()
