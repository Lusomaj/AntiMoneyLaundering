"""
Anti-Gravity AML — Phase 1: Data Preparation & Feature Engineering
Orchestrates: DataLoader → GraphBuilder → MotifDetector → FeatureEngineer → RulesEngine
Run this FIRST before model training.
"""

import os, sys, pickle, yaml, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

# Force UTF-8 stdout so Unicode chars (→ ✅ ⏳) don't crash on Windows cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from aml_engine.data_loader      import load_config, load_and_merge
from aml_engine.graph_builder    import build_graph, compute_sna_features, save_graph
from aml_engine.motif_detector   import apply_motif_features
from aml_engine.feature_engineer import engineer_all_features, ALL_FEATURES
from aml_engine.rules_engine     import apply_rules


ATM_PATH  = r'E:\MASTERSProject\AMLProject\CARD_TRANSACTIONS_DATASET.csv'
MOMO_PATH = r'E:\MASTERSProject\AMLProject\MoMTSim_202408AML.csv'
CONFIG_PATH = r'E:\MASTERSProject\AMLProject\aml_config.yaml'


def main():
    print("=" * 70)
    print("  ANTI-GRAVITY AML SYSTEM — Phase 1: Data Preparation")
    print("=" * 70)

    cfg = load_config()
    paths = cfg['paths']
    processed_dir = paths['processed_dir']
    graphs_dir    = paths['graphs_dir']
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # Step 1: Load & merge datasets
    print("\n[Phase 1] Step 1: Ingesting datasets...")
    df = load_and_merge(ATM_PATH, MOMO_PATH, cfg)

    # Step 2: Build transaction graph
    print("\n[Phase 1] Step 2: Building transaction graph...")
    G = build_graph(df, cfg)

    # Step 3: Compute SNA features
    print("\n[Phase 1] Step 3: Computing SNA features...")
    sna_features = compute_sna_features(G, cfg)
    save_graph(G, sna_features, graphs_dir)

    # Step 4: Detect motifs
    print("\n[Phase 1] Step 4: Detecting laundering motifs...")
    df, motif_meta = apply_motif_features(df, G, cfg)
    with open(os.path.join(graphs_dir, 'motif_metadata.pkl'), 'wb') as f:
        pickle.dump(motif_meta, f)

    # Step 5: Full feature engineering
    print("\n[Phase 1] Step 5: Engineering features...")
    df = engineer_all_features(df, sna_features, cfg)

    # Step 6: Apply rules engine
    print("\n[Phase 1] Step 6: Applying hard rules...")
    df = apply_rules(df, cfg)

    # Step 7: Save processed dataset
    print("\n[Phase 1] Step 7: Saving processed dataset...")
    feature_cols = [f for f in ALL_FEATURES if f in df.columns]
    save_cols = feature_cols + ['is_suspicious', 'rule_triggered', 'rule_score',
                                  'source', 'target', 'terminal_id', 'dataset', 'step',
                                  'tran_type', 'amount']
    save_cols = [c for c in save_cols if c in df.columns]

    output_path = os.path.join(processed_dir, 'features_full.csv')
    df[save_cols].to_csv(output_path, index=False)
    print(f"[Phase 1] ✅ Saved → {output_path} ({len(df):,} rows)")

    # Summary
    print("\n" + "=" * 70)
    print("  Phase 1 Complete! Summary:")
    print(f"    Total transactions : {len(df):,}")
    print(f"    Suspicious (label) : {df['is_suspicious'].sum():,} ({100*df['is_suspicious'].mean():.2f}%)")
    print(f"    Rule-triggered     : {df['rule_triggered'].sum():,}")
    print(f"    Motif-circular     : {df.get('motif_circular', 0).sum() if 'motif_circular' in df else 'N/A'}")
    print(f"    Motif-smurfing     : {df.get('motif_smurfing', 0).sum() if 'motif_smurfing' in df else 'N/A'}")
    print(f"    Motif-reversal     : {df.get('motif_reversal', 0).sum() if 'motif_reversal' in df else 'N/A'}")
    print(f"    Feature columns    : {len(feature_cols)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
