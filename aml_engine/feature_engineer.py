"""
XAI-SNA AML — Feature Engineer
Combines three categories of features for every transaction:
  1. Tabular (raw transaction attributes)
  2. SNA (graph-derived: degree, betweenness, PageRank, community)
  3. Temporal Velocity (burst detection, historical averages)

Outputs a flat feature matrix ready for model training.
"""

import numpy as np
import pandas as pd
from typing import Dict


# ─────────────────────────────────────────────────────────
# FEATURE COLUMNS DEFINITION
# ─────────────────────────────────────────────────────────

TABULAR_FEATURES = [
    'amount', 'amount_log', 'is_small_amount', 'is_large_amount',
    'isWithdrawTrx', 'isTransferTrx', 'isRefundTrx', 'isDepositTrx', 'isPurchaseTrx',
    'tran_type_encoded',
]

SNA_FEATURES = [
    'source_degree_cent', 'source_betweenness', 'source_pagerank',
    'target_degree_cent', 'target_betweenness', 'target_pagerank',
    'terminal_pagerank', 'community_id', 'community_size',
    'is_cross_community', 'source_is_infrastructure', 'target_is_infrastructure',
]

TEMPORAL_FEATURES = [
    'hist_tx_count', 'tx_count_last_step', 'amount_vs_hist_mean',
    'total_amount_last_step', 'time_since_last_tx',
]

MOTIF_FEATURES = [
    'motif_circular', 'motif_smurfing', 'motif_reversal',
]

ALL_FEATURES = TABULAR_FEATURES + SNA_FEATURES + TEMPORAL_FEATURES + MOTIF_FEATURES

# Raw-only features (for ablation: "ML without SNA")
RAW_ONLY_FEATURES = TABULAR_FEATURES + TEMPORAL_FEATURES + MOTIF_FEATURES

# SNA-heavy features set
SNA_ENHANCED_FEATURES = ALL_FEATURES


def _encode_tran_type(df: pd.DataFrame) -> pd.Series:
    type_map = {
        'TRANSFER': 1, 'WITHDRAWAL': 2, 'CASH_OUT': 3, 'CASH_IN': 4,
        'PAYMENT': 5, 'DEBIT': 6, 'CREDIT': 7, 'REVERSAL': 8,
        'PURCHASE': 9, 'REFUND': 10, 'UNKNOWN': 0,
    }
    return df['tran_type'].str.upper().map(type_map).fillna(0).astype(int)


def build_tabular_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Extract and engineer raw tabular features."""
    rules = cfg['hard_rules']
    threshold     = rules['amount_threshold_ugx']
    smurfing_low  = rules['smurfing_low_amount_ugx']

    df['amount_log']       = np.log1p(df['amount'])
    df['is_small_amount']  = (df['amount'] < smurfing_low).astype(int)
    df['is_large_amount']  = (df['amount'] >= threshold).astype(int)
    df['tran_type_encoded'] = _encode_tran_type(df)

    # Ensure all required flag columns exist
    for col in ['isWithdrawTrx', 'isTransferTrx', 'isRefundTrx', 'isDepositTrx', 'isPurchaseTrx']:
        if col not in df.columns:
            df[col] = 0

    return df


def build_sna_features(df: pd.DataFrame, sna_features: Dict) -> pd.DataFrame:
    """Map SNA node features back to each transaction row."""
    print("[FeatureEngineer] Mapping SNA features to transactions...")

    # Source node features
    df['source_degree_cent']       = df['source'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('degree_centrality', 0.0))
    df['source_betweenness']       = df['source'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('betweenness_centrality', 0.0))
    df['source_pagerank']          = df['source'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('pagerank', 0.0))
    df['source_is_infrastructure'] = df['source'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('is_infrastructure', 0))
    df['community_id']             = df['source'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('community_id', -1))
    df['community_size']           = df['source'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('community_size', 1))

    # Target node features
    df['target_degree_cent']       = df['target'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('degree_centrality', 0.0))
    df['target_betweenness']       = df['target'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('betweenness_centrality', 0.0))
    df['target_pagerank']          = df['target'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('pagerank', 0.0))
    df['target_is_infrastructure'] = df['target'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('is_infrastructure', 0))

    # Terminal (infrastructure) node PageRank
    df['terminal_pagerank']        = df['terminal_id'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('pagerank', 0.0))

    # Is transaction crossing community boundaries?
    target_comm = df['target'].astype(str).map(
        lambda x: sna_features.get(x, {}).get('community_id', -99))
    df['is_cross_community'] = (df['community_id'] != target_comm).astype(int)

    return df


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer temporal and velocity features."""
    print("[FeatureEngineer] Engineering temporal/velocity features...")
    df = df.sort_values('step').copy()

    # Cumulative transaction count per source
    df['hist_tx_count'] = df.groupby('source').cumcount()

    # Transaction count in current step (burst indicator)
    df['tx_count_last_step'] = df.groupby(['source', 'step'])['amount'].transform('count')

    # Historical mean amount per source (to identify amount spikes)
    hist_mean = df.groupby('source')['amount'].transform(
        lambda x: x.expanding().mean().shift().fillna(x.iloc[0] if len(x) > 0 else 0)
    )
    df['amount_vs_hist_mean'] = df['amount'] / (hist_mean + 1e-5)

    # Total amount in current step
    df['total_amount_last_step'] = df.groupby(['source', 'step'])['amount'].transform('sum')

    # Time since last transaction (step-based proxy)
    df['time_since_last_tx'] = df.groupby('source')['step'].diff().fillna(999).clip(upper=999)

    return df


def engineer_all_features(df: pd.DataFrame, sna_features: Dict, cfg: dict) -> pd.DataFrame:
    """
    Master function: run all three feature engineering stages.
    Assumes motif columns (motif_circular, motif_smurfing, motif_reversal) 
    are already in df from motif_detector.apply_motif_features().
    """
    print("[FeatureEngineer] Starting full feature engineering pipeline...")

    df = build_tabular_features(df, cfg)
    df = build_sna_features(df, sna_features)
    df = build_temporal_features(df)

    # Ensure motif columns exist even if motif detector was skipped
    for col in MOTIF_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Fill any NaNs in feature columns
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print(f"[FeatureEngineer] Feature engineering complete. "
          f"Feature columns: {len(ALL_FEATURES)} | Rows: {len(df):,}")
    return df

