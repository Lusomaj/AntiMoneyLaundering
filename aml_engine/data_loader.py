"""
XAI-SNA AML — Data Loader
Handles:
  1. Ingestion of CARD_TRANSACTIONS_DATASET (Interswitch ATM) and MoMTSim (Mobile Money)
  2. Super-Node fusion: encryptedCard + from_account_id → super_node_id
  3. Unified schema normalization across both datasets
  4. Heuristic labeling for ATM data (Option C: Rules + Synthesis)
  5. Merged output ready for graph building and feature engineering
"""

import hashlib
import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'aml_config.yaml')


def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _make_super_node_id(card: str, account: str) -> str:
    """SHA-256 hash of card+account for privacy-preserving Super-Node ID."""
    raw = f"{str(card).strip()}|{str(account).strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_atm_dataset(path: str, cfg: dict, sample_size: int = None) -> pd.DataFrame:
    """
    Load Interswitch ATM card transaction dataset.
    Performs Super-Node fusion and heuristic fraud labeling.
    """
    print(f"[DataLoader] Loading ATM dataset from: {path}")
    nrows = sample_size if sample_size else None
    df = pd.read_csv(path, nrows=nrows, low_memory=False, on_bad_lines='skip')
    print(f"[DataLoader] ATM raw rows loaded: {len(df):,}")

    cfg_sn = cfg['super_node']
    rules  = cfg['hard_rules']

    # --- Super-Node Fusion ---
    card_col    = cfg_sn['card_col']       # encryptedCard
    account_col = cfg_sn['account_col']    # from_account_id
    terminal_col= cfg_sn['terminal_col']   # terminal_id
    amount_col  = cfg_sn['amount_col']     # settleAmountReq
    dt_col      = cfg_sn['datetime_col']   # datetimeReq
    ttype_col   = cfg_sn['tran_type_col']  # tran_type_desciption

    df['super_node_id'] = df.apply(
        lambda r: _make_super_node_id(r.get(card_col, ''), r.get(account_col, '')),
        axis=1
    )

    # --- Parse datetime ---
    df['timestamp'] = pd.to_datetime(df[dt_col], errors='coerce')

    # --- Unified schema ---
    unified = pd.DataFrame()
    unified['source']       = df['super_node_id']
    unified['target']       = df.get('to_account_id', df[account_col]).astype(str)
    unified['amount']       = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
    unified['tran_type']    = df.get(ttype_col, 'UNKNOWN').fillna('UNKNOWN')
    unified['timestamp']    = df['timestamp']
    unified['terminal_id']  = df[terminal_col].astype(str)
    unified['step']         = df['timestamp'].rank(method='dense').astype(int) if 'timestamp' in df else 0
    unified['dataset']      = 'ATM'

    # Raw flags (one-hot encoded transaction types)
    for flag_col in ['isWithdrawTrx', 'isTransferTrx', 'isRefundTrx', 'isDepositTrx', 'isPurchaseTrx']:
        if flag_col in df.columns:
            unified[flag_col] = pd.to_numeric(df[flag_col], errors='coerce').fillna(0).astype(int)
        else:
            unified[flag_col] = 0

    unified['settle_currency'] = df.get('currency_alpha_code', 'UGX').fillna('UGX')

    # --- Heuristic Fraud Labeling (Option C) ---
    threshold      = rules['amount_threshold_ugx']
    smurfing_low   = rules['smurfing_low_amount_ugx']
    high_risk_types= rules['high_risk_transaction_types']

    # Rule 1: Amount above threshold
    flag_threshold = (unified['amount'] >= threshold).astype(int)

    # Rule 2: Withdrawal + high amount
    flag_withdraw  = ((unified['isWithdrawTrx'] == 1) & (unified['amount'] > threshold * 0.5)).astype(int)

    # Rule 3: Small amount structuring suspicion
    flag_small     = (unified['amount'] < smurfing_low) & (unified['amount'] > 0)

    # Rule 4: High-risk type
    flag_type      = unified['tran_type'].isin(high_risk_types).astype(int)

    # Composite heuristic label: 2+ rules → suspicious
    combined_score = flag_threshold + flag_withdraw + flag_type + flag_small.astype(int)
    unified['is_suspicious'] = (combined_score >= 2).astype(int)

    suspicious_count = unified['is_suspicious'].sum()
    total = len(unified)
    print(f"[DataLoader] ATM heuristic labels — Suspicious: {suspicious_count:,} ({100*suspicious_count/total:.2f}%)")

    return unified.reset_index(drop=True)


def load_momo_dataset(path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load MoMTSim Mobile Money Simulation dataset.
    Has real isFraud labels.
    """
    print(f"[DataLoader] Loading MoMo dataset from: {path}")
    nrows = sample_size if sample_size else None
    df = pd.read_csv(path, nrows=nrows, low_memory=False, on_bad_lines='skip')
    print(f"[DataLoader] MoMo raw rows loaded: {len(df):,}")

    unified = pd.DataFrame()
    unified['source']       = df['initiatorID'].astype(str)
    unified['target']       = df['recipientID'].astype(str)
    unified['amount']       = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    unified['tran_type']    = df['transactionType'].fillna('UNKNOWN')
    unified['timestamp']    = pd.NaT   # MoMo uses step not datetime
    unified['terminal_id']  = 'MOMO_VIRTUAL'
    unified['step']         = df['step']
    unified['dataset']      = 'MOMO'

    # MoMo doesn't have these flags; set sensibly from tran_type
    unified['isWithdrawTrx'] = (df['transactionType'].isin(['CASH_OUT'])).astype(int)
    unified['isTransferTrx'] = (df['transactionType'].isin(['TRANSFER'])).astype(int)
    unified['isRefundTrx']   = 0
    unified['isDepositTrx']  = (df['transactionType'].isin(['CASH_IN'])).astype(int)
    unified['isPurchaseTrx'] = (df['transactionType'].isin(['PAYMENT'])).astype(int)
    unified['settle_currency'] = 'UGX'
    unified['is_suspicious']  = df['isFraud'].astype(int)

    fraud_count = unified['is_suspicious'].sum()
    total = len(unified)
    print(f"[DataLoader] MoMo labeled — Fraud: {fraud_count:,} ({100*fraud_count/total:.2f}%)")

    return unified.reset_index(drop=True)


def load_and_merge(atm_path: str, momo_path: str, cfg: dict) -> pd.DataFrame:
    """
    Main entry point: load both datasets, merge, and return unified DataFrame.
    """
    sample_cfg = cfg.get('sampling', {})
    training_sample = sample_cfg.get('training_sample_size', 200000)
    sna_sample = sample_cfg.get('sna_sample_size', 150000)

    # Load with sampling for training
    atm_df  = load_atm_dataset(atm_path, cfg, sample_size=training_sample)
    momo_df = load_momo_dataset(momo_path, sample_size=training_sample // 2)

    # Merge
    merged = pd.concat([atm_df, momo_df], ignore_index=True)
    print(f"\n[DataLoader] Merged dataset: {len(merged):,} rows | "
          f"Suspicious: {merged['is_suspicious'].sum():,} "
          f"({100*merged['is_suspicious'].mean():.2f}%)")

    # Sort by step/timestamp for temporal features
    merged = merged.sort_values('step').reset_index(drop=True)
    return merged

