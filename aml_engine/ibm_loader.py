"""
Anti-Gravity AML — IBM AML Data Loader
Handles the IBM HI-Large AML dataset (5GB, labeled).

Key features:
  - Stratified sampling from the 5GB file (takes ~500k rows but preserves fraud rows)
  - Column deduplication: IBM has two 'Account' columns (From/To) — reads by position
  - Pattern file parsing: extracts labeled laundering attempt blocks
  - Graph construction + SNA feature engineering using IBM data
  - Outputs unified schema matching the Interswitch pipeline
"""

import os
import csv
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict


IBM_COLUMNS = [
    'Timestamp', 'From Bank', 'Account_From', 'To Bank', 'Account_To',
    'Amount Received', 'Receiving Currency', 'Amount Paid',
    'Payment Currency', 'Payment Format', 'Is Laundering'
]

PAYMENT_FORMAT_MAP = {
    'bitcoin': 'CRYPTO',    'cash':          'CASH_OUT',
    'wire':    'TRANSFER',  'ach':           'TRANSFER',
    'cheque':  'PAYMENT',   'credit card':   'PURCHASE',
    'reinvestment': 'TRANSFER',
}


def _encode_payment_format(fmt: str, tl_cfg: dict) -> int:
    """Map payment format to risk-tier integer."""
    risk_map = tl_cfg.get('payment_format_risk', {})
    return risk_map.get(fmt, 0)


def _to_unified_tran_type(fmt: str) -> str:
    return PAYMENT_FORMAT_MAP.get(str(fmt).lower().strip(), 'TRANSFER')


def load_ibm_dataset(ibm_path: str, tl_cfg: dict, root_dir: str = '.') -> pd.DataFrame:
    """
    Load IBM HI-Large AML dataset with stratified sampling.
    Reads by column position to handle duplicate 'Account' header.

    Returns unified DataFrame with same schema as Interswitch pipeline.
    """
    full_path = os.path.join(root_dir, ibm_path)
    sample_size = tl_cfg.get('ibm_sample_size', 500000)
    oversample_n = tl_cfg.get('ibm_fraud_oversample', 50)

    print(f"[IBM Loader] Loading from: {full_path}")
    print(f"[IBM Loader] Sample size: {sample_size:,} rows | Fraud oversample: {oversample_n}x")

    normal_rows, fraud_rows = [], []
    normal_cap = sample_size
    fraud_cap  = sample_size  # no cap on fraud rows — we need all of them

    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for i, row in enumerate(reader):
            if len(row) < 11:
                continue
            is_fraud = int(row[10].strip()) if row[10].strip().isdigit() else 0

            parsed = {
                'Timestamp':        row[0].strip(),
                'From_Bank':        row[1].strip(),
                'Account_From':     row[2].strip(),
                'To_Bank':          row[3].strip(),
                'Account_To':       row[4].strip(),
                'Amount_Received':  row[5].strip(),
                'Receiving_Currency': row[6].strip(),
                'Amount_Paid':      row[7].strip(),
                'Payment_Currency': row[8].strip(),
                'Payment_Format':   row[9].strip(),
                'Is_Laundering':    is_fraud,
            }

            if is_fraud:
                fraud_rows.append(parsed)
            else:
                if len(normal_rows) < normal_cap:
                    normal_rows.append(parsed)

            if len(normal_rows) >= normal_cap and i > sample_size * 2:
                break

    fraud_count  = len(fraud_rows)
    normal_count = len(normal_rows)
    print(f"[IBM Loader] Loaded: {normal_count:,} normal | {fraud_count:,} fraud")

    if fraud_count == 0:
        raise ValueError("[IBM Loader] No fraud rows found in sample! Increase ibm_sample_size.")

    # Oversample fraud rows to reach minimum useful count
    min_fraud = tl_cfg.get('min_fraud_rows', 200)
    if fraud_count < min_fraud:
        repeat_times = (min_fraud // fraud_count) + 1
        fraud_rows = fraud_rows * repeat_times
        print(f"[IBM Loader] Fraud rows oversampled: {len(fraud_rows):,}")

    # Combine
    all_rows = normal_rows + fraud_rows
    df_raw   = pd.DataFrame(all_rows)

    # Build unified schema
    df = pd.DataFrame()
    df['source']     = df_raw['Account_From'].astype(str)
    df['target']     = df_raw['Account_To'].astype(str)
    df['from_bank']  = df_raw['From_Bank'].astype(str)
    df['to_bank']    = df_raw['To_Bank'].astype(str)
    df['amount']     = pd.to_numeric(df_raw['Amount_Paid'], errors='coerce').fillna(0)

    # Timestamp → step (hours since epoch start)
    df['timestamp'] = pd.to_datetime(df_raw['Timestamp'], errors='coerce')
    t_min = df['timestamp'].min()
    df['step'] = ((df['timestamp'] - t_min).dt.total_seconds() / 3600).fillna(0).astype(int)

    df['tran_type']    = df_raw['Payment_Format'].apply(_to_unified_tran_type)
    df['currency']     = df_raw['Payment_Currency']
    df['payment_format_risk'] = df_raw['Payment_Format'].apply(
        lambda x: _encode_payment_format(x, tl_cfg))
    df['is_cross_bank'] = (df_raw['From_Bank'] != df_raw['To_Bank']).astype(int)
    df['is_suspicious'] = df_raw['Is_Laundering'].astype(int)
    df['terminal_id']   = 'IBM_VIRTUAL'   # no terminal in IBM data
    df['dataset']       = 'IBM'

    # Add flag columns (matching Interswitch schema)
    df['isWithdrawTrx'] = (df['tran_type'] == 'CASH_OUT').astype(int)
    df['isTransferTrx'] = (df['tran_type'] == 'TRANSFER').astype(int)
    df['isRefundTrx']   = 0
    df['isDepositTrx']  = 0
    df['isPurchaseTrx'] = (df['tran_type'] == 'PURCHASE').astype(int)
    df['settle_currency'] = df['currency']

    fraud_final  = df['is_suspicious'].sum()
    total_final  = len(df)
    print(f"[IBM Loader] Final dataset: {total_final:,} rows | "
          f"Fraud: {fraud_final:,} ({100*fraud_final/total_final:.3f}%)")

    return df.sort_values('step').reset_index(drop=True)


def parse_ibm_patterns(patterns_path: str, root_dir: str = '.') -> Dict[str, list]:
    """
    Parse the IBM Patterns file to extract labeled laundering attempt blocks.
    Returns dict: pattern_type → list of transaction dicts in that pattern.
    Used to enrich motif metadata for the UI.
    """
    full_path = os.path.join(root_dir, patterns_path)
    patterns  = {'STACK': [], 'CYCLE': [], 'FAN-IN': [], 'FAN-OUT': [], 'SCATTER-GATHER': []}
    current_type = None
    current_block = []

    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('BEGIN LAUNDERING ATTEMPT'):
                for ptype in patterns:
                    if ptype in line:
                        current_type = ptype
                        current_block = []
                        break
            elif line.startswith('END LAUNDERING ATTEMPT'):
                if current_type and current_block:
                    patterns[current_type].append(current_block[:])
                current_type = None
                current_block = []
            elif current_type and ',' in line:
                parts = line.split(',')
                if len(parts) >= 11:
                    current_block.append({
                        'timestamp': parts[0], 'from_bank': parts[1],
                        'source': parts[2], 'to_bank': parts[3],
                        'target': parts[4], 'amount': parts[7],
                        'currency': parts[8], 'format': parts[9],
                    })

    summary = {k: len(v) for k, v in patterns.items()}
    print(f"[IBM Loader] Parsed pattern blocks: {summary}")
    return patterns
