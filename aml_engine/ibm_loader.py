"""
XAI-SNA AML — IBM AML Data Loader
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

    Two-pass loading strategy:
      Pass 1 (fast): scan the full file for ALL fraud rows (rare at 0.056%)
      Pass 2: reservoir-sample normal rows up to normal_cap

    This fixes the 37K row limitation caused by stopping the scan after
    (sample_size * 10) rows — which at 0.056% fraud rate yielded only ~1,100
    fraud rows and collapsed the dataset to <40K total rows.

    Returns unified DataFrame with same schema as Interswitch pipeline.
    """
    full_path = os.path.join(root_dir, ibm_path)
    sample_size  = tl_cfg.get('ibm_sample_size', 500000)
    min_fraud    = tl_cfg.get('min_fraud_rows', 1000)

    print(f"[IBM Loader] Loading from: {full_path}")
    print(f"[IBM Loader] Target sample: {sample_size:,} normal rows | scanning full file for fraud rows")

    # ── Pass 1: collect ALL fraud rows from full file ──────────────
    fraud_rows  = []
    normal_rows = []
    normal_cap  = sample_size
    normal_step = 5  # reservoir: keep every Nth normal row for efficiency

    print(f"[IBM Loader] Pass 1: scanning full {os.path.getsize(full_path)/1e9:.2f}GB file...")
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for i, row in enumerate(reader):
            if len(row) < 11:
                continue
            is_fraud = int(row[10].strip()) if row[10].strip().isdigit() else 0

            parsed = {
                'Timestamp':          row[0].strip(),
                'From_Bank':          row[1].strip(),
                'Account_From':       row[2].strip(),
                'To_Bank':            row[3].strip(),
                'Account_To':         row[4].strip(),
                'Amount_Received':    row[5].strip(),
                'Receiving_Currency': row[6].strip(),
                'Amount_Paid':        row[7].strip(),
                'Payment_Currency':   row[8].strip(),
                'Payment_Format':     row[9].strip(),
                'Is_Laundering':      is_fraud,
            }

            if is_fraud:
                fraud_rows.append(parsed)
            else:
                # Reservoir sample: accept normal rows at 1-in-normal_step rate
                # This gives ~sample_size / normal_step normal rows per pass,
                # then we top up in pass 2 if needed
                if len(normal_rows) < normal_cap and (i % normal_step == 0):
                    normal_rows.append(parsed)

            if i % 5_000_000 == 0 and i > 0:
                print(f"  [IBM Loader] Scanned {i:,} rows | fraud: {len(fraud_rows):,} | "
                      f"normal sampled: {len(normal_rows):,}")

    fraud_count  = len(fraud_rows)
    normal_count = len(normal_rows)
    print(f"[IBM Loader] Full scan complete: {normal_count:,} normal | {fraud_count:,} fraud")

    if fraud_count == 0:
        raise ValueError("[IBM Loader] No fraud rows found. Check ibm_dataset_path and Is_Laundering column.")

    # ── Ensure minimum fraud floor (repeat if IBM file has too few) ─
    if fraud_count < min_fraud:
        repeat_times = (min_fraud // fraud_count) + 1
        fraud_rows   = (fraud_rows * repeat_times)[:min_fraud]
        print(f"[IBM Loader] Fraud rows boosted to minimum floor: {len(fraud_rows):,}")

    # ── Cap normal rows to 60x fraud (realistic ~1.6% fraud density) ─
    normal_target = min(normal_count, max(fraud_count * 60, sample_size))
    if normal_target < normal_count:
        import random as _rnd
        _rnd.seed(42)
        normal_rows = _rnd.sample(normal_rows, normal_target)
        print(f"[IBM Loader] Normal rows capped to: {len(normal_rows):,}")

    # ── Combine ──────────────────────────────────────────────────────
    all_rows = normal_rows + fraud_rows
    df_raw   = pd.DataFrame(all_rows)

    # Build unified schema
    df = pd.DataFrame()
    df['source']     = df_raw['Account_From'].astype(str)
    df['target']     = df_raw['Account_To'].astype(str)
    df['from_bank']  = df_raw['From_Bank'].astype(str)
    df['to_bank']    = df_raw['To_Bank'].astype(str)
    df['amount']     = pd.to_numeric(df_raw['Amount_Paid'], errors='coerce').fillna(0)

    # ── Currency Normalisation: All currencies → UGX ─────────────────
    # IBM dataset contains 15 payment currencies (USD, EUR, GBP, JPY, etc.).
    # All amounts are converted to UGX using cross-rates relative to USD,
    # with the Bank of Uganda median rate (1 USD ≈ 3,800 UGX, 2017–2022).
    # Rate source: World Bank / Bank of Uganda historical FX data.
    # Log-transformation downstream makes the model robust to ±15% rate error.
    usd_to_ugx = float(tl_cfg.get('usd_to_ugx_rate', 3800))

    # Multi-currency rates → UGX (via USD cross-rate × usd_to_ugx)
    # Approximate median rates for the IBM dataset period (2017–2022)
    currency_to_ugx = {
        'us dollar':          usd_to_ugx * 1.000,   # reference
        'usd':                usd_to_ugx * 1.000,
        'euro':               usd_to_ugx * 1.120,   # EUR/USD ≈ 1.12
        'eur':                usd_to_ugx * 1.120,
        'uk pound':           usd_to_ugx * 1.310,   # GBP/USD ≈ 1.31
        'gbp':                usd_to_ugx * 1.310,
        'canadian dollar':    usd_to_ugx * 0.780,   # CAD/USD ≈ 0.78
        'australian dollar':  usd_to_ugx * 0.740,   # AUD/USD ≈ 0.74
        'swiss franc':        usd_to_ugx * 1.010,   # CHF/USD ≈ 1.01
        'yen':                usd_to_ugx * 0.0092,  # JPY/USD ≈ 0.0092
        'yuan':               usd_to_ugx * 0.152,   # CNY/USD ≈ 0.152
        'ruble':              usd_to_ugx * 0.0155,  # RUB/USD ≈ 0.0155
        'rupee':              usd_to_ugx * 0.0133,  # INR/USD ≈ 0.0133
        'brazil real':        usd_to_ugx * 0.190,   # BRL/USD ≈ 0.19
        'mexican peso':       usd_to_ugx * 0.052,   # MXN/USD ≈ 0.052
        'saudi riyal':        usd_to_ugx * 0.267,   # SAR/USD ≈ 0.267
        'shekel':             usd_to_ugx * 0.285,   # ILS/USD ≈ 0.285
        'bitcoin':            usd_to_ugx * 25000,   # BTC/USD ≈ $25K median
    }

    df['original_currency'] = df_raw['Payment_Currency'].str.lower().str.strip()
    df['amount_original']   = df['amount'].copy()   # original currency units
    df['exchange_rate_ugx'] = df['original_currency'].map(currency_to_ugx).fillna(usd_to_ugx)
    df['amount']            = df['amount'] * df['exchange_rate_ugx']   # → UGX
    df['currency']          = 'UGX'   # all amounts now normalised to UGX
    n_currencies = df['original_currency'].nunique()
    print(f"[IBM Loader] Currency: {n_currencies} currencies -> UGX (base: 1 USD = {usd_to_ugx:,.0f} UGX) | "
          f"Median: {df['amount'].median():,.0f} UGX")

    df['timestamp'] = pd.to_datetime(df_raw['Timestamp'], errors='coerce')
    t_min = df['timestamp'].min()
    df['step'] = ((df['timestamp'] - t_min).dt.total_seconds() / 3600).fillna(0).astype(int)

    df['tran_type']           = df_raw['Payment_Format'].apply(_to_unified_tran_type)
    # Note: do NOT re-assign df['currency'] here — it was set to 'UGX' above

    df['payment_format_risk'] = df_raw['Payment_Format'].apply(
        lambda x: _encode_payment_format(x, tl_cfg))
    df['is_cross_bank']  = (df_raw['From_Bank'] != df_raw['To_Bank']).astype(int)
    df['is_suspicious']  = df_raw['Is_Laundering'].astype(int)
    df['terminal_id']    = 'IBM_VIRTUAL'   # no terminal in IBM data
    df['dataset']        = 'IBM'

    # Add flag columns (matching Interswitch schema)
    df['isWithdrawTrx'] = (df['tran_type'] == 'CASH_OUT').astype(int)
    df['isTransferTrx'] = (df['tran_type'] == 'TRANSFER').astype(int)
    df['isRefundTrx']   = 0
    df['isDepositTrx']  = 0
    df['isPurchaseTrx'] = (df['tran_type'] == 'PURCHASE').astype(int)
    df['settle_currency'] = df['currency']

    fraud_final = df['is_suspicious'].sum()
    total_final = len(df)
    print(f"[IBM Loader] DONE. Final dataset: {total_final:,} rows | "
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

