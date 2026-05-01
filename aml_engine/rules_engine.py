"""
XAI-SNA AML — Rules Engine
Executes the "Hard Rules" compliance layer (Layer 1 of the Tri-Layer Defense).
Reads configuration from aml_config.yaml — no code changes needed to update rules.

Rules evaluated per transaction:
  R1: Amount above threshold
  R2: High-risk transaction type
  R3: Rapid reversal motif flagged
  R4: Smurfing fan-out flagged
  R5: Circular flow flagged
  R6: Velocity breach (too many tx in current step)
  R7: Terminal burst (terminal used by many accounts)
"""

import yaml
import os
import pandas as pd
import numpy as np
from typing import Tuple


CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'aml_config.yaml')


def load_config(config_path: str = None) -> dict:
    path = config_path or CONFIG_PATH
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


RULE_LABELS = {
    'R1_threshold':       'Amount Exceeds Reporting Threshold',
    'R2_high_risk_type':  'High-Risk Transaction Type',
    'R3_reversal':        'Rapid Reversal Pattern Detected',
    'R4_smurfing':        'Smurfing / Fan-Out Pattern Detected',
    'R5_circular':        'Circular Flow / Layering Detected',
    'R6_velocity':        'Velocity Breach — Too Many Transactions',
    'R7_terminal_burst':  'Terminal Hub Activity Spike',
}


def apply_rules(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply all hard rules to each transaction row.
    Returns df with columns:
        rule_triggered: 1 if any rule fires
        rule_score:     count of rules fired (0–7)
        rule_flags:     JSON-like string listing which rules fired
    """
    rules = cfg['hard_rules']
    threshold_ugx = rules['amount_threshold_ugx']           # UGX (Interswitch)
    threshold_usd = rules.get('amount_threshold_usd', 5000)  # USD (IBM) — default ~18M UGX
    high_types  = [t.upper() for t in rules['high_risk_transaction_types']]
    vel_max     = rules['velocity_max_tx_per_hour']
    term_thresh = rules['terminal_high_tx_threshold']

    print("[RulesEngine] Applying hard rules...")

    # ── R1: Amount threshold (all amounts now normalised to UGX) ────
    # IBM amounts are converted to UGX in ibm_loader.py before reaching here.
    # Interswitch amounts are natively in UGX.
    # Both datasets now use the same UGX threshold — no currency branching needed.
    # This is the key benefit of the USD→UGX normalisation step.
    r1 = (df['amount'] >= threshold_ugx).astype(int)
    r2 = df['tran_type'].str.upper().isin(high_types).astype(int)
    r3 = df.get('motif_reversal', pd.Series(0, index=df.index)).astype(int)
    r4 = df.get('motif_smurfing', pd.Series(0, index=df.index)).astype(int)
    r5 = df.get('motif_circular', pd.Series(0, index=df.index)).astype(int)
    r6 = (df.get('tx_count_last_step', pd.Series(0, index=df.index)) >= vel_max).astype(int)

    # Terminal burst: count how many distinct sources use each terminal in this step
    if 'terminal_id' in df.columns and 'step' in df.columns:
        term_burst = df.groupby(['terminal_id', 'step'])['source'].transform('nunique')
        r7 = (term_burst >= term_thresh).astype(int)
    else:
        r7 = pd.Series(0, index=df.index)

    rule_matrix = pd.DataFrame({
        'R1_threshold':      r1,
        'R2_high_risk_type': r2,
        'R3_reversal':       r3,
        'R4_smurfing':       r4,
        'R5_circular':       r5,
        'R6_velocity':       r6,
        'R7_terminal_burst': r7,
    })

    df['rule_score']     = rule_matrix.sum(axis=1)
    df['rule_triggered'] = (df['rule_score'] >= 1).astype(int)

    def get_flags(row):
        flags = [RULE_LABELS[k] for k in RULE_LABELS if row.get(k, 0) == 1]
        return ' | '.join(flags) if flags else 'None'

    # Attach individual rule columns for explainability
    for col in rule_matrix.columns:
        df[col] = rule_matrix[col]

    df['rule_flags'] = rule_matrix.apply(
        lambda row: ' | '.join([RULE_LABELS[k] for k in RULE_LABELS if row.get(k, 0) == 1]) or 'None',
        axis=1
    )

    triggered = df['rule_triggered'].sum()
    print(f"[RulesEngine] Rules applied — Triggered: {triggered:,} / {len(df):,} "
          f"({100*triggered/len(df):.2f}%)")
    return df


def evaluate_single_transaction(tx: dict, cfg: dict) -> dict:
    """
    Evaluate a single transaction dict against all rules.
    Used by the Live Detection tab in the dashboard.
    Returns rule results as a dict.
    """
    rules = cfg['hard_rules']
    threshold  = rules['amount_threshold_ugx']
    high_types = [t.upper() for t in rules['high_risk_transaction_types']]
    vel_max    = rules['velocity_max_tx_per_hour']

    results = {}
    results['R1_threshold']      = int(tx.get('amount', 0) >= threshold)
    results['R2_high_risk_type'] = int(str(tx.get('tran_type', '')).upper() in high_types)
    results['R3_reversal']       = int(tx.get('motif_reversal', 0))
    results['R4_smurfing']       = int(tx.get('motif_smurfing', 0))
    results['R5_circular']       = int(tx.get('motif_circular', 0))
    results['R6_velocity']       = int(tx.get('tx_count_last_step', 0) >= vel_max)
    results['R7_terminal_burst'] = 0  # cannot compute from single tx without graph context

    results['rule_score']     = sum(results[k] for k in results)
    results['rule_triggered'] = int(results['rule_score'] >= 1)
    results['rule_flags']     = {k: RULE_LABELS[k] for k in RULE_LABELS if results.get(k, 0) == 1}

    return results

