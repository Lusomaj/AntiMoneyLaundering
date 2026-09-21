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
    'R8_structuring':     'Structuring / Smurfing (Just Below Threshold)',
    'R9_passthrough':     'Pass-Through / Transit Account Activity',
    'R10_mule_terminal':  'High-Risk Terminal / Mule Aggregation',
    'R11_card_testing':   'Card Testing / Probing Velocity',
    'R12_off_peak':       'Off-Peak High-Value Transaction',
    'R13_high_risk_currency': 'High-Risk Currency Flow',
    'R14_dormant_spike':  'Dormant Account Sudden Spike (>90d, >5x mean)',
}


def apply_rules(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply all hard rules to each transaction row.
    Returns df with columns:
        rule_triggered: 1 if any rule fires
        rule_score:     count of rules fired (0–14)
        rule_flags:     JSON-like string listing which rules fired
    """
    rules = cfg['hard_rules']
    threshold_ugx = rules['amount_threshold_ugx']           # UGX (Interswitch)
    threshold_usd = rules.get('amount_threshold_usd', 5000)  # USD (IBM) — default ~18M UGX
    high_types  = [t.upper() for t in rules['high_risk_transaction_types']]
    vel_max     = rules['velocity_max_tx_per_hour']
    term_thresh = rules['terminal_high_tx_threshold']
    
    struct_thresh_pct = rules.get('structuring_threshold_percent', 0.90)
    struct_min_cnt = rules.get('structuring_min_count', 2)
    term_cards_max = rules.get('terminal_max_unique_cards_per_day', 5)
    failed_tx_cnt = rules.get('max_failed_tx_count', 3)
    off_peak_hrs = rules.get('off_peak_hours', [0,1,2,3,4,5])
    off_peak_mult = rules.get('off_peak_amount_multiplier', 0.5)
    hr_currencies = [c.upper() for c in rules.get('high_risk_currencies', [])]

    print("[RulesEngine] Applying hard rules...")

    # ── R1: Amount threshold (all amounts now normalised to UGX) ────
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

    # ── R8: Structuring
    r8 = pd.Series(0, index=df.index)
    if 'source' in df.columns and 'amount' in df.columns:
        struct_mask = (df['amount'] >= (threshold_ugx * struct_thresh_pct)) & (df['amount'] < threshold_ugx)
        struct_counts = df[struct_mask].groupby('source')['amount'].transform('count')
        df_struct_cnt = pd.Series(0, index=df.index)
        df_struct_cnt.loc[struct_mask] = struct_counts
        r8 = (df_struct_cnt >= struct_min_cnt).astype(int)

    # ── R9: Passthrough
    r9 = df.get('motif_passthrough', pd.Series(0, index=df.index)).astype(int) 

    # ── R10: Mule Terminal
    r10 = pd.Series(0, index=df.index)
    if 'terminal_id' in df.columns and 'step' in df.columns:
        term_cards = df.groupby(['terminal_id', 'step'])['source'].transform('nunique')
        r10 = (term_cards >= term_cards_max).astype(int)
        
    # ── R11: Card Testing
    r11 = pd.Series(0, index=df.index)
    if 'transactionResponse' in df.columns:
        failed_mask = (df['transactionResponse'] != 'APPROVED') & (df['transactionResponse'] != 'SUCCESS')
        bad_sources = df[failed_mask].groupby('source').size()
        bad_sources = bad_sources[bad_sources >= failed_tx_cnt].index
        r11 = df['source'].isin(bad_sources).astype(int)

    # ── R12: Off-Peak
    r12 = pd.Series(0, index=df.index)
    if 'hour' in df.columns:
        r12 = (df['hour'].isin(off_peak_hrs) & (df['amount'] >= (threshold_ugx * off_peak_mult))).astype(int)
    elif 'step' in df.columns:
        r12 = ((df['step'] % 24).isin(off_peak_hrs) & (df['amount'] >= (threshold_ugx * off_peak_mult))).astype(int)

    # ── R13: High-Risk Currency
    r13 = pd.Series(0, index=df.index)
    if 'currency' in df.columns:
        r13 = df['currency'].str.upper().isin(hr_currencies).astype(int)
    elif 'settle_currency_code' in df.columns:
        r13 = df['settle_currency_code'].str.upper().isin(hr_currencies).astype(int)

    # ── R14: Dormant Account Spike (RUL-L1-07)
    r14 = df.get('is_dormant_spike', pd.Series(0, index=df.index)).astype(int)

    rule_matrix = pd.DataFrame({
        'R1_threshold':      r1,
        'R2_high_risk_type': r2,
        'R3_reversal':       r3,
        'R4_smurfing':       r4,
        'R5_circular':       r5,
        'R6_velocity':       r6,
        'R7_terminal_burst': r7,
        'R8_structuring':    r8,
        'R9_passthrough':    r9,
        'R10_mule_terminal': r10,
        'R11_card_testing':  r11,
        'R12_off_peak':      r12,
        'R13_high_risk_currency': r13,
        'R14_dormant_spike': r14,
    })

    df['rule_score']     = rule_matrix.sum(axis=1)
    df['rule_triggered'] = (df['rule_score'] >= 1).astype(int)

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


def apply_composite_rules(df: pd.DataFrame, ml_probs: np.ndarray,
                           high_risk_cutoff: float = 0.75,
                           auto_clear_cutoff: float = 0.15) -> pd.DataFrame:
    """
    Layer 4: Composite Rules Engine (Interswitch AML Rulebook Spec)
    
    Rules:
      - RUL-L4-01 (Tier-1 Critical Escalation):
          Condition: Layer1_Flags > 0 AND Layer2_Flags > 0 AND P_Model >= 0.75
          Action: High certainty multi-vector operation -> Auto-SAR Draft Generation
          
      - RUL-L4-02 (Low-Risk False-Positive Auto-Clear):
          Condition: rule_score == 1 AND P_Model < 0.15
          Action: Legitimate transaction -> Auto-clear alert with audit trail, 
                  bypassing human review.
    """
    probs = np.asarray(ml_probs)
    rule_score = df['rule_score'] if 'rule_score' in df.columns else df.get('rule_triggered', pd.Series(0, index=df.index))
    
    # Layer 2 flags: motifs (circular, smurfing, reversal) or high SNA centrality
    l2_flags = (
        (df.get('motif_circular', pd.Series(0, index=df.index)) > 0) |
        (df.get('motif_smurfing', pd.Series(0, index=df.index)) > 0) |
        (df.get('motif_reversal', pd.Series(0, index=df.index)) > 0) |
        (df.get('source_degree_cent', pd.Series(0, index=df.index)) > 0.05) |
        (df.get('source_betweenness', pd.Series(0, index=df.index)) > 0.05)
    ).astype(int)
    
    l1_flags = (rule_score > 0).astype(int)
    
    # RUL-L4-01: Critical Escalation
    r_l4_01 = ((l1_flags == 1) & (l2_flags == 1) & (probs >= high_risk_cutoff)).astype(int)
    
    # RUL-L4-02: FP Auto-Clear (exactly 1 rule triggered, low ML prob)
    r_l4_02 = ((rule_score == 1) & (probs < auto_clear_cutoff)).astype(int)
    
    df['tier1_critical_escalation'] = r_l4_01
    df['auto_cleared']              = r_l4_02
    
    status = pd.Series('NORMAL', index=df.index)
    status[l1_flags == 1] = 'STANDARD_REVIEW'
    status[r_l4_02 == 1]  = 'AUTO_CLEARED'
    status[r_l4_01 == 1]  = 'CRITICAL_ESCALATION'
    df['composite_disposition'] = status
    
    n_clear = int(r_l4_02.sum())
    n_crit  = int(r_l4_01.sum())
    print(f"[CompositeRules] Layer 4 Applied — Auto-Cleared: {n_clear:,} | Critical Escalations: {n_crit:,}")
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
    results['R8_structuring']    = 0
    results['R9_passthrough']    = int(tx.get('motif_passthrough', 0))
    results['R10_mule_terminal'] = 0
    results['R11_card_testing']  = 0
    
    # Live off-peak
    hr = tx.get('hour', tx.get('step', 0) % 24)
    results['R12_off_peak'] = int((hr in rules.get('off_peak_hours', [])) and (tx.get('amount', 0) >= threshold * rules.get('off_peak_amount_multiplier', 0.5)))
    
    # Live currency
    curr = str(tx.get('currency', tx.get('settle_currency_code', ''))).upper()
    results['R13_high_risk_currency'] = int(curr in [c.upper() for c in rules.get('high_risk_currencies', [])])

    # Live dormant spike
    results['R14_dormant_spike'] = int(tx.get('is_dormant_spike', 0))

    results['rule_score']     = sum(results[k] for k in results)
    results['rule_triggered'] = int(results['rule_score'] >= 1)
    results['rule_flags']     = {k: RULE_LABELS[k] for k in RULE_LABELS if results.get(k, 0) == 1}

    return results

