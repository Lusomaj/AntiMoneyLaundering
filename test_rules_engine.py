"""Unit tests for AML Rules Engine."""
import pandas as pd
import pytest
from aml_engine.rules_engine import load_config, apply_rules


def test_load_config():
    cfg = load_config()
    assert cfg is not None
    assert 'hard_rules' in cfg
    assert 'amount_threshold_ugx' in cfg['hard_rules']


def test_apply_rules_threshold():
    cfg = load_config()
    threshold = cfg['hard_rules']['amount_threshold_ugx']

    # Create synthetic test transactions: one below threshold, one above
    df = pd.DataFrame([
        {
            'tran_id': 'TX1',
            'amount': threshold - 1000,
            'tran_type': 'TRANSFER',
            'step': 1,
            'source': 'ACC_001',
            'terminal_id': 'TERM_01',
        },
        {
            'tran_id': 'TX2',
            'amount': threshold + 50000,
            'tran_type': 'TRANSFER',
            'step': 1,
            'source': 'ACC_002',
            'terminal_id': 'TERM_01',
        }
    ])

    result_df = apply_rules(df, cfg)
    assert 'rule_triggered' in result_df.columns
    assert 'rule_score' in result_df.columns
    assert 'R1_threshold' in result_df.columns

    # TX1 shouldn't trigger R1, TX2 should trigger R1
    assert result_df.loc[0, 'R1_threshold'] == 0
    assert result_df.loc[1, 'R1_threshold'] == 1
    assert result_df.loc[1, 'rule_triggered'] == 1
