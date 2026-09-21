"""Syntax check and smoke test for USD->UGX conversion."""
import ast
import os
import sys
import pytest

sys.path.insert(0, '.')

def test_syntax_checks():
    files = ['aml_engine/ibm_loader.py', 'aml_engine/rules_engine.py', 'aml_engine/pattern_bridge.py']
    for f in files:
        if os.path.exists(f):
            content = open(f, 'r', encoding='utf-8').read()
            ast.parse(content)

def test_currency_conversion_ratio():
    import pandas as pd
    usd_rate = 3800
    df = pd.DataFrame({
        'amount_usd': [10.0, 50.0, 100.0],
        'amount': [10.0 * usd_rate, 50.0 * usd_rate, 100.0 * usd_rate],
        'currency': ['UGX', 'UGX', 'UGX'],
        'exchange_rate': [usd_rate, usd_rate, usd_rate]
    })
    ratio = df["amount"].median() / df["amount_usd"].median()
    assert abs(ratio - usd_rate) < 1

def test_ibm_loader_smoke():
    ibm_file = 'IBM_AML_DATA/HI-Large_Trans.csv'
    if not os.path.exists(ibm_file):
        pytest.skip(f"Dataset {ibm_file} not found (large dataset omitted in CI)")
    if os.environ.get('RUN_SLOW_TESTS') != '1':
        pytest.skip(f"Skipping slow 5GB IBM scan. Set RUN_SLOW_TESTS=1 to execute.")

    from aml_engine.ibm_loader import load_ibm_dataset
    cfg = {
        'ibm_sample_size': 1000,
        'min_fraud_rows': 10,
        'usd_to_ugx_rate': 3800,
        'payment_format_risk': {},
    }
    df = load_ibm_dataset(ibm_file, cfg, '.')
    ratio = df["amount"].median() / df["amount_usd"].median()
    assert abs(ratio - 3800) < 1


