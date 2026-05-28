"""Syntax check and smoke test for USD->UGX conversion."""
import ast, os, sys

sys.path.insert(0, '.')

files = ['aml_engine/ibm_loader.py', 'aml_engine/rules_engine.py', 'aml_engine/pattern_bridge.py']
for f in files:
    ast.parse(open(f, 'r', encoding='utf-8').read())
    print(f'SYNTAX OK: {f}')

# Quick smoke test on loader conversion
from aml_engine.ibm_loader import load_ibm_dataset

cfg = {
    'ibm_sample_size': 5000,
    'min_fraud_rows': 100,
    'usd_to_ugx_rate': 3800,
    'payment_format_risk': {},
}
df = load_ibm_dataset('IBM_AML_DATA/HI-Large_Trans.csv', cfg, '.')
print()
print('=== Smoke Test Results ===')
print(f'  Rows loaded: {len(df):,} | Fraud: {int(df["is_suspicious"].sum()):,}')
print(f'  amount (UGX) median:  {df["amount"].median():,.0f}')
print(f'  amount_usd median:    {df["amount_usd"].median():,.2f}')
print(f'  currency col unique:  {list(df["currency"].unique())}')
print(f'  exchange_rate unique: {list(df["exchange_rate"].unique())}')
ratio = df["amount"].median() / df["amount_usd"].median()
print(f'  Ratio (should be ~3800): {ratio:.1f}')
print()
print('OK - currency conversion working correctly' if abs(ratio - 3800) < 1 else 'ERROR - ratio mismatch')
