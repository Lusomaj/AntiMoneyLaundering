"""Check pipeline state before continuing."""
import os, json
import pandas as pd
from datetime import datetime

def age(f):
    if not os.path.exists(f):
        return 'MISSING'
    m = os.path.getmtime(f)
    mins = (datetime.now().timestamp() - m) / 60
    h = mins / 60
    return f'{h:.1f}h ago' if h > 1 else f'{mins:.0f}min ago'

files = {
    'IBM model':    'data/models/ibm_best_model.pkl',
    'IBM results':  'data/models/ibm_model_comparison.csv',
    'ISW scored':   'data/processed/interswitch_scored.csv',
    'Bridge JSON':  'data/processed/pattern_bridge.json',
    'KPIs JSON':    'data/processed/operational_kpis.json',
    'IBM features': 'data/processed/ibm_features_full.csv',
}
for label, f in files.items():
    if os.path.exists(f):
        print(f'{label:20s}: {age(f)} | {os.path.getsize(f)/1e6:.1f}MB')
    else:
        print(f'{label:20s}: MISSING')

# Check if IBM features have UGX scale amounts
print()
if os.path.exists('data/processed/ibm_features_full.csv'):
    df = pd.read_csv('data/processed/ibm_features_full.csv', nrows=3)
    has_ugx_cols = 'original_currency' in df.columns or 'exchange_rate_ugx' in df.columns
    print(f'IBM features has UGX cols: {has_ugx_cols}')

    amt_df = pd.read_csv('data/processed/ibm_features_full.csv', usecols=['amount'], nrows=10000)
    med = amt_df['amount'].median()
    print(f'IBM amount median: {med:,.0f}')
    print(f'Scale status: {"UGX (>100K = already converted)" if med > 100000 else "USD (still needs conversion)"}')

# Check bridge
if os.path.exists('data/processed/pattern_bridge.json'):
    br = json.load(open('data/processed/pattern_bridge.json', encoding='utf-8'))
    print()
    print(f'Bridge similarity: {br.get("overall_similarity", "N/A")}')
    print(f'Bridge has amount_distribution: {"amount_distribution" in br}')
    print(f'Bridge has currency_normalisation: {"currency_normalisation" in br}')
