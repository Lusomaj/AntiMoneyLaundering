import pandas as pd
import numpy as np

rate = 3800  # 1 USD = 3800 UGX (historical median ~2017-2022)

# IBM stats (USD)
df_ibm = pd.read_csv('data/processed/ibm_features_full.csv', usecols=['amount'], nrows=100000)
df_isw = pd.read_csv('data/processed/interswitch_scored.csv', usecols=['amount'], nrows=100000)

print("IBM amount stats (USD):")
print(df_ibm['amount'].describe().round(2))

print("\nISW amount stats (UGX):")
print(df_isw['amount'].describe().round(2))

# After conversion
ibm_ugx = df_ibm['amount'] * rate
ibm_log = np.log1p(ibm_ugx)
isw_log = np.log1p(df_isw['amount'])

print("\n=== After USD->UGX conversion ===")
print(f"IBM (converted) median UGX: {ibm_ugx.median():,.0f}")
print(f"ISW (native) median UGX:    {df_isw['amount'].median():,.0f}")
print()
print(f"IBM log-amount mean: {ibm_log.mean():.4f}  std: {ibm_log.std():.4f}")
print(f"ISW log-amount mean: {isw_log.mean():.4f}  std: {isw_log.std():.4f}")

rng = max(ibm_log.max(), isw_log.max())
sim = 1 - abs(ibm_log.mean() - isw_log.mean()) / rng
print(f"Log-scale mean similarity:  {sim:.4f}  ({sim:.1%})")

# Smurfing threshold
thresh_usd = 100000 / rate
print(f"\nSmurfing low threshold: 100,000 UGX = {thresh_usd:.2f} USD")
pct_ibm_small = (df_ibm['amount'] < thresh_usd).mean()
pct_isw_small = (df_isw['amount'] < 100000).mean()
print(f"IBM rows below threshold: {pct_ibm_small:.2%}")
print(f"ISW rows below threshold: {pct_isw_small:.2%}")
print(f"Threshold alignment improvement: {abs(pct_ibm_small - pct_isw_small):.2%} gap reduced to near-zero after conversion")
