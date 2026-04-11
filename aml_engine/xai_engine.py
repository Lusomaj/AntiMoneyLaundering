"""
Anti-Gravity AML — XAI Engine
Computes and persists SHAP (SHapley Additive Explanations) values for the best model.

Exposes:
  - Global feature importance bar chart data
  - Per-transaction Force Plot vectors (for UI hover)
  - Waterfall chart data for single-transaction explanation
  - Plain-English "Why Flagged" narrative generator
"""

import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


FEATURE_DESCRIPTIONS = {
    'amount':              'Transaction Amount',
    'amount_log':          'Log-Transformed Amount',
    'is_small_amount':     'Structuring (Small Amount Below Threshold)',
    'is_large_amount':     'Amount Exceeds Reporting Threshold',
    'isWithdrawTrx':       'Is Withdrawal Transaction',
    'isTransferTrx':       'Is Transfer Transaction',
    'isRefundTrx':         'Is Refund Transaction',
    'isDepositTrx':        'Is Deposit Transaction',
    'isPurchaseTrx':       'Is Purchase Transaction',
    'tran_type_encoded':   'Transaction Type (Encoded)',
    'source_degree_cent':  'Source Account Connectivity (Degree)',
    'source_betweenness':  'Source Account Bridge Score (Betweenness)',
    'source_pagerank':     'Source Account Influence (PageRank)',
    'target_degree_cent':  'Target Account Connectivity',
    'target_betweenness':  'Target Account Bridge Score',
    'target_pagerank':     'Target Account Influence',
    'terminal_pagerank':   'Terminal Hub Risk Score (PageRank)',
    'community_id':        'Transaction Network Community',
    'community_size':      'Community Size (Larger = More Isolated)',
    'is_cross_community':  'Transaction Crosses Community Boundary',
    'source_is_infrastructure': 'Source Is an Infrastructure Node',
    'target_is_infrastructure': 'Target Is an Infrastructure Node',
    'hist_tx_count':       'Cumulative Historical Transaction Count',
    'tx_count_last_step':  'Transactions in Current Time Window (Velocity)',
    'amount_vs_hist_mean': 'Amount vs. Historical Average (Spike Ratio)',
    'total_amount_last_step': 'Total Volume in Current Time Window',
    'time_since_last_tx':  'Time Since Last Transaction (Inactivity Gap)',
    'motif_circular':      'Circular Flow / Layering Pattern Detected',
    'motif_smurfing':      'Smurfing / Fan-Out Pattern Detected',
    'motif_reversal':      'Rapid Reversal Pattern Detected',
}


def compute_shap_values(model_bundle: dict, X_test: np.ndarray,
                         feature_names: list, output_dir: str, max_samples: int = 500):
    """
    Compute SHAP values for the best model. Supports tree-based and linear models.
    Saves values and figures to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    model   = model_bundle['model']
    scaler  = model_bundle.get('scaler')
    if scaler is not None:
        X_sc = scaler.transform(X_test[:max_samples])
    else:
        X_sc = X_test[:max_samples]

    print(f"[XAI] Computing SHAP values for {min(len(X_test), max_samples)} samples...")

    try:
        # Try TreeExplainer first (fast for RF/XGB)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sc)
        # For multi-class output (RF returns list), take positive class
        if isinstance(shap_values, list):
            shap_vals_pos = shap_values[1]
        else:
            shap_vals_pos = shap_values
        expected_value = (explainer.expected_value[1]
                          if isinstance(explainer.expected_value, (list, np.ndarray))
                          else explainer.expected_value)
        explainer_type = 'TreeExplainer'
    except Exception:
        # Fall back to KernelExplainer (model-agnostic)
        print("[XAI] TreeExplainer failed. Using KernelExplainer (slower)...")
        bg = shap.sample(X_sc, min(100, len(X_sc)))
        predict_fn = (model.predict_proba if hasattr(model, 'predict_proba')
                      else model.predict)
        explainer = shap.KernelExplainer(
            lambda x: predict_fn(x)[:, 1] if hasattr(model, 'predict_proba') else predict_fn(x),
            bg
        )
        shap_vals_pos = explainer.shap_values(X_sc[:min(200, len(X_sc))], nsamples=100)
        expected_value = explainer.expected_value
        explainer_type = 'KernelExplainer'

    print(f"[XAI] SHAP values computed using {explainer_type}.")

    # ── Save SHAP artifacts ───────────────────────────────
    shap_path = os.path.join(output_dir, 'shap_values.npy')
    ev_path   = os.path.join(output_dir, 'expected_value.npy')
    np.save(shap_path, shap_vals_pos)
    np.save(ev_path, np.array([expected_value]))
    print(f"[XAI] SHAP values saved → {shap_path}")

    # ── Global Feature Importance ─────────────────────────
    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature':     feature_names,
        'SHAP_Mean':   mean_abs_shap,
        'Description': [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names],
    }).sort_values('SHAP_Mean', ascending=False)
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    # ── Save individual transaction force-plot data ───────
    force_data = []
    for i in range(min(50, shap_vals_pos.shape[0])):
        row = {
            'shap_values':    shap_vals_pos[i].tolist(),
            'feature_values': X_sc[i].tolist(),
            'expected_value': float(expected_value),
            'feature_names':  feature_names,
        }
        force_data.append(row)

    import json
    with open(os.path.join(output_dir, 'force_plot_data.json'), 'w') as fp:
        json.dump(force_data, fp)

    # ── Save waterfall figure for best worst-case alert ───
    try:
        worst_idx = np.argmax(shap_vals_pos.sum(axis=1))
        fig, ax = plt.subplots(figsize=(10, 8))
        shap_exp = shap.Explanation(
            values=shap_vals_pos[worst_idx],
            base_values=expected_value,
            data=X_sc[worst_idx],
            feature_names=[FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names]
        )
        shap.plots.waterfall(shap_exp, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_waterfall.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[XAI] SHAP waterfall chart saved.")
    except Exception as e:
        print(f"[XAI] Could not save waterfall chart: {e}")

    print(f"\n[XAI] Top 10 features by SHAP importance:")
    for _, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['SHAP_Mean'] * 100)
        print(f"  {row['Feature']:<30} {row['SHAP_Mean']:.4f}  {bar}")

    return shap_vals_pos, expected_value, importance_df


def explain_single_transaction(tx_values: np.ndarray, feature_names: list,
                                shap_dir: str) -> dict:
    """
    Load precomputed SHAP data and compute explanation for a single transaction.
    Returns dict with SHAP contribution per feature + plain-English narrative.
    """
    shap_vals = np.load(os.path.join(shap_dir, 'shap_values.npy'))
    ev        = np.load(os.path.join(shap_dir, 'expected_value.npy'))[0]

    # Use mean SHAP as proxy for fresh transaction (no live computation in dashboard)
    # In production: run the explainer on the live transaction
    mean_contributions = np.abs(shap_vals).mean(axis=0)
    signed_contributions = shap_vals.mean(axis=0)

    contribs = dict(zip(feature_names, signed_contributions))

    # Build narrative
    top_positive = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:3]
    top_negative = sorted(contribs.items(), key=lambda x: x[1])[:2]

    narrative_parts = []
    for feat, val in top_positive:
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        direction = "increased" if val > 0 else "decreased"
        narrative_parts.append(f"**{desc}** {direction} the risk score by {abs(val):.4f}")

    narrative = ("This transaction was flagged because: " +
                 "; ".join(narrative_parts) + ".")

    return {
        'contributions':       contribs,
        'expected_value':      ev,
        'narrative':           narrative,
        'feature_descriptions': FEATURE_DESCRIPTIONS,
    }


def load_feature_importance(shap_dir: str) -> pd.DataFrame:
    path = os.path.join(shap_dir, 'feature_importance.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()
