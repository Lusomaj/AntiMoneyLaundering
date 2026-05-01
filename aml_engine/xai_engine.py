"""
XAI-SNA AML — XAI Engine (Enhanced)
Computes and persists SHAP (SHapley Additive Explanations) values for the best model.

Enhancements over v1:
  - Serializes the SHAP explainer for live per-transaction scoring (fixes population-mean bug)
  - Calibration curve (Reliability Diagram) computation and persistence
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
    Saves values, figures, the serialized explainer, and calibration data to output_dir.
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

    # ── Serialize the explainer for live per-transaction scoring ──────
    # FIX: Previously explain_single_transaction() used population means.
    # Now we save the explainer so the dashboard can run it on any single tx.
    try:
        explainer_path = os.path.join(output_dir, 'shap_explainer.pkl')
        with open(explainer_path, 'wb') as _ef:
            pickle.dump({'explainer': explainer, 'explainer_type': explainer_type,
                         'expected_value': expected_value}, _ef)
        print(f"[XAI] SHAP explainer serialized → {explainer_path}")
    except Exception as e:
        print(f"[XAI] WARNING: Could not serialize explainer ({e}). Live SHAP will use mean fallback.")

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

    # ── Calibration Curve (Reliability Diagram) ───────────
    # This proves model probability scores are meaningful (not just ordinal ranks)
    if hasattr(model, 'predict_proba'):
        try:
            from sklearn.calibration import calibration_curve
            probs_cal = model.predict_proba(X_sc)[:, 1]
            y_dummy   = (probs_cal >= 0.5).astype(int)  # proxy labels from high-confidence predictions

            # Check if we have test labels available
            y_test_path = os.path.join(os.path.dirname(output_dir), 'models', 'y_test.npy')
            if os.path.exists(y_test_path):
                y_cal = np.load(y_test_path)[:max_samples]
                if len(y_cal) == len(probs_cal) and y_cal.sum() > 5:
                    n_bins = 8
                    fraction_of_positives, mean_predicted = calibration_curve(
                        y_cal, probs_cal, n_bins=n_bins, strategy='quantile'
                    )
                    cal_data = {
                        'mean_predicted_value': mean_predicted.tolist(),
                        'fraction_of_positives': fraction_of_positives.tolist(),
                        'source': 'y_test labels',
                        'brier_score': round(float(np.mean((probs_cal - y_cal) ** 2)), 4),
                    }
                    with open(os.path.join(output_dir, 'calibration_data.json'), 'w') as cf:
                        json.dump(cal_data, cf)
                    print(f"[XAI] Calibration curve saved (Brier score: {cal_data['brier_score']:.4f}).")
        except Exception as e:
            print(f"[XAI] Calibration curve error: {e}")

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


def explain_single_transaction(tx_features: np.ndarray, feature_names: list,
                                shap_dir: str, model_bundle: dict = None) -> dict:
    """
    Compute SHAP explanation for a SINGLE transaction.

    FIXED: Previously used population mean SHAP values (incorrect for live scoring).
    Now uses the serialized TreeExplainer for per-transaction attribution.
    Falls back to mean-based approximation if explainer is not available.

    Args:
        tx_features: 1D numpy array of feature values (same order as feature_names)
        feature_names: list of feature names
        shap_dir: directory containing shap_explainer.pkl
        model_bundle: optional model bundle for fallback

    Returns:
        dict with per-feature SHAP contributions + plain-English narrative
    """
    # Try to load serialized explainer (live/per-transaction scoring)
    explainer_path = os.path.join(shap_dir, 'shap_explainer.pkl')
    live_shap_available = False
    contribs = {}
    expected_value = 0.0

    if os.path.exists(explainer_path):
        try:
            with open(explainer_path, 'rb') as ef:
                exp_bundle = pickle.load(ef)
            explainer    = exp_bundle['explainer']
            expected_value = float(exp_bundle.get('expected_value', 0.0))
            explainer_type = exp_bundle.get('explainer_type', 'Unknown')

            tx_2d = tx_features.reshape(1, -1)

            if explainer_type == 'TreeExplainer':
                sv = explainer.shap_values(tx_2d)
                sv_row = (sv[1][0] if isinstance(sv, list) else sv[0])
            else:
                sv_row = explainer.shap_values(tx_2d, nsamples=100)
                if isinstance(sv_row, np.ndarray) and sv_row.ndim > 1:
                    sv_row = sv_row[0]

            contribs = dict(zip(feature_names, sv_row.tolist()))
            live_shap_available = True
            print(f"[XAI] Live per-transaction SHAP computed ({explainer_type}).")
        except Exception as e:
            print(f"[XAI] Live SHAP failed ({e}). Using population-mean fallback.")

    # Fallback: use population mean SHAP (approximate)
    if not live_shap_available:
        shap_vals_path = os.path.join(shap_dir, 'shap_values.npy')
        ev_path        = os.path.join(shap_dir, 'expected_value.npy')
        if os.path.exists(shap_vals_path):
            shap_vals  = np.load(shap_vals_path)
            expected_value = float(np.load(ev_path)[0]) if os.path.exists(ev_path) else 0.0
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
    if not live_shap_available:
        narrative += " *(Note: using population-mean SHAP approximation — run pipeline for exact values)*"

    return {
        'contributions':        contribs,
        'expected_value':       expected_value,
        'narrative':            narrative,
        'feature_descriptions': FEATURE_DESCRIPTIONS,
        'is_live_shap':         live_shap_available,
    }


def load_feature_importance(shap_dir: str) -> pd.DataFrame:
    path = os.path.join(shap_dir, 'feature_importance.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_calibration_data(shap_dir: str) -> dict:
    """Load calibration curve data if available."""
    path = os.path.join(shap_dir, 'calibration_data.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


import json  # ensure json is available at module level

