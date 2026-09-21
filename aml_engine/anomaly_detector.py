"""
XAI-SNA AML — Topology Isolation Forest (Unsupervised Graph Anomaly Detector)

Provides an unsupervised defense against adversarial amount manipulation (e.g., structuring).
Evaluates transactions purely on network topological features (centrality, degree, clustering,
community boundaries, velocity) with ZERO dependency on monetary amount.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


TOPOLOGY_FEATURES = [
    'source_degree_cent', 'target_degree_cent',
    'source_pagerank', 'target_pagerank',
    'is_cross_community',
    'source_in_degree', 'source_out_degree',
    'target_in_degree', 'target_out_degree',
    'source_flow_through_ratio',
    'source_clustering_coef', 'target_clustering_coef',
    'time_since_last_tx', 'tx_count_last_step',
]


class TopologyIsolationForest:
    """
    Unsupervised Graph Topology Anomaly Detector.
    Detects structural network anomalies invariant to transaction amounts.
    """

    def __init__(self, contamination: float = 0.03, random_state: int = 42, n_estimators: int = 100):
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.feature_names = TOPOLOGY_FEATURES
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.min_score_ = 0.0
        self.max_score_ = 1.0
        self.threshold_ = 0.5

    def _extract_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and fill topology features."""
        feats = [f for f in self.feature_names if f in df.columns]
        if len(feats) < len(self.feature_names):
            # Fill missing columns with 0.0
            data = {}
            for f in self.feature_names:
                data[f] = df[f].values if f in df.columns else np.zeros(len(df))
            X = pd.DataFrame(data)[self.feature_names].fillna(0).values
        else:
            X = df[self.feature_names].fillna(0).values
        return X

    def fit(self, df: pd.DataFrame, max_samples: Optional[int] = 100_000) -> 'TopologyIsolationForest':
        """Fit Isolation Forest on graph topology."""
        print(f"[TopologyIForest] Fitting on {len(df):,} transactions using {len(self.feature_names)} topology features...")
        X = self._extract_matrix(df)
        if max_samples and len(X) > max_samples:
            idx = np.random.RandomState(self.random_state).choice(len(X), max_samples, replace=False)
            X_fit = X[idx]
        else:
            X_fit = X

        X_sc = self.scaler.fit_transform(X_fit)
        self.model.fit(X_sc)
        self.is_fitted = True

        # Calibrate normalized score bounds
        raw_scores = -self.model.score_samples(X_sc) # Higher = more anomalous
        self.min_score_ = float(raw_scores.min())
        self.max_score_ = float(raw_scores.max())
        self.threshold_ = float(np.percentile(raw_scores, (1 - self.contamination) * 100))
        print(f"[TopologyIForest] Fitted successfully. Anomaly threshold (P{100*(1-self.contamination):.0f}): {self.threshold_:.4f}")
        return self

    def score(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Compute anomaly scores for given dataframe.
        Returns:
            dict with 'scores' (normalized 0-1) and 'is_anomaly' (0 or 1).
        """
        if not self.is_fitted:
            raise RuntimeError("TopologyIsolationForest must be fitted before scoring.")

        X = self._extract_matrix(df)
        X_sc = self.scaler.transform(X)
        raw_scores = -self.model.score_samples(X_sc)

        # Normalize to 0-1 range
        norm_scores = (raw_scores - self.min_score_) / (self.max_score_ - self.min_score_ + 1e-12)
        norm_scores = np.clip(norm_scores, 0.0, 1.0)

        # Decision based on contamination threshold
        is_anomaly = (raw_scores >= self.threshold_).astype(int)

        return {
            'scores': norm_scores,
            'is_anomaly': is_anomaly,
            'raw_scores': raw_scores,
        }

    def save(self, filepath: str):
        """Save model bundle to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'threshold_': self.threshold_,
            'min_score_': self.min_score_,
            'max_score_': self.max_score_,
            'is_fitted': self.is_fitted,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"[TopologyIForest] Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TopologyIsolationForest':
        """Load model bundle from disk."""
        with open(filepath, 'rb') as f:
            bundle = pickle.load(f)
        obj = cls(contamination=bundle['contamination'])
        obj.model = bundle['model']
        obj.scaler = bundle['scaler']
        obj.feature_names = bundle['feature_names']
        obj.threshold_ = bundle['threshold_']
        obj.min_score_ = bundle['min_score_']
        obj.max_score_ = bundle['max_score_']
        obj.is_fitted = bundle['is_fitted']
        print(f"[TopologyIForest] Loaded model from {filepath}")
        return obj
