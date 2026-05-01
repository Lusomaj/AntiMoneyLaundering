"""
XAI-SNA AML — Community Analyzer
Computes module-level quality metrics and homophily analysis on the transaction graph.

This module answers the question:
  "Are our detected communities structurally meaningful for AML?"

Key analyses:
  1. Modularity Score (Q): Quality of the Louvain partition
  2. Homophily Analysis: Are suspicious transactions disproportionately intra- or cross-community?
  3. Community Risk Profiling: Which communities have the highest fraud concentration?
  4. Infrastructure Clustering: Do terminals cluster within communities?

Outputs:
  - community_analysis.json: full analysis report
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from typing import Dict, Tuple

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


def compute_modularity(G: nx.DiGraph, community_map: dict) -> float:
    """
    Compute the modularity Q of the community partition.
    Q in [0, 1]: higher = more meaningful community structure.
    Q > 0.3 is considered good. Q > 0.6 is considered excellent.

    Uses NetworkX's modularity function on the undirected version of the graph.
    """
    if not community_map:
        return 0.0
    try:
        G_undir = G.to_undirected()
        # Convert community_map to list of sets as required by nx.community.modularity
        comm_sets = defaultdict(set)
        for node, comm_id in community_map.items():
            comm_sets[comm_id].add(node)
        communities = list(comm_sets.values())
        q = nx.community.modularity(G_undir, communities, weight='weight')
        return round(float(q), 4)
    except Exception as e:
        print(f"[CommunityAnalyzer] Modularity computation failed: {e}")
        return 0.0


def analyze_homophily(df: pd.DataFrame, sna_features: dict) -> Dict:
    """
    Homophily Analysis: Are suspicious transactions more common within or across communities?

    A cross-community edge (is_cross_community=1) that is suspicious suggests
    that laundering USES community boundaries as a structural trick (layering across clusters).
    An intra-community suspicious edge suggests internal collusion.

    Returns a dict with rates and odds ratio.
    """
    if 'is_suspicious' not in df.columns or 'is_cross_community' not in df.columns:
        return {'error': 'Required columns not available'}

    df_ana = df[['is_suspicious', 'is_cross_community']].copy()
    df_ana['is_suspicious'] = df_ana['is_suspicious'].astype(int)
    df_ana['is_cross_community'] = df_ana['is_cross_community'].astype(int)

    total         = len(df_ana)
    intra         = df_ana[df_ana['is_cross_community'] == 0]
    cross         = df_ana[df_ana['is_cross_community'] == 1]

    intra_susp_rate = intra['is_suspicious'].mean() if len(intra) > 0 else 0.0
    cross_susp_rate = cross['is_suspicious'].mean() if len(cross) > 0 else 0.0
    overall_rate    = df_ana['is_suspicious'].mean()

    # Odds ratio: how much more likely is a cross-community tx to be suspicious?
    odds_cross = cross_susp_rate / (1 - cross_susp_rate + 1e-9)
    odds_intra = intra_susp_rate / (1 - intra_susp_rate + 1e-9)
    odds_ratio = round(float(odds_cross / (odds_intra + 1e-9)), 4)

    # Chi-squared test for statistical significance
    chi2_stat, chi2_p = None, None
    try:
        from scipy import stats as scipy_stats
        ct = pd.crosstab(df_ana['is_cross_community'], df_ana['is_suspicious'])
        if ct.shape == (2, 2):
            chi2_stat, chi2_p, _, _ = scipy_stats.chi2_contingency(ct)
            chi2_stat = round(float(chi2_stat), 4)
            chi2_p    = round(float(chi2_p), 6)
    except Exception:
        pass

    return {
        'total_transactions':        total,
        'intra_community_count':     len(intra),
        'cross_community_count':     len(cross),
        'overall_fraud_rate':        round(float(overall_rate), 4),
        'intra_community_fraud_rate': round(float(intra_susp_rate), 4),
        'cross_community_fraud_rate': round(float(cross_susp_rate), 4),
        'odds_ratio_cross_vs_intra': odds_ratio,
        'chi2_statistic':            chi2_stat,
        'chi2_pvalue':               chi2_p,
        'significant':               bool(chi2_p < 0.05) if chi2_p is not None else None,
        'interpretation': (
            f"Cross-community transactions have a {cross_susp_rate:.2%} fraud rate vs "
            f"{intra_susp_rate:.2%} for intra-community. "
            f"Odds ratio: {odds_ratio:.2f}x — "
            f"{'cross-community transactions are MORE suspicious' if odds_ratio > 1 else 'intra-community transactions are more suspicious'}. "
            + (f"Chi-squared p={chi2_p:.4f} — "
               f"{'statistically significant' if chi2_p and chi2_p < 0.05 else 'not statistically significant at α=0.05'}"
               if chi2_p is not None else "")
        ),
    }


def profile_communities(df: pd.DataFrame, sna_features: dict, top_n: int = 20) -> Dict:
    """
    Community Risk Profiling: rank communities by fraud concentration.
    Identifies which communities are "laundering clusters".
    """
    if 'community_id' not in df.columns:
        # Try to populate from sna_features
        df = df.copy()
        df['community_id'] = df['source'].astype(str).map(
            lambda x: sna_features.get(x, {}).get('community_id', -1))

    if 'is_suspicious' not in df.columns:
        return {'error': 'is_suspicious column not available'}

    group = df.groupby('community_id').agg(
        total_tx       = ('is_suspicious', 'count'),
        suspicious_tx  = ('is_suspicious', 'sum'),
        total_amount   = ('amount', 'sum') if 'amount' in df.columns else ('is_suspicious', 'count'),
    ).reset_index()
    group['fraud_rate'] = group['suspicious_tx'] / group['total_tx'].clip(lower=1)
    group = group[group['total_tx'] >= 10]  # only communities with meaningful size
    group = group.sort_values('fraud_rate', ascending=False)

    top_communities = group.head(top_n)
    overall_rate = df['is_suspicious'].mean() if 'is_suspicious' in df.columns else 0

    laundering_clusters = top_communities[top_communities['fraud_rate'] > overall_rate * 2]

    return {
        'total_communities':       int(group['community_id'].nunique()),
        'community_profiles':      top_communities.to_dict(orient='records'),
        'laundering_cluster_count': len(laundering_clusters),
        'laundering_cluster_ids':  laundering_clusters['community_id'].tolist(),
        'overall_fraud_rate':      round(float(overall_rate), 4),
        'interpretation': (
            f"{len(laundering_clusters)} communities have fraud rates ≥2× the overall rate "
            f"({overall_rate:.2%}). These are potential 'laundering clusters' — tightly-knit "
            f"account groups used for coordinated layering."
        ),
    }


def analyze_infrastructure_clustering(G: nx.DiGraph, community_map: dict, sna_features: dict) -> Dict:
    """
    Check whether ATM/Agent terminals cluster within specific communities
    (suggesting complicit terminal networks).
    """
    terminal_communities = [
        community_map.get(node, -1)
        for node, data in G.nodes(data=True)
        if data.get('is_terminal', False) and node in community_map
    ]

    if not terminal_communities:
        return {'terminal_community_analysis': 'No terminal nodes found in graph.'}

    comm_counter = Counter(terminal_communities)
    top_terminal_comms = comm_counter.most_common(10)

    # Entropy: low entropy = terminals concentrated in few communities (suspicious)
    total = sum(comm_counter.values())
    probs = [c / total for c in comm_counter.values()]
    entropy = -sum(p * np.log2(p + 1e-9) for p in probs)
    max_entropy = np.log2(max(len(comm_counter), 1))
    normalized_entropy = round(entropy / max_entropy, 4) if max_entropy > 0 else 1.0

    return {
        'total_terminals':                  len(terminal_communities),
        'terminal_communities_count':       len(comm_counter),
        'top_terminal_communities':         [(int(c), int(n)) for c, n in top_terminal_comms],
        'concentration_entropy':            round(float(entropy), 4),
        'normalized_entropy':               normalized_entropy,
        'interpretation': (
            f"{len(terminal_communities)} terminals distributed across {len(comm_counter)} communities. "
            f"Normalized concentration entropy: {normalized_entropy:.2f} "
            f"({'low — terminals concentrated in few communities (potential complicit clusters)' if normalized_entropy < 0.5 else 'high — terminals spread across many communities (normal distribution)'})."
        ),
    }


def run_community_analysis(G: nx.DiGraph, df: pd.DataFrame, sna_features: dict,
                            community_map: dict, output_dir: str) -> dict:
    """
    Master function: run all community analyses and save to JSON.
    """
    print("[CommunityAnalyzer] Running community quality and homophily analysis...")

    modularity_q = compute_modularity(G, community_map)
    q_interpretation = (
        "Excellent community structure (Q > 0.6) — communities are highly cohesive."
        if modularity_q > 0.6 else
        "Good community structure (Q > 0.3) — communities are meaningful."
        if modularity_q > 0.3 else
        "Weak community structure (Q < 0.3) — communities may be arbitrary."
    )

    homophily = analyze_homophily(df, sna_features)
    profiles  = profile_communities(df, sna_features)
    infra     = analyze_infrastructure_clustering(G, community_map, sna_features)

    report = {
        'modularity_score':          modularity_q,
        'modularity_interpretation': q_interpretation,
        'modularity_context': {
            'excellent': '>= 0.6',
            'good':      '0.3 – 0.6',
            'weak':      '< 0.3',
        },
        'homophily_analysis':        homophily,
        'community_risk_profiles':   profiles,
        'infrastructure_clustering': infra,
        'summary': (
            f"Modularity Q={modularity_q:.3f} ({q_interpretation[:20]}). "
            f"Cross-community fraud rate: {homophily.get('cross_community_fraud_rate', 0):.2%} "
            f"vs intra: {homophily.get('intra_community_fraud_rate', 0):.2%} "
            f"(OR={homophily.get('odds_ratio_cross_vs_intra', 'N/A')}). "
            f"{profiles.get('laundering_cluster_count', 0)} laundering clusters identified."
        ),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'community_analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[CommunityAnalyzer] Modularity Q = {modularity_q:.4f} ({q_interpretation[:40]})")
    print(f"[CommunityAnalyzer] Cross-community fraud rate: {homophily.get('cross_community_fraud_rate', 0):.2%}")
    print(f"[CommunityAnalyzer] Community analysis saved → {out_path}")

    return report


def load_community_analysis(output_dir: str) -> dict:
    path = os.path.join(output_dir, 'community_analysis.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}

