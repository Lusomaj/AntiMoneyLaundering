"""
XAI-SNA AML — Motif Detector
Detects three canonical money laundering motifs in the transaction graph:

  1. CIRCULAR FLOW  : A → B → C → ... → A  (cyclic layering)
  2. FAN-OUT (SMURFING): 1 source → N distinct targets in a short window
  3. RAPID REVERSAL : A → B  followed by B → A within a time window

Returns per-node binary flags for each motif, usable as ML features.
"""

import networkx as nx
import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict
from typing import Dict, Set, Tuple


# ─────────────────────────────────────────────────────────
# 1. CIRCULAR FLOW DETECTION
# ─────────────────────────────────────────────────────────

def detect_circular_flows(G: nx.DiGraph, max_depth: int = 4) -> Tuple[Set[str], list]:
    """
    Detect nodes involved in circular transaction flows (cycles).
    Uses Johnson's algorithm (limited by max_depth for performance).

    Returns:
        nodes_in_cycles: set of node IDs involved in any cycle
        all_cycles: list of detected cycles (lists of nodes)
    """
    print(f"[MotifDetector] Scanning for circular flows (max_depth={max_depth})...")
    nodes_in_cycles: Set[str] = set()
    all_cycles = []

    # For very large graphs, sample a subgraph of highest-degree nodes
    if G.number_of_nodes() > 5000:
        # Focus on high-betweenness nodes (most likely in laundering clusters)
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:3000]
        sub_G = G.subgraph(top_nodes).copy()
    else:
        sub_G = G

    try:
        for cycle in nx.simple_cycles(sub_G):
            if 2 < len(cycle) <= max_depth:
                all_cycles.append(cycle)
                nodes_in_cycles.update(cycle)
                if len(all_cycles) >= 500:  # cap for performance
                    break
    except Exception as e:
        print(f"[MotifDetector] Cycle detection error: {e}")

    print(f"[MotifDetector] Circular flows found: {len(all_cycles):,} | Nodes affected: {len(nodes_in_cycles):,}")
    return nodes_in_cycles, all_cycles


# ─────────────────────────────────────────────────────────
# 2. FAN-OUT / SMURFING DETECTION
# ─────────────────────────────────────────────────────────

def detect_smurfing(df: pd.DataFrame, cfg: dict) -> Set[str]:
    """
    Fan-out smurfing: one source sends to N+ distinct targets within a time window.
    Works on the raw transaction dataframe (requires 'step' as proxy for time).

    Returns:
        smurfing_sources: set of source node IDs flagged for smurfing
    """
    rules = cfg['hard_rules']
    fan_out_count  = rules['smurfing_fan_out_count']
    window_hours   = rules['smurfing_window_hours']

    print(f"[MotifDetector] Scanning for smurfing (fan-out≥{fan_out_count}, window={window_hours}h)...")

    # Treat each step as a time bucket; window = window_hours relative steps
    steps_per_hour = max(df['step'].max() / 24, 1) if df['step'].max() > 0 else 1
    window_steps   = int(window_hours * steps_per_hour)

    smurfing_sources: Set[str] = set()

    grouped = df.groupby('source')
    for source, grp in grouped:
        grp_sorted = grp.sort_values('step')
        # Sliding window: count unique targets in window
        steps = grp_sorted['step'].values
        targets = grp_sorted['target'].values
        for i in range(len(steps)):
            window_mask = (steps >= steps[i]) & (steps <= steps[i] + window_steps)
            unique_targets = len(set(targets[window_mask]))
            if unique_targets >= fan_out_count:
                smurfing_sources.add(str(source))
                break

    print(f"[MotifDetector] Smurfing sources detected: {len(smurfing_sources):,}")
    return smurfing_sources


# ─────────────────────────────────────────────────────────
# 3. RAPID REVERSAL DETECTION
# ─────────────────────────────────────────────────────────

def detect_rapid_reversals(df: pd.DataFrame, cfg: dict) -> Set[str]:
    """
    Rapid reversal: A sends to B, then B sends back to A within a time window.
    Suggests wash transactions or round-trip layering.

    Returns:
        reversing_nodes: set of node IDs involved in a reversal pair
    """
    rules = cfg['hard_rules']
    window_min = rules['rapid_reversal_window_minutes']

    print(f"[MotifDetector] Scanning for rapid reversals (window={window_min}min)...")

    # Convert step to proxy minutes (assume 1 step = 1 minute for MoMo sim)
    reversing_nodes: Set[str] = set()

    # Build a lookup: (A, B) → list of steps where A sent to B
    send_log: Dict[Tuple, list] = defaultdict(list)
    for _, row in df.iterrows():
        src = str(row['source'])
        tgt = str(row['target'])
        step = int(row['step'])
        send_log[(src, tgt)].append(step)

    # For every (A, B) pair, check if (B, A) exists within window
    for (src, tgt), steps_fwd in send_log.items():
        steps_rev = send_log.get((tgt, src), [])
        if not steps_rev:
            continue
        for sf in steps_fwd:
            for sr in steps_rev:
                if 0 < abs(sr - sf) <= window_min:
                    reversing_nodes.add(src)
                    reversing_nodes.add(tgt)
                    break

    print(f"[MotifDetector] Reversal nodes detected: {len(reversing_nodes):,}")
    return reversing_nodes


# ─────────────────────────────────────────────────────────
# MASTER: Apply all motifs and return feature columns
# ─────────────────────────────────────────────────────────

def apply_motif_features(df: pd.DataFrame, G: nx.DiGraph, cfg: dict) -> pd.DataFrame:
    """
    Run all three motif detectors and add binary feature columns to df.
    Columns added:
        motif_circular  : 1 if source node is involved in a circular flow
        motif_smurfing  : 1 if source node is a smurfing source
        motif_reversal  : 1 if source or target node is in a reversal pair
    """
    max_depth = cfg['hard_rules']['circular_flow_max_depth']

    circular_nodes, cycles  = detect_circular_flows(G, max_depth=max_depth)
    smurfing_sources        = detect_smurfing(df, cfg)
    reversal_nodes          = detect_rapid_reversals(df, cfg)

    df['motif_circular'] = df['source'].astype(str).isin(circular_nodes).astype(int)
    df['motif_smurfing'] = df['source'].astype(str).isin(smurfing_sources).astype(int)
    df['motif_reversal'] = (
        df['source'].astype(str).isin(reversal_nodes) |
        df['target'].astype(str).isin(reversal_nodes)
    ).astype(int)

    print(f"\n[MotifDetector] Motif feature summary:")
    print(f"  motif_circular : {df['motif_circular'].sum():,} transactions affected")
    print(f"  motif_smurfing : {df['motif_smurfing'].sum():,} transactions affected")
    print(f"  motif_reversal : {df['motif_reversal'].sum():,} transactions affected")

    # Store cycle metadata for UI visualization
    return df, {
        'circular_nodes': list(circular_nodes)[:200],
        'cycles':         [c for c in cycles[:50]],
        'smurfing_nodes': list(smurfing_sources)[:200],
        'reversal_nodes': list(reversal_nodes)[:200],
    }

