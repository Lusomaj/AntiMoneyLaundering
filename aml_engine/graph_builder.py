"""
Anti-Gravity AML — Graph Builder
Constructs a directed transaction graph and computes SNA metrics:
  - Degree Centrality (in/out)
  - Betweenness Centrality (approximate, k=500)
  - PageRank (treats terminal_id nodes as infrastructure hubs)
  - Louvain Community Detection
  - Infrastructure Node marking for terminal_id nodes
"""

import networkx as nx
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("[GraphBuilder] WARNING: python-louvain not installed. Using greedy modularity fallback.")


def build_graph(df: pd.DataFrame, cfg: dict) -> nx.DiGraph:
    """
    Build a directed weighted transaction graph.
    Terminal IDs are added as special Infrastructure Nodes.
    """
    print("[GraphBuilder] Building transaction graph...")
    sample_cfg = cfg.get('sampling', {})
    graph_sample = min(sample_cfg.get('sna_sample_size', 150000), len(df))

    df_sample = df.head(graph_sample).copy()

    G = nx.DiGraph()

    # Add account-to-account edges
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Building edges"):
        src = str(row['source'])
        tgt = str(row['target'])
        amt = float(row['amount'])
        terminal = str(row.get('terminal_id', 'UNKNOWN'))
        ttype = str(row.get('tran_type', 'UNKNOWN'))
        is_susp = int(row.get('is_suspicious', 0))

        # Account → Account edge
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += amt
            G[src][tgt]['count'] += 1
        else:
            G.add_edge(src, tgt, weight=amt, count=1, tran_type=ttype)

        # Mark node attributes
        G.nodes[src]['node_type'] = 'account'
        G.nodes[tgt]['node_type'] = 'account'

        # Account → Terminal edge (infrastructure link)
        if terminal and terminal != 'MOMO_VIRTUAL':
            if G.has_edge(src, terminal):
                G[src][terminal]['weight'] += amt
                G[src][terminal]['count'] += 1
            else:
                G.add_edge(src, terminal, weight=amt, count=1, tran_type='TERMINAL_USE')
            G.nodes[terminal]['node_type'] = 'infrastructure'
            G.nodes[terminal]['is_terminal'] = True

    # Mark any remaining nodes
    for node in G.nodes:
        if 'node_type' not in G.nodes[node]:
            G.nodes[node]['node_type'] = 'account'
        if 'is_terminal' not in G.nodes[node]:
            G.nodes[node]['is_terminal'] = False

    print(f"[GraphBuilder] Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def compute_sna_features(G: nx.DiGraph, cfg: dict) -> dict:
    """
    Compute all SNA features. Returns dict: {node_id → feature_dict}
    """
    print("[GraphBuilder] Computing Degree Centrality...")
    in_degree  = dict(G.in_degree(weight='weight'))
    out_degree = dict(G.out_degree(weight='weight'))
    deg_cent   = nx.degree_centrality(G)

    print("[GraphBuilder] Computing PageRank (infrastructure-aware)...")
    try:
        pagerank = nx.pagerank(G, weight='weight', max_iter=100, tol=1e-4)
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes}

    print("[GraphBuilder] Computing Betweenness Centrality (k=500 approx)...")
    try:
        n_nodes = G.number_of_nodes()
        k_approx = min(500, n_nodes - 1) if n_nodes > 2 else None
        between_cent = nx.betweenness_centrality(G, k=k_approx, weight='weight', normalized=True)
    except Exception:
        between_cent = {n: 0.0 for n in G.nodes}

    print("[GraphBuilder] Computing Community Detection (Louvain)...")
    G_undirected = G.to_undirected()
    community_map = {}
    if LOUVAIN_AVAILABLE:
        try:
            partition = community_louvain.best_partition(G_undirected, weight='weight')
            community_map = partition
        except Exception as e:
            print(f"[GraphBuilder] Louvain failed: {e}. Using greedy fallback.")
    if not community_map:
        try:
            communities = list(nx.community.greedy_modularity_communities(G_undirected, weight='weight'))
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i
        except Exception:
            community_map = {n: 0 for n in G.nodes}

    # Build community size map
    from collections import Counter
    comm_sizes = Counter(community_map.values())

    # Assemble final per-node feature dict
    sna_features = {}
    for node in G.nodes:
        comm_id = community_map.get(node, -1)
        sna_features[node] = {
            'degree_centrality':    deg_cent.get(node, 0.0),
            'in_degree':            in_degree.get(node, 0),
            'out_degree':           out_degree.get(node, 0),
            'betweenness_centrality': between_cent.get(node, 0.0),
            'pagerank':             pagerank.get(node, 0.0),
            'community_id':         comm_id,
            'community_size':       comm_sizes.get(comm_id, 1),
            'is_infrastructure':    int(G.nodes[node].get('is_terminal', False)),
        }

    print(f"[GraphBuilder] SNA features computed for {len(sna_features):,} nodes.")
    return sna_features


def save_graph(G: nx.DiGraph, sna_features: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, 'transaction_graph.gpickle')
    sna_path   = os.path.join(output_dir, 'sna_features.pkl')
    nx.write_gpickle(G, graph_path) if hasattr(nx, 'write_gpickle') else pickle.dump(G, open(graph_path, 'wb'))
    with open(sna_path, 'wb') as f:
        pickle.dump(sna_features, f)
    print(f"[GraphBuilder] Graph saved → {graph_path}")
    print(f"[GraphBuilder] SNA features saved → {sna_path}")


def load_graph_artifacts(output_dir: str):
    import pickle
    graph_path = os.path.join(output_dir, 'transaction_graph.gpickle')
    sna_path   = os.path.join(output_dir, 'sna_features.pkl')
    try:
        G = pickle.load(open(graph_path, 'rb'))
    except Exception:
        G = None
    with open(sna_path, 'rb') as f:
        sna_features = pickle.load(f)
    return G, sna_features


def get_graph_stats(G: nx.DiGraph) -> dict:
    """Return high-level graph statistics for dashboard display."""
    if G is None:
        return {}
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_infrastructure_nodes': sum(1 for n in G.nodes if G.nodes[n].get('is_terminal', False)),
        'density': nx.density(G),
        'avg_in_degree': np.mean([d for _, d in G.in_degree()]) if G.number_of_nodes() > 0 else 0,
    }
