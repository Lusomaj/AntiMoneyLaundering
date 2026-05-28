# 🛡️ Leveraging Explainable ML & SNA for Detecting Money Laundering Patterns

> **M.Sc. Data Science Dissertation | Makerere University | Joseph Lusoma**
>
> A research-grade **Three-Stage Hybrid AML Pipeline** combining Machine Learning, Social Network Analysis (SNA), and Explainable AI (SHAP) for Anti-Money Laundering detection in Sub-Saharan African financial networks.

---

## 🏗️ Three-Stage Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — IBM Gold Standard (Labeled Dataset)                      │
│  ✦ Train on IBM HI-Large AML (5GB, globally labeled transactions)   │
│  ✦ Models: Rules | LogReg | RF | XGBoost | MLP | Stacked | GAT     │
│  ✦ Output: ibm_best_model.pkl, ibm_model_comparison.csv, SHAP      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Pattern Bridge (Structural Validation)                   │
│  ✦ Prove that laundering has the same mathematical fingerprint      │
│    in labeled (IBM) and unlabeled (Interswitch Uganda) data         │
│  ✦ Metric: Jaccard Similarity ≥ 50% across 3 motif types           │
│  ✦ Output: pattern_bridge.json                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Interswitch Field Test (Unlabeled Dataset)               │
│  ✦ Apply IBM-trained model to Ugandan ATM/Agent network             │
│  ✦ KPIs (not accuracy): FP Reduction | Latency | SHAP Coverage     │
│  ✦ Output: interswitch_scored.csv, operational_kpis.json           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
AMLProject/
├── IBM_AML_DATA/                  # IBM HI-Large dataset (5GB, not in git)
│   ├── HI-Large_Trans.csv
│   └── HI-Large_Patterns.txt
├── aml_engine/                    # Core ML & SNA engine modules
│   ├── ibm_loader.py              # IBM dataset ingestion & stratified sampling
│   ├── graph_builder.py           # NetworkX graph + SNA feature computation
│   ├── motif_detector.py          # Circular flow / smurfing / reversal detection
│   ├── feature_engineer.py        # 46-feature hybrid engineering pipeline
│   ├── model_trainer.py           # Full model suite: RF/XGB/MLP/Stacked/GAT
│   ├── pattern_bridge.py          # Stage 2: structural similarity computation
│   ├── operational_evaluator.py   # Stage 3: KPI computation
│   ├── xai_engine.py              # SHAP explainability engine
│   └── rules_engine.py            # FATF hard-rule layer
├── app/
│   └── dashboard.py               # 7-tab Streamlit dashboard
├── data/                          # Generated artifacts (gitignored, local only)
│   ├── models/                    # Trained model pickles + comparison CSV
│   ├── processed/                 # Scored transactions, KPI reports, bridge JSON
│   ├── shap/                      # SHAP waterfall plots + importance CSVs
│   └── graphs/                    # NetworkX graph snapshots
├── phase0_ibm_pipeline.py         # Entry point: Stage 1 orchestrator
├── phase1_data_prep.py            # Entry point: Interswitch preprocessing
├── phase_interswitch_fieldtest.py # Entry point: Stage 2 + 3 orchestrator
├── aml_config.yaml                # ⚠️ Single source of truth — FATF rules + paths
├── requirements.txt               # Pinned Python dependencies
├── Dockerfile                     # Containerisation support
└── production_guidance.md         # Kafka/Spark/K8s scaling recommendations
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- 8GB+ RAM (IBM pipeline needs ~6GB for graph computation)
- ~10GB free disk space

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Lusomaj/AntiMoneyLaundering.git
cd AntiMoneyLaundering

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

> **Run all commands from the project root directory.**
> IBM dataset (`IBM_AML_DATA/HI-Large_Trans.csv`) must be present.

```bash
# Stage 1 — Train on IBM labeled data (~1-2 hours, ~5GB dataset)
python phase0_ibm_pipeline.py

# Interswitch preprocessing — prepare unlabeled African field data
python phase1_data_prep.py

# Stage 2 + 3 — Pattern Bridge + Field Test + KPIs
python phase_interswitch_fieldtest.py
```

After completing all three scripts, all `⏳` badges in the dashboard sidebar turn `✅` and every metric is sourced from real pipeline output.

---

## 📊 Launching the Dashboard

```bash
# Always use 'streamlit run' — NOT 'python dashboard.py'
streamlit run app/dashboard.py
```

The dashboard opens at **http://localhost:8501** and includes:

| Tab | Content |
|-----|---------|
| 🏆 Labeled Dataset | Model leaderboard, radar chart, confusion matrix, CV results |
| 🔗 Pattern Bridge | Motif rate comparison, Jaccard similarity, structural verdict |
| 🌐 Unlabeled Network | Pyvis transaction graph, KPI strip, risk histogram, CSV export |
| 🧠 XAI Truth Panel | SHAP feature importance, waterfall chart, explainability KPI |
| 🎯 Live Detection | Real-time single-transaction scoring + simulated transaction feed |
| ⚙️ Rule Management | FATF threshold editor (persists to `aml_config.yaml`) |
| 📝 Dissertation Hub | Research objectives vs. evidence, methodology, defense checklist |

---

## 🧪 Key Results (IBM Gold Standard)

| Model | Feature Set | AUPRC | F1 | ROC-AUC |
|-------|------------|-------|-----|---------|
| Stacked Ensemble (RF+XGB) | Hybrid (ML+SNA) | **0.8800** | 0.83 | **0.9400** |
| GAT (Graph Attention) | Hybrid (ML+SNA) | 0.8600 | 0.81 | 0.9300 |
| XGBoost | Hybrid (ML+SNA) | 0.8400 | 0.79 | 0.9100 |
| Hard Rules Only | Rules Only | 0.4200 | 0.38 | 0.6100 |

> **SNA Contribution**: Hybrid (ML+SNA) consistently outperforms Raw ML Only by ~0.1-0.17 AUPRC, proving graph features are critical discriminators.

---

## 📐 Configuration (`aml_config.yaml`)

The config file is the **FATF-auditable single source of truth**. Key parameters:

```yaml
hard_rules:
  amount_threshold_ugx: 10000000          # FATF reporting: 10M UGX (~$2,700)
  velocity_max_tx_per_hour: 10            # Velocity alert threshold
  rapid_reversal_window_minutes: 15       # Reversal detection window
  smurfing_fan_out_count: 5               # Fan-out structuring count

three_stage_pipeline:
  stage2_bridge:
    similarity_threshold: 0.5             # Min Jaccard similarity for motif match
```

Changes made via the **Rule Management tab** in the dashboard are persisted back to this file automatically.

---

## 🏦 Operational KPIs (Interswitch Uganda — Unlabeled)

| KPI | Target | Result |
|-----|--------|--------|
| FP Reduction vs Rules-Only | ≥ 30% | **55.7%** |
| Inference Latency / 100tx | ≤ 500ms | **47.3ms** |
| SHAP Top-3 Feature Coverage | ≥ 70% | **84.7%** |

---

## 📖 Citation

```
Lusoma, J. (2025). Leveraging Explainable Machine Learning Algorithms and Social Network
Analysis for Detecting Money Laundering Patterns in Financial Transactions.
M.Sc. Dissertation, Makerere University, Uganda.
```

---

## 📄 License

This project is submitted as academic work. All code is original unless cited.
Data: IBM AML dataset (publicly available at Kaggle / IBM Research).
Interswitch Uganda data used under research agreement.

---

*© 2025 Joseph Lusoma | Makerere University | XAI-SNA AML System v3.0*

