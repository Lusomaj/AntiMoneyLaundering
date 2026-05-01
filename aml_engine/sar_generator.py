"""
XAI-SNA AML — Suspicious Activity Report (SAR) Generator
Generates a professional PDF Suspicious Activity Report for flagged transactions.

Produces a formatted PDF containing:
  - Report header (institution, date, reference number)
  - Transaction summary table (flagged transactions)
  - Rule violation details
  - SHAP explanation narrative per transaction
  - Risk score + percentile rank
  - Network community membership
  - FATF compliance notes

Usage:
    from aml_engine.sar_generator import generate_sar_pdf
    pdf_bytes = generate_sar_pdf(flagged_df, kpis, cfg)
"""

import io
import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Optional


def generate_sar_pdf(flagged_df: pd.DataFrame,
                     kpis: dict,
                     cfg: dict,
                     shap_explanations: Optional[list] = None,
                     institution_name: str = "XAI-SNA AML System",
                     max_transactions: int = 20) -> bytes:
    """
    Generate a PDF Suspicious Activity Report (SAR).

    Args:
        flagged_df: DataFrame of flagged/suspicious transactions
        kpis: operational KPI report dict
        cfg: aml_config.yaml dict
        shap_explanations: optional list of per-transaction SHAP narratives
        institution_name: institution name for the report header
        max_transactions: cap on transactions included in the SAR

    Returns:
        bytes: PDF file content (can be sent as Streamlit download)
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        REPORTLAB_AVAILABLE = True
    except ImportError:
        REPORTLAB_AVAILABLE = False

    if not REPORTLAB_AVAILABLE:
        # Fallback: generate a plain-text "PDF" (really just formatted text as bytes)
        return _generate_text_sar(flagged_df, kpis, cfg, shap_explanations,
                                   institution_name, max_transactions)

    # ── PDF Document Setup ────────────────────────────────
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Suspicious Activity Report (SAR)"
    )

    styles = getSampleStyleSheet()
    story  = []

    # Custom styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
                                  fontSize=16, spaceAfter=6,
                                  textColor=colors.HexColor('#1e3a5f'))
    sub_style   = ParagraphStyle('SubStyle', parent=styles['Normal'],
                                  fontSize=10, spaceAfter=4,
                                  textColor=colors.HexColor('#4b5563'))
    head_style  = ParagraphStyle('HeadStyle', parent=styles['Heading2'],
                                  fontSize=12, textColor=colors.HexColor('#1d4ed8'),
                                  spaceBefore=12, spaceAfter=6)
    body_style  = ParagraphStyle('BodyStyle', parent=styles['Normal'],
                                  fontSize=9, spaceAfter=3)
    warn_style  = ParagraphStyle('WarnStyle', parent=styles['Normal'],
                                  fontSize=9, textColor=colors.HexColor('#dc2626'),
                                  spaceAfter=3)

    now    = datetime.datetime.now()
    ref_no = f"SAR-{now.strftime('%Y%m%d%H%M%S')}-AML"

    # ── HEADER ────────────────────────────────────────────
    story.append(Paragraph("SUSPICIOUS ACTIVITY REPORT (SAR)", title_style))
    story.append(Paragraph(f"XAI-SNA AML System | {institution_name}", sub_style))
    story.append(HRFlowable(width="100%", thickness=2,
                              color=colors.HexColor('#1d4ed8'), spaceAfter=8))

    header_data = [
        ['Reference No.', ref_no,          'Generated',    now.strftime('%Y-%m-%d %H:%M UTC')],
        ['System',        'XAI-SNA v3', 'Dataset',     kpis.get('dataset', 'Interswitch Uganda')],
        ['Total Flagged', f"{len(flagged_df):,}", 'KPIs Passed', f"{kpis.get('kpis_met',0)}/3"],
        ['Threshold',     f"UGX {cfg.get('hard_rules',{}).get('amount_threshold_ugx',10000000):,}",
         'Threshold Source', kpis.get('threshold_source', 'F1-optimal')],
    ]

    header_table = Table(header_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eff6ff')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#eff6ff')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.3*cm))

    # ── EXECUTIVE SUMMARY ─────────────────────────────────
    story.append(Paragraph("1. Executive Summary", head_style))
    kpi1 = kpis.get('kpi_1_fp_reduction', {})
    kpi3 = kpis.get('kpi_3_explainability', {})
    adv  = kpis.get('adversarial_robustness', {})

    story.append(Paragraph(
        f"The XAI-SNA system flagged <b>{len(flagged_df):,}</b> transactions for "
        f"suspicious activity review. The ML layer filtered "
        f"<b>{kpi1.get('fp_reduction_rate', 0):.1%}</b> of rule-only alerts (FP Reduction KPI). "
        f"SHAP explainability covers <b>{kpi3.get('mean_top3_coverage', 0):.1%}</b> of alert SHAP "
        f"magnitude via top-3 features (XAI KPI). "
        f"Adversarial ML recall: <b>{adv.get('adversarial_ml_recall', 'N/A')}</b> — the model "
        f"remains effective even when rule thresholds are bypassed. "
        f"F1-optimal detection threshold: <b>{kpis.get('threshold_used', 0.5):.4f}</b>.",
        body_style
    ))

    verdict = kpis.get('operational_verdict', '')
    if verdict:
        story.append(Paragraph(verdict, body_style))

    story.append(Spacer(1, 0.2*cm))

    # ── OPERATIONAL KPI SUMMARY ───────────────────────────
    story.append(Paragraph("2. Operational KPI Summary", head_style))
    kpi2 = kpis.get('kpi_2_latency', {})
    psi  = kpis.get('psi_drift_analysis', {})

    kpi_data = [
        ['KPI', 'Result', 'Target', 'Status'],
        ['FP Reduction vs Rules-Only',
         f"{kpi1.get('fp_reduction_rate', 0):.1%}",
         '≥ 30%',
         '✓ PASS' if kpi1.get('fp_reduction_rate', 0) >= 0.30 else '✗ FAIL'],
        ['Inference Latency (100 tx)',
         f"{kpi2.get('mean_latency_ms', 0):.1f} ms",
         '≤ 500ms',
         '✓ PASS' if kpi2.get('meets_target') else '✗ FAIL'],
        ['SHAP Top-3 Coverage',
         f"{kpi3.get('mean_top3_coverage', 0):.1%}",
         '≥ 70%',
         '✓ PASS' if kpi3.get('meets_target') else '✗ FAIL'],
        ['Model Drift (PSI)',
         psi.get('overall_avg_psi', 'N/A'),
         '< 0.10',
         psi.get('overall_status', 'N/A')],
        ['Adversarial ML Recall',
         str(adv.get('adversarial_ml_recall', 'N/A')),
         '≥ 50%',
         '✓ ROBUST' if adv.get('meets_robustness_target') else '⚠ BRITTLE'],
    ]

    kpi_table = Table(kpi_data, colWidths=[7*cm, 3*cm, 2.5*cm, 4*cm])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1d4ed8')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.3*cm))

    # ── FLAGGED TRANSACTION TABLE ─────────────────────────
    story.append(Paragraph("3. Flagged Transactions", head_style))
    story.append(Paragraph(
        f"Showing top {min(max_transactions, len(flagged_df))} highest-risk transactions "
        f"(sorted by ML risk score, descending):", body_style))

    df_top = flagged_df.copy()
    if 'ml_risk_score' in df_top.columns:
        df_top = df_top.sort_values('ml_risk_score', ascending=False)
    df_top = df_top.head(max_transactions)

    tx_cols_avail = [c for c in ['source', 'target', 'amount', 'tran_type',
                                   'ml_risk_score', 'rule_flags', 'community_id'] if c in df_top.columns]
    tx_cols_labels = {
        'source': 'Source', 'target': 'Target',
        'amount': 'Amount', 'tran_type': 'Type',
        'ml_risk_score': 'ML Risk', 'rule_flags': 'Rule Violations',
        'community_id': 'Community'
    }

    tx_header = [tx_cols_labels.get(c, c) for c in tx_cols_avail]
    tx_data   = [tx_header]
    for _, row in df_top.iterrows():
        tx_row = []
        for c in tx_cols_avail:
            val = row.get(c, '')
            if c == 'amount':
                tx_row.append(f"{float(val):,.0f}" if val else '—')
            elif c == 'ml_risk_score':
                tx_row.append(f"{float(val):.2%}" if val else '—')
            elif c in ('source', 'target'):
                tx_row.append(str(val)[:12])
            elif c == 'rule_flags':
                tx_row.append(str(val)[:40] if val else 'None')
            else:
                tx_row.append(str(val)[:15])
        tx_data.append(tx_row)

    col_widths = [max(1.5*cm, 17*cm / len(tx_cols_avail))] * len(tx_cols_avail)
    tx_table = Table(tx_data, colWidths=col_widths, repeatRows=1)
    tx_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#d1d5db')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef2f2')]),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('WORDWRAP', (0, 0), (-1, -1), True),
    ]))
    story.append(tx_table)
    story.append(Spacer(1, 0.3*cm))

    # ── SHAP EXPLANATIONS ─────────────────────────────────
    if shap_explanations:
        story.append(Paragraph("4. SHAP Explainability Narratives", head_style))
        story.append(Paragraph(
            "The following narratives explain WHY each transaction was flagged, "
            "derived from SHAP (SHapley Additive Explanations) values:", body_style))

        for i, explanation in enumerate(shap_explanations[:10]):
            narrative = explanation if isinstance(explanation, str) else str(explanation)
            story.append(Paragraph(f"[TX {i+1}]: {narrative}", body_style))
            story.append(Spacer(1, 0.1*cm))

    # ── FATF COMPLIANCE NOTE ──────────────────────────────
    story.append(Paragraph("5. FATF Regulatory Compliance Notes", head_style))
    rules = cfg.get('hard_rules', {})
    story.append(Paragraph(
        f"This report is generated in accordance with FATF Recommendation 20 "
        f"(Reporting of Suspicious Transactions) and Recommendation 10 (Record Keeping). "
        f"Active thresholds: Amount ≥ UGX {rules.get('amount_threshold_ugx', 10000000):,} | "
        f"Velocity ≥ {rules.get('velocity_max_tx_per_hour', 10)} tx/hr | "
        f"Reversal window: {rules.get('rapid_reversal_window_minutes', 15)} minutes | "
        f"Fan-out threshold: {rules.get('smurfing_fan_out_count', 5)} unique targets.",
        body_style
    ))
    story.append(Paragraph(
        "All thresholds are maintained in aml_config.yaml — the FATF-auditable "
        "single source of truth for this system. Rule changes are version-controlled.",
        body_style
    ))
    story.append(Spacer(1, 0.3*cm))

    # ── FOOTER ────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                              color=colors.HexColor('#d1d5db'), spaceAfter=4))
    story.append(Paragraph(
        f"CONFIDENTIAL — Generated by XAI-SNA AML System v3.0 | "
        f"Makerere University M.Sc. Dissertation | Joseph Lusoma | {ref_no}",
        ParagraphStyle('footer', parent=styles['Normal'], fontSize=7,
                        textColor=colors.HexColor('#9ca3af'), alignment=TA_CENTER)
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _generate_text_sar(flagged_df, kpis, cfg, shap_explanations,
                        institution_name, max_transactions) -> bytes:
    """
    Fallback SAR generator when reportlab is not installed.
    Produces a structured plain-text report.
    """
    now    = datetime.datetime.now()
    ref_no = f"SAR-{now.strftime('%Y%m%d%H%M%S')}-AML"

    lines = [
        "=" * 70,
        "SUSPICIOUS ACTIVITY REPORT (SAR)",
        f"XAI-SNA AML System | {institution_name}",
        "=" * 70,
        f"Reference:  {ref_no}",
        f"Generated:  {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Flagged Tx: {len(flagged_df):,}",
        f"KPIs Passed:{kpis.get('kpis_met', 0)}/3",
        "",
        "OPERATIONAL KPIs",
        "-" * 40,
    ]

    kpi1 = kpis.get('kpi_1_fp_reduction', {})
    kpi2 = kpis.get('kpi_2_latency', {})
    kpi3 = kpis.get('kpi_3_explainability', {})
    lines += [
        f"FP Reduction:  {kpi1.get('fp_reduction_rate', 0):.1%}  (target ≥30%)",
        f"Latency:       {kpi2.get('mean_latency_ms', 0):.1f}ms  (target ≤500ms)",
        f"SHAP Coverage: {kpi3.get('mean_top3_coverage', 0):.1%}  (target ≥70%)",
        "",
        "TOP FLAGGED TRANSACTIONS",
        "-" * 40,
    ]

    df_top = flagged_df.head(max_transactions)
    for idx, (_, row) in enumerate(df_top.iterrows()):
        lines.append(
            f"[{idx+1:02d}] Source: {str(row.get('source',''))[:12]:<12} | "
            f"Target: {str(row.get('target',''))[:12]:<12} | "
            f"Amt: {float(row.get('amount', 0)):>12,.0f} | "
            f"Risk: {float(row.get('ml_risk_score', 0)):.2%}"
        )

    lines += [
        "",
        "FATF COMPLIANCE NOTE",
        "-" * 40,
        f"Threshold: UGX {cfg.get('hard_rules',{}).get('amount_threshold_ugx',10000000):,}",
        "Config: aml_config.yaml (FATF-auditable single source of truth)",
        "",
        "=" * 70,
        f"CONFIDENTIAL | XAI-SNA AML v3.0 | {ref_no}",
        "=" * 70,
    ]

    return "\n".join(lines).encode('utf-8')

