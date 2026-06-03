# ── app.py — Late Delivery Risk Predictor ────────────────────────────
# Run with: python -m streamlit run app.py

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Late Delivery Risk",
    page_icon  = "🚚",
    layout     = "wide",
)

# Tighten default Streamlit padding + style the radio/select boxes yellow
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    h1 { margin-top: 0 !important; padding-top: 0 !important; }

    /* Yellow shading for radio + selectbox to match the insight callout */
    div[role="radiogroup"] {
        background: #fffbeb;
        border: 1px solid #f3e3b3;
        border-radius: 8px;
        padding: 8px 12px;
    }
    div[data-baseweb="select"] > div {
        background: #fffbeb !important;
        border-color: #f3e3b3 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts + data ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('model.pkl')
    schema = json.load(open('schema.json'))
    return model, schema

@st.cache_data
@st.cache_data
def load_overview_data():
    return pd.read_parquet('overview_data.parquet')

model, schema = load_artifacts()
overview_df   = load_overview_data()

# ── Header ───────────────────────────────────────────────────────────
st.markdown("### 🚚 Late Delivery Risk Predictor")
st.caption("DataCo Global Supply Chain · ~180K historical orders")

# ── Page selector + slice selector, side by side ────────────────────
sel_col1, sel_col2 = st.columns([1, 1])

with sel_col1:
    st.markdown("**Page**")
    view = st.radio(
        "Page",
        ["📊 Overview", "🔍 One-pager (single order)"],
        horizontal=True,
        label_visibility="collapsed",
    )

with sel_col2:
    if view == "📊 Overview":
        st.markdown("**Slice by**")
        dim = st.selectbox(
            "Slice by",
            options=schema['categorical_cols'],
            index=0,
            label_visibility="collapsed",
        )
    else:
        dim = None

st.markdown("")


# ── Helper function for rendering a slice section ───────────────────
def render_slice_section(df, group_col, title):
    """Render a section header + table + chart for a given grouping column."""
    st.markdown(
        f"<p style='font-size:14px; font-weight:600; margin:0 0 8px; color:#374151;'>"
        f"{title}</p>",
        unsafe_allow_html=True,
    )

    summary = (
        df.groupby(group_col)['Late_delivery_risk']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'Late rate', 'count': 'Orders'})
        .sort_values('Late rate', ascending=False)
        .reset_index()
    )
    summary['Late rate'] = (summary['Late rate'] * 100).round(1)

    def flag(rate):
        if rate >= 70: return "🔴 High"
        elif rate >= 40: return "🟡 Medium"
        else: return "🟢 Low"
    summary['Risk band'] = summary['Late rate'].apply(flag)
    summary = summary[[group_col, 'Late rate', 'Risk band', 'Orders']]

    st.dataframe(summary, use_container_width=True, hide_index=True)

    top_n = summary.head(15) if len(summary) > 15 else summary
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(top_n) * 0.32)))
    colors = [
        '#C41230' if r >= 70 else '#BA7517' if r >= 40 else '#3B6D11'
        for r in top_n['Late rate']
    ]
    ax.barh(top_n[group_col].astype(str), top_n['Late rate'], color=colors)
    ax.axvline(50, color='grey', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel("Late delivery rate (%)", fontsize=9)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════
#  VIEW 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
if view == "📊 Overview":

    st.subheader("Executive summary")

    # ── Business problem statement ─────────────────────────────────
    st.markdown(
        """
        <div style='background:#eff6ff; border-left:4px solid #185FA5; padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0 16px;'>
            <p style='font-size:13px; margin:0; color:#374151; line-height:1.6;'>
                <strong>Business problem:</strong> predict the probability of late delivery
                <em>at order time</em>, so logistics managers can prioritise interventions
                before shipments fail.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"{len(overview_df):,} historical orders · "
        "Late = arrived after scheduled delivery date"
    )

    # ── Pre-compute numbers for KPI cards ────────────────────────────
    overall_rate  = overview_df['Late_delivery_risk'].mean() * 100
    n_orders      = len(overview_df)
    n_late        = int(overview_df['Late_delivery_risk'].sum())

    high_risk_modes = ['First Class', 'Second Class']
    high_risk_mask  = overview_df['Shipping Mode'].isin(high_risk_modes)
    n_high_risk     = int(high_risk_mask.sum())
    pct_high_risk   = n_high_risk / n_orders * 100

    # Revenue at risk — late orders' revenue and share of total
    overview_df['order_value'] = (
        overview_df['Order Item Product Price'] *
        overview_df['Order Item Quantity']
    )
    total_revenue   = overview_df['order_value'].sum()
    late_revenue    = overview_df.loc[
        overview_df['Late_delivery_risk'] == 1, 'order_value'
    ].sum()
    pct_revenue     = late_revenue / total_revenue * 100
    late_rev_m      = late_revenue / 1_000_000  # convert to millions

    # ── KPI cards (darker label colour) ──────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Overall late rate",
             f"{overall_rate:.0f}%",
             f"{n_late:,} of {n_orders:,} orders",
             "#C41230"),
        (c2, "Orders flagged high-risk",
             f"{n_high_risk:,}",
             f"{pct_high_risk:.0f}% on First/Second Class tiers",
             "#C41230"),
        (c3, "Top driver",
             "Shipping Mode",
             "explains most of late-rate variance",
             "#1f2937"),
        (c4, "Secondary driver",
             "Scheduled days",
             "promise length matters most after tier",
             "#1f2937"),
        (c5, "Revenue at risk",
             f"${late_rev_m:.1f}M | {pct_revenue:.0f}% of revenue",
             "estimated revenue at risk from late orders",
             "#C41230"),
    ]
    for col, label, big, sub, color in cards:
        with col:
            st.markdown(
                f"""
                <div style='background:#f3f4f6; padding:14px; border-radius:8px; min-height:118px;'>
                    <p style='font-size:11px; color:#374151; margin:0; text-transform:uppercase; letter-spacing:.04em; font-weight:600;'>{label}</p>
                    <p style='font-size:22px; font-weight:600; color:{color}; margin:6px 0 4px;'>{big}</p>
                    <p style='font-size:11px; color:#374151; margin:0;'>{sub}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Decide layout: 3 columns by default, 4 if user picked an extra slice ──
    show_extra = dim not in ['Shipping Mode', 'Days for shipment (scheduled)']

    if show_extra:
        col_insight, col_mode, col_days, col_extra = st.columns([1.2, 1.4, 1.4, 1.4])
    else:
        col_insight, col_mode, col_days = st.columns([1.2, 1.4, 1.4])

    # ── Insight callout (single <p> tags, no blank lines inside) ─────
    with col_insight:
        st.markdown(
            """<div style='background:#f3f4f6; border-left:4px solid #BA7517; padding:14px 16px; border-radius:0 8px 8px 0; min-height:500px;'>
<p style='font-size:14px; font-weight:600; margin:0 0 10px; color:#374151;'>🔍 What's driving lateness</p>
<p style='font-size:13px; margin:0 0 10px; color:#374151; line-height:1.7;'><strong style='color:#3B6D11;'>✅ TOP SIGNAL — Shipping Mode.</strong> First Class is late ~95% of the time; Standard Class only ~38%. The secondary signal, <em>scheduled days</em>, says the same thing: the faster you promise, the more often you fail to deliver.</p>
<p style='font-size:13px; margin:0 0 10px; color:#374151; line-height:1.7;'><strong style='color:#C41230;'>❌ NON-SIGNALS — Market, Customer Segment, Region.</strong> All hover around the ~55% baseline, meaning geography and customer type don't predict lateness. Tier choice does.</p>
<p style='font-size:13px; margin:0; color:#374151; line-height:1.7;'><strong style='color:#BA7517;'>⚠️ CAUTION — Multicollinearity.</strong> "Days for shipment (scheduled)" re-encodes Shipping Mode — two predictors saying the same thing.</p>
</div>""",
            unsafe_allow_html=True,
        )

    # ── Top driver: Shipping Mode ────────────────────────────────────
    with col_mode:
        render_slice_section(
            overview_df,
            'Shipping Mode',
            "Late delivery rate by Shipping Mode",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Secondary driver: Scheduled days ─────────────────────────────
    with col_days:
        render_slice_section(
            overview_df,
            'Days for shipment (scheduled)',
            "Late delivery rate by Scheduled days",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 4th column: extra slice picked from the dropdown ─────────────
    if show_extra:
        with col_extra:
            render_slice_section(
                overview_df,
                dim,
                f"Late delivery rate by {dim}",
            )

    # ── Technical footnote at the very bottom ───────────────────────
    st.divider()
    with st.expander("🔧 Technical details (model & methodology)"):
        st.markdown("**Model comparison**")
        leaderboard_df = pd.DataFrame(
            list(schema['leaderboard'].items()),
            columns=['Model', 'Test AUC-ROC']
        ).sort_values('Test AUC-ROC', ascending=False)
        st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Selected model: **{schema['model_name']}**. "
            "Trained on order-time features only — post-shipment columns "
            "(`Days for shipping (real)`, `Delivery Status`) excluded to avoid "
            "data leakage. The ~0.74 AUC ceiling reflects that late delivery in "
            "this dataset is structurally driven by the chosen shipping tier."
        )


# ════════════════════════════════════════════════════════════════════
#  VIEW 2 — ONE-PAGER (SINGLE ORDER)
# ════════════════════════════════════════════════════════════════════
else:

    st.sidebar.header("Order Details")
    st.sidebar.caption("Adjust inputs to score a hypothetical order")

    user_input = {}

    for feature in schema['features']:
        if feature in schema['categorical_cols']:
            options = schema['categories'][feature]
            user_input[feature] = st.sidebar.selectbox(feature, options)
        else:
            r = schema['numeric_ranges'][feature]
            if feature in ['Days for shipment (scheduled)', 'Order Item Quantity']:
                user_input[feature] = st.sidebar.slider(
                    feature,
                    min_value = int(r['min']),
                    max_value = int(r['max']),
                    value     = int(round(r['mean'])),
                    step      = 1,
                )
            else:
                user_input[feature] = st.sidebar.slider(
                    feature,
                    min_value = float(r['min']),
                    max_value = float(r['max']),
                    value     = float(r['mean']),
                )

    predict_clicked = st.sidebar.button("🔍 Predict Risk", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.caption(
        "⚠️ This is a **predictive** model, not causal. "
        "It scores risk based on order profile, not what would change if you tweaked a field."
    )

    if predict_clicked:

        input_df = pd.DataFrame([user_input])
        prob = model.predict_proba(input_df)[0, 1]

        if prob < 0.40:
            badge_color, badge_label = "#3B6D11", "🟢 Low Risk"
        elif prob < 0.70:
            badge_color, badge_label = "#BA7517", "🟡 Medium Risk"
        else:
            badge_color, badge_label = "#C41230", "🔴 High Risk"

        st.subheader("Prediction")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f"""
                <div style='background:#f3f4f6; padding:18px; border-radius:8px; text-align:center;'>
                    <p style='font-size:12px; color:#374151; margin:0; text-transform:uppercase; letter-spacing:.04em; font-weight:600;'>Late risk</p>
                    <p style='font-size:42px; font-weight:600; color:{badge_color}; margin:4px 0;'>{prob*100:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div style='background:#f3f4f6; padding:18px; border-radius:8px; text-align:center;'>
                    <p style='font-size:12px; color:#374151; margin:0; text-transform:uppercase; letter-spacing:.04em; font-weight:600;'>Risk band</p>
                    <p style='font-size:24px; font-weight:600; color:{badge_color}; margin:14px 0 4px;'>{badge_label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            confidence = max(prob, 1 - prob) * 100
            st.markdown(
                f"""
                <div style='background:#f3f4f6; padding:18px; border-radius:8px; text-align:center;'>
                    <p style='font-size:12px; color:#374151; margin:0; text-transform:uppercase; letter-spacing:.04em; font-weight:600;'>Model confidence</p>
                    <p style='font-size:42px; font-weight:600; color:#1f2937; margin:4px 0;'>{confidence:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("##### Order summary")
            for k, v in user_input.items():
                v_disp = f"{v:.2f}" if isinstance(v, float) else v
                st.markdown(
                    f"<div style='padding:6px 0; border-bottom:1px solid #e5e7eb;'>"
                    f"<span style='font-size:12px; color:#6b7280;'>{k}</span><br>"
                    f"<span style='font-size:14px; color:#111827;'>{v_disp}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with col2:
            st.markdown("##### What's driving this risk?")
            st.caption(
                "Red bars push risk **higher**, blue bars push it **lower**. "
                "Bar length = impact size."
            )

            try:
                preprocessor = model.named_steps['prep']
                classifier   = model.named_steps['clf']

                input_transformed = preprocessor.transform(input_df)
                feature_names = [
                    name.replace('num__', '').replace('cat__', '').replace('_', ' ')
                    for name in preprocessor.get_feature_names_out()
                ]

                explainer   = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(input_transformed)

                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]

                shap_df = (
                    pd.DataFrame({'feature': feature_names, 'shap_val': sv})
                    .assign(abs_val=lambda d: d['shap_val'].abs())
                    .sort_values('abs_val', ascending=False)
                    .head(6)
                    .sort_values('shap_val')
                )

                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ['#C41230' if v > 0 else '#185FA5' for v in shap_df['shap_val']]
                ax.barh(shap_df['feature'], shap_df['shap_val'], color=colors, height=0.55)
                ax.axvline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
                ax.set_xlabel("SHAP value (impact on late delivery risk)", fontsize=8)
                ax.tick_params(axis='y', labelsize=7)
                ax.tick_params(axis='x', labelsize=7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

    else:
        st.info("👈 Fill in the order details in the sidebar, then click **Predict Risk**.")
        st.markdown("""
        ### About this view
        Score a single hypothetical order and see **why** the model rates it the way
        it does, using only order-time features (no post-shipment data leakage).
        """)