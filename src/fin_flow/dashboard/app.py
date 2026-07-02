"""Fin-Flow CFO Dashboard"""

import io
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from fin_flow.agents import (
    AdvisorAgent, CategorizerAgent, ForecasterAgent,
    compute_period_deltas, monthly_summary, yoy_comparison,
)
from fin_flow.ingestion import deduplicate, load_file
from fin_flow.ingestion.normalizer import IngestionError
from fin_flow.reports.tax_report import (
    build_tax_report, tax_report_to_csv_rows, generate_tax_pdf,
    TAX_DISCLAIMER, SCHEDULE_C_MAPPING,
)
from fin_flow.storage import InMemoryVectorStore, HashingEmbedder
from fin_flow.utils.analytics import log_event, submit_email, save_email_locally

st.set_page_config(
    page_title="Fin-Flow CFO",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame()
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "store" not in st.session_state:
    st.session_state.store = InMemoryVectorStore(embedder=HashingEmbedder(dim=128))
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "starting_balance" not in st.session_state:
    st.session_state.starting_balance = 0.0
if "filter_needs_review" not in st.session_state:
    st.session_state.filter_needs_review = False
if "fallback_note_shown" not in st.session_state:
    st.session_state.fallback_note_shown = False
if "category_cache" not in st.session_state:
    st.session_state.category_cache = {}  # raw_hash -> (category, confidence)
if "session_logged" not in st.session_state:
    log_event("session_start")
    st.session_state.session_logged = True


def _to_float(x):
    return float(x)


with st.sidebar:
    st.title("Fin-Flow CFO")
    st.caption("Upload bank exports, explore your finances, ask questions.")

    uploaded = st.file_uploader(
        "Upload bank exports",
        type=["csv", "xlsx", "xls", "json", "pdf"],
        accept_multiple_files=True,
    )
    st.caption(
        "Your files are processed in your session only. "
        "We never store your transactions or financial data. "
        "Closing this tab deletes everything."
    )

    if st.session_state.transactions.empty:
        if st.button("Try with sample data", use_container_width=True):
            sample_path = _ROOT / "data" / "samples" / "chase_sample.csv"
            if sample_path.exists():
                sample_df = load_file(sample_path, source="chase_sample.csv")
                cat = CategorizerAgent(prefer_llm=True)
                classified = cat.classify_dataframe(
                    sample_df,
                    cache=st.session_state.category_cache,
                )
                st.session_state.transactions = classified
                fc_agent = ForecasterAgent(prefer_prophet=False)
                st.session_state.forecast = fc_agent.forecast(
                    classified,
                    starting_balance=st.session_state.starting_balance,
                    horizon_days=90,
                )
                log_event("sample_data_used")
                st.rerun()

    starting_balance = st.number_input(
        "Starting cash balance ($)",
        min_value=0.0,
        value=st.session_state.starting_balance,
        step=500.0,
        format="%.2f",
    )
    st.session_state.starting_balance = starting_balance

    horizon = st.slider("Forecast horizon (days)", 30, 365, 90)

    if uploaded:
        frames = []
        for f in uploaded:
            tmp = _ROOT / "data" / "processed" / f.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(f.read())
            try:
                frames.append(load_file(tmp, source=f.name))
            except IngestionError as e:
                st.error(f"{f.name}: {e}")

        if frames:
            raw = deduplicate(pd.concat(frames, ignore_index=True))
            cat = CategorizerAgent(prefer_llm=True)
            _progress_bar = st.progress(0, text="Categorizing transactions...")

            def _on_progress(done, total):
                _progress_bar.progress(
                    done / total,
                    text=f"Categorizing transactions ({done}/{total} chunks)...",
                )

            df = cat.classify_dataframe(
                raw,
                cache=st.session_state.category_cache,
                progress_callback=_on_progress,
            )
            _progress_bar.empty()
            st.session_state.transactions = df

            fc_agent = ForecasterAgent(prefer_prophet=False)
            st.session_state.forecast = fc_agent.forecast(
                df,
                starting_balance=starting_balance,
                horizon_days=horizon,
            )
            log_event("file_uploaded", row_count=len(df))
            st.success(f"Loaded {len(df)} transactions from {len(uploaded)} file(s).")
            if cat.last_backend_used == "gemini":
                st.caption("Categorized with Gemini Flash.")
            else:
                st.caption(
                    "Using rule-based categorization set "
                    "GEMINI_API_KEY for AI-powered categorization."
                )

    # ── Email capture ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Stay Updated")
    _formspree = os.environ.get("FORMSPREE_ENDPOINT")
    _email_input = st.text_input(
        "Get tips and product updates",
        placeholder="you@company.com",
        key="email_input",
    )
    if st.button("Subscribe", key="email_btn") and _email_input.strip():
        if _formspree:
            if submit_email(_email_input.strip(), _formspree):
                st.success("Subscribed!")
            else:
                st.error("Could not subscribe, please try again.")
        else:
            _local_path = str(_ROOT / "data" / "email_subscribers.json")
            if save_email_locally(_email_input.strip(), _local_path):
                st.success("Thanks! We'll keep you posted.")
            else:
                st.error("Could not save, please try again.")

    st.divider()
    st.subheader("Business Context")
    note_text = st.text_area("Add a business note", placeholder="e.g. We plan to scale marketing spend by 20% in Q3.")
    note_type = st.selectbox("Note type", ["strategic_goal", "constraint", "memo", "note"])
    if st.button("Save Note") and note_text.strip():
        advisor = AdvisorAgent(
            vector_store=st.session_state.store, prefer_llm=False
        )
        advisor.add_note(note_text, type=note_type)
        st.success("Note saved.")


df = st.session_state.transactions
fc = st.session_state.forecast

if df.empty:
    st.header("Welcome to Fin-Flow CFO")
    st.info("Upload one or more bank exports in the sidebar to get started.")
    st.stop()

total_income = float(df[df["amount"].map(_to_float) > 0]["amount"].map(_to_float).sum())
total_expenses = float(df[df["amount"].map(_to_float) < 0]["amount"].map(_to_float).sum())
net = total_income + total_expenses
n_transactions = len(df)

dq = CategorizerAgent.data_quality(df)
deltas = compute_period_deltas(df)

# ── 1. Headline insight banner ───────────────────────────────────────
_headline = []
if fc:
    _headline.append(fc.summary())
if dq.uncategorized_pct > 0:
    _headline.append(
        f"{dq.uncategorized_pct:.0f}% of transactions still uncategorized."
    )
if _headline:
    _banner = " ".join(_headline)
    if (fc and fc.death_date) or dq.uncategorized_pct > 25:
        st.warning(_banner)
    else:
        st.info(_banner)

# ── 2. Primary metrics (with period-over-period deltas) ──────────────
col1, col2, col3, col4 = st.columns(4)
if deltas.has_prior:
    # When the prior half has very few transactions, percentage
    # swings are misleading — fall back to dollar deltas.
    _thin = deltas.prior_count < 5
    col1.metric(
        "Transactions", f"{n_transactions:,}",
        delta=f"{deltas.transactions_delta:+d} vs prior",
    )
    if _thin or deltas.income_delta_pct is None:
        _inc_d = f"${deltas.income_delta:+,.0f} vs prior"
    else:
        _inc_d = f"{deltas.income_delta_pct:+.1f}%"
    col2.metric("Total Income", f"${total_income:,.2f}", delta=_inc_d)

    if _thin or deltas.expenses_delta_pct is None:
        _exp_d = f"${deltas.expenses_delta:+,.0f} vs prior"
    else:
        _exp_d = f"{deltas.expenses_delta_pct:+.1f}%"
    col3.metric(
        "Total Expenses", f"${abs(total_expenses):,.2f}",
        delta=_exp_d,
        delta_color="inverse",
    )
    col4.metric(
        "Net Cashflow", f"${net:,.2f}",
        delta=f"${deltas.net_delta:+,.2f}",
    )
else:
    col1.metric("Transactions", f"{n_transactions:,}")
    col2.metric("Total Income", f"${total_income:,.2f}")
    col3.metric("Total Expenses", f"${abs(total_expenses):,.2f}")
    col4.metric(
        "Net Cashflow", f"${net:,.2f}",
        delta="positive" if net >= 0 else "negative",
    )

# ── 3. Secondary metrics: Needs Review + Burn Rate ───────────────────
_s1, _s2, _s3, _ = st.columns(4)

with _s1:
    st.metric("Needs Review", dq.needs_review_count)
    if dq.needs_review_count > 0:
        _lbl = (
            "Clear filter"
            if st.session_state.filter_needs_review
            else "Show these ↓"
        )
        if st.button(_lbl, key="toggle_review_btn"):
            st.session_state.filter_needs_review = (
                not st.session_state.filter_needs_review
            )
            st.rerun()

if fc:
    _s2.metric("Avg Daily Income", f"${fc.mean_daily_income:,.2f}")
    _s3.metric("Avg Daily Burn", f"${fc.mean_daily_burn:,.2f}")

st.divider()

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Spending by Category")
    expenses = df[df["amount"].map(_to_float) < 0].copy()
    if not expenses.empty:
        expenses["abs_amount"] = expenses["amount"].map(lambda x: abs(_to_float(x)))
        by_cat = expenses.groupby("category")["abs_amount"].sum().reset_index()
        by_cat.columns = ["Category", "Amount"]
        by_cat = by_cat.sort_values("Amount", ascending=False)
        fig_cat = px.bar(
            by_cat,
            x="Amount",
            y="Category",
            orientation="h",
            color="Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_cat.update_layout(showlegend=False, yaxis=dict(autorange="reversed"), height=400)
        st.plotly_chart(fig_cat, width='stretch')

with chart_right:
    st.subheader("Daily Cashflow")
    daily = df.copy()
    daily["date"] = pd.to_datetime(daily["transaction_date"])
    daily["amount_f"] = daily["amount"].map(_to_float)
    daily_agg = daily.groupby("date")["amount_f"].sum().reset_index()
    daily_agg.columns = ["Date", "Net"]
    fig_daily = px.bar(
        daily_agg,
        x="Date",
        y="Net",
        color_discrete_sequence=["#4C78A8"],
    )
    fig_daily.update_layout(height=400)
    st.plotly_chart(fig_daily, width='stretch')

if fc and not fc.projection.empty:
    st.divider()
    st.subheader("Cash Runway Forecast")

    proj = fc.projection.copy()
    proj["date"] = pd.to_datetime(proj["date"])

    fig_fc = go.Figure()
    fig_fc.add_trace(
        go.Scatter(
            x=proj["date"],
            y=proj["projected_balance"],
            mode="lines",
            name="Projected Balance",
            line=dict(color="#4C78A8", width=2),
        )
    )
    # zero line
    fig_fc.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="$0")
    if fc.death_date:
        fig_fc.add_vline(
            x=str(fc.death_date),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Cash runs out: {fc.death_date}",
        )
    fig_fc.update_layout(
        yaxis_title="Projected Balance ($)",
        xaxis_title="Date",
        height=350,
    )
    st.plotly_chart(fig_fc, width='stretch')
    st.caption(fc.summary())

# ── 4. Monthly Trends ───────────────────────────────────────────────────
_monthly = monthly_summary(df)
if len(_monthly) >= 3:
    st.divider()
    st.subheader("Monthly Trends")
    _monthly["month_label"] = _monthly.apply(
        lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
    )
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=_monthly["month_label"], y=_monthly["income"],
        name="Income", marker_color="#66BB6A",
    ))
    fig_monthly.add_trace(go.Bar(
        x=_monthly["month_label"], y=_monthly["expenses"],
        name="Expenses", marker_color="#EF5350",
    ))
    fig_monthly.update_layout(
        barmode="group", height=350,
        yaxis_title="Amount ($)", xaxis_title="Month",
    )
    st.plotly_chart(fig_monthly, width='stretch')

# ── 5. Year-over-Year Comparison ────────────────────────────────────────
_yoy = yoy_comparison(df)
if _yoy.has_prior_year:
    st.divider()
    st.subheader(f"Year-over-Year: {_yoy.prior_year} vs {_yoy.current_year}")
    log_event("yoy_viewed")

    yc1, yc2, yc3, yc4 = st.columns(4)
    yc1.metric(
        f"Income ({_yoy.ytd_label})",
        f"${_yoy.ytd_current_income:,.2f}",
        delta=f"{_yoy.ytd_income_delta_pct:+.1f}%" if _yoy.ytd_income_delta_pct is not None else f"${_yoy.ytd_income_delta:+,.0f}",
    )
    yc2.metric(
        f"Expenses ({_yoy.ytd_label})",
        f"${_yoy.ytd_current_expenses:,.2f}",
        delta=f"{_yoy.ytd_expenses_delta_pct:+.1f}%" if _yoy.ytd_expenses_delta_pct is not None else f"${_yoy.ytd_expenses_delta:+,.0f}",
        delta_color="inverse",
    )

    # Monthly income comparison chart
    if not _yoy.monthly.empty:
        _ym = _yoy.monthly
        fig_yoy = go.Figure()
        fig_yoy.add_trace(go.Bar(
            x=_ym["month_name"], y=_ym["prior_expenses"],
            name=str(_yoy.prior_year), marker_color="#90CAF9",
        ))
        fig_yoy.add_trace(go.Bar(
            x=_ym["month_name"], y=_ym["current_expenses"],
            name=str(_yoy.current_year), marker_color="#1565C0",
        ))
        fig_yoy.update_layout(
            barmode="group", height=300,
            yaxis_title="Expenses ($)", xaxis_title="Month",
        )
        st.plotly_chart(fig_yoy, width='stretch')

    # Top category changes
    if _yoy.top_category_changes:
        st.caption("Largest category changes (expenses):")
        _change_rows = []
        for ch in _yoy.top_category_changes:
            pct_str = f"{ch['delta_pct']:+.1f}%" if ch["delta_pct"] is not None else "N/A"
            _change_rows.append({
                "Category": ch["category"],
                f"{_yoy.prior_year}": f"${ch['prior_amount']:,.2f}",
                f"{_yoy.current_year}": f"${ch['current_amount']:,.2f}",
                "Change": f"${ch['delta']:+,.2f} ({pct_str})",
            })
        st.dataframe(
            pd.DataFrame(_change_rows),
            use_container_width=True,
            hide_index=True,
        )

# ── 6. Tax-Ready Report ────────────────────────────────────────────────
st.divider()
with st.expander("Tax-Ready Report (Schedule C)", expanded=False):
    _years = sorted(pd.to_datetime(df["transaction_date"]).dt.year.unique(), reverse=True)
    if _years:
        _tax_year = st.selectbox("Tax year", _years, key="tax_year_select")
        _tax_report = build_tax_report(df, _tax_year)

        if _tax_report.transaction_count > 0:
            st.caption(f"{_tax_report.transaction_count} transactions in {_tax_year}")
            st.metric("Business Income", f"${_tax_report.income_total:,.2f}")

            # Expense lines table
            if _tax_report.expense_lines:
                st.subheader("Deductible Expenses")
                _exp_display = []
                for line in _tax_report.expense_lines:
                    row = {
                        "Category": line["category"],
                        "Schedule C": line["schedule_c_line"],
                        "Amount": f"${line['amount']:,.2f}",
                    }
                    if line["note"]:
                        row["Note"] = line["note"]
                    _exp_display.append(row)
                st.dataframe(
                    pd.DataFrame(_exp_display),
                    use_container_width=True,
                    hide_index=True,
                )
                st.metric("Total Deductible", f"${_tax_report.expense_total:,.2f}")

            # Excluded table
            if _tax_report.excluded:
                st.subheader("Excluded from Schedule C")
                _excl_display = [{
                    "Category": e["category"],
                    "Amount": f"${e['amount']:,.2f}",
                    "Reason": e["reason"],
                } for e in _tax_report.excluded]
                st.dataframe(
                    pd.DataFrame(_excl_display),
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(TAX_DISCLAIMER)

            # Downloads — payment gating
            _stripe_link = os.environ.get("STRIPE_PAYMENT_LINK")
            _paid = st.query_params.get("paid") == "1"

            if _stripe_link and not _paid:
                st.info("Download your full tax report package:")
                st.link_button(
                    "Unlock Downloads ($29)",
                    _stripe_link,
                    use_container_width=True,
                )
            else:
                _dl1, _dl2 = st.columns(2)
                with _dl1:
                    _csv_rows = tax_report_to_csv_rows(_tax_report, df)
                    _csv_df = pd.DataFrame(_csv_rows)
                    _csv_bytes = _csv_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download CSV",
                        data=_csv_bytes,
                        file_name=f"fin_flow_tax_{_tax_year}.csv",
                        mime="text/csv",
                        key="tax_csv_dl",
                        on_click=lambda: log_event("tax_download_clicked", format="csv"),
                    )
                with _dl2:
                    try:
                        _pdf_bytes = generate_tax_pdf(_tax_report)
                        st.download_button(
                            "Download PDF",
                            data=_pdf_bytes,
                            file_name=f"fin_flow_tax_{_tax_year}.pdf",
                            mime="application/pdf",
                            key="tax_pdf_dl",
                            on_click=lambda: log_event("tax_download_clicked", format="pdf"),
                        )
                    except Exception:
                        st.caption("PDF generation requires fpdf2. Install with: pip install fpdf2")

            log_event("tax_report_generated", year=_tax_year)

            # Email capture after tax report
            st.divider()
            _tax_email = st.text_input(
                "Get a monthly summary of features and tips",
                placeholder="you@company.com",
                key="tax_email_input",
            )
            if st.button("Subscribe", key="tax_email_btn") and _tax_email.strip():
                _fs = os.environ.get("FORMSPREE_ENDPOINT")
                if _fs:
                    if submit_email(_tax_email.strip(), _fs):
                        st.success("Subscribed!")
                    else:
                        st.error("Could not subscribe, please try again.")
                else:
                    _local_path = str(_ROOT / "data" / "email_subscribers.json")
                    if save_email_locally(_tax_email.strip(), _local_path):
                        st.success("Thanks! We'll keep you posted.")
                    else:
                        st.error("Could not save, please try again.")
        else:
            st.info(f"No transactions found for {_tax_year}.")

st.divider()
with st.expander(
    "Transaction Details",
    expanded=st.session_state.filter_needs_review,
):
    display = df[["transaction_date", "description", "amount", "category", "ai_confidence_score"]].copy()
    display["transaction_date"] = pd.to_datetime(display["transaction_date"]).dt.strftime("%Y-%m-%d")
    display.columns = ["Date", "Description", "Amount", "Category", "Confidence"]

    if st.session_state.filter_needs_review:
        _conf_raw = df["ai_confidence_score"].map(lambda x: float(x or 0))
        display = display[_conf_raw.values < 0.6]
        st.caption(
            f"Showing {len(display)} transaction(s) with confidence below 60%."
        )

    display["Amount"] = display["Amount"].map(lambda x: f"${_to_float(x):,.2f}")
    display["Confidence"] = display["Confidence"].map(lambda x: f"{float(x or 0):.0%}")
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn(width="small"),
            "Description": st.column_config.TextColumn(width="large"),
            "Amount": st.column_config.TextColumn(width="small"),
            "Category": st.column_config.TextColumn(width="medium"),
            "Confidence": st.column_config.TextColumn(width="small"),
        },
    )

    # Download categorized transactions CSV
    _txn_export = df[["transaction_date", "description", "amount", "category", "ai_confidence_score"]].copy()
    _txn_export.columns = ["Date", "Description", "Amount", "Category", "Confidence"]
    _txn_csv = _txn_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download categorized transactions (CSV)",
        data=_txn_csv,
        file_name="fin_flow_transactions.csv",
        mime="text/csv",
        key="txn_csv_dl",
    )

st.divider()
st.subheader("Ask Fin-Flow")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a financial question...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    advisor = AdvisorAgent(
        vector_store=st.session_state.store, prefer_llm=True
    )
    answer = advisor.ask(
        prompt,
        transactions=df,
        starting_balance=starting_balance,
    )

    log_event("chat_question", intent=answer.intent, backend=answer.backend)

    # One-time fallback note on the first rules response
    answer_text = answer.answer
    if (
        answer.backend == "rules"
        and not st.session_state.fallback_note_shown
    ):
        answer_text = (
            "Showing a quick summary — full AI analysis "
            "available on your next question.\n\n" + answer_text
        )
        st.session_state.fallback_note_shown = True

    reply_parts = [answer_text.strip()]
    if answer.citations:
        reply_parts.append(f"*Grounded on {len(answer.citations)} transaction(s).*")
    if answer.retrieved_notes:
        notes_str = "\n".join(
            f"- {h.text}" for h in answer.retrieved_notes[:3]
        )
        reply_parts.append(f"**Related context:**\n\n{notes_str}")

    reply = "\n\n".join(reply_parts)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)