"""Fin-Flow CFO Dashboard"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from fin_flow.agents import AdvisorAgent, CategorizerAgent, ForecasterAgent
from fin_flow.ingestion import deduplicate, load_file
from fin_flow.ingestion.normalizer import IngestionError
from fin_flow.storage import InMemoryVectorStore, HashingEmbedder

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


def _to_float(x):
    return float(x)


with st.sidebar:
    st.title("Fin-Flow CFO")
    st.caption("Upload bank exports, explore your finances, ask questions.")

    uploaded = st.file_uploader(
        "Upload bank exports",
        type=["csv", "xlsx", "xls", "json"],
        accept_multiple_files=True,
    )

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
            cat = CategorizerAgent(prefer_llm=False)
            df = cat.classify_dataframe(raw)
            st.session_state.transactions = df

            fc_agent = ForecasterAgent(prefer_prophet=False)
            st.session_state.forecast = fc_agent.forecast(
                df,
                starting_balance=starting_balance,
                horizon_days=horizon,
            )
            st.success(f"Loaded {len(df)} transactions from {len(uploaded)} file(s).")

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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", f"{n_transactions:,}")
col2.metric("Total Income", f"${total_income:,.2f}")
col3.metric("Total Expenses", f"${abs(total_expenses):,.2f}")
col4.metric("Net Cashflow", f"${net:,.2f}", delta=f"{'positive' if net >= 0 else 'negative'}")

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
        st.plotly_chart(fig_cat, use_container_width=True)

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
    st.plotly_chart(fig_daily, use_container_width=True)

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
    st.plotly_chart(fig_fc, use_container_width=True)
    st.caption(fc.summary())

st.divider()
with st.expander("Transaction Details", expanded=False):
    display = df[["transaction_date", "description", "amount", "category", "ai_confidence_score"]].copy()
    display.columns = ["Date", "Description", "Amount", "Category", "Confidence"]
    display["Amount"] = display["Amount"].map(lambda x: f"${_to_float(x):,.2f}")
    display["Confidence"] = display["Confidence"].map(lambda x: f"{float(x or 0):.0%}")
    st.dataframe(display, use_container_width=True, hide_index=True)

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
        vector_store=st.session_state.store, prefer_llm=False
    )
    answer = advisor.ask(
        prompt,
        transactions=df,
        starting_balance=starting_balance,
    )

    reply_parts = [answer.answer]
    if answer.citations:
        reply_parts.append(f"\n*Grounded on {len(answer.citations)} transaction(s).*")
    if answer.retrieved_notes:
        notes_str = "\n".join(
            f"- {h.text}" for h in answer.retrieved_notes[:3]
        )
        reply_parts.append(f"\n**Related context:**\n{notes_str}")

    reply = "\n".join(reply_parts)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
