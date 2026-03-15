import streamlit as st
import pandas as pd
import json

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="IPL Performance Predictor",
    page_icon="🏏",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    batsman_data = pd.read_csv('batsman_data.csv')
    bowler_data  = pd.read_csv('bowler_data.csv')
    with open('players_data.json') as f:
        players = json.load(f)
    return batsman_data, bowler_data, players

batsman_data, bowler_data, players = load_data()

# ── Header ────────────────────────────────────────────────
st.title("🏏 IPL Player Performance Predictor")
st.markdown("#### Predict whether an IPL player will perform in their next match using Machine Learning")
st.divider()

# ── Key stats row ─────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Matches",
        value=f"{batsman_data['match_id'].nunique():,}"
    )
with col2:
    st.metric(
        label="Batsmen Tracked",
        value=f"{batsman_data['batter'].nunique():,}"
    )
with col3:
    st.metric(
        label="Bowlers Tracked",
        value=f"{bowler_data['bowler'].nunique():,}"
    )
with col4:
    st.metric(
        label="Seasons Covered",
        value=f"{batsman_data['season'].nunique()}"
    )

st.divider()

# ── About section ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 About This Project")
    st.markdown("""
    This app uses **Machine Learning** to predict IPL player performance
    based on historical match data from **2008 to 2024**.

    **Models used:**
    - XGBoost Classifier with Sklearn Pipeline
    - Trained on 13,000+ player-match records
    - Features engineered from ball-by-ball data

    **Predictions available:**
    - 🏏 Will a batsman score **30+ runs**?
    - 🎳 Will a bowler take **at least 1 wicket**?
    """)

with col2:
    st.subheader("📊 Model Performance")
    st.markdown("""
    | Model | Accuracy | ROC-AUC |
    |-------|----------|---------|
    | Batsman Predictor | 57.71% | 0.656 |
    | Bowler Predictor | 58.89% | 0.560 |

    > **Note:** Cricket performance has high natural variance.
    > Even world-class players are unpredictable match to match.
    > These models provide data-driven probability estimates,
    > not guaranteed predictions.
    """)

st.divider()

# ── Features used ─────────────────────────────────────────
st.subheader("🔧 Features Used in the Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Batsman Features**")
    bat_features = {
        "avg_runs_last5"   : "Average runs in last 5 matches",
        "career_avg"       : "Overall career batting average",
        "past_strike_rate" : "Historical strike rate",
        "venue_avg"        : "Average runs at this venue",
        "vs_team_avg"      : "Average runs vs this opponent",
        "consistency"      : "Standard deviation of scores",
        "form_trend"       : "Recent vs long-term form difference",
        "matches_played"   : "Total IPL experience"
    }
    for feat, desc in bat_features.items():
        st.markdown(f"- **{feat}** — {desc}")

with col2:
    st.markdown("**Bowler Features**")
    bowl_features = {
        "avg_wickets_last5"    : "Average wickets in last 5 matches",
        "career_wickets_avg"   : "Overall career wicket average",
        "past_economy"         : "Historical economy rate",
        "venue_wickets_avg"    : "Average wickets at this venue",
        "vs_team_wickets_avg"  : "Average wickets vs this opponent",
        "bowl_consistency"     : "Standard deviation of wickets",
        "bowl_form_trend"      : "Recent vs long-term form difference",
        "matches_bowled"       : "Total IPL bowling experience"
    }
    for feat, desc in bowl_features.items():
        st.markdown(f"- **{feat}** — {desc}")

st.divider()

# ── Navigation hint ───────────────────────────────────────
st.subheader("🚀 Get Started")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("🏏 **Batsman Predictor**\nGo to page 1 in the sidebar")
with col2:
    st.info("🎳 **Bowler Predictor**\nGo to page 2 in the sidebar")
with col3:
    st.info("📊 **Player Analytics**\nGo to page 3 in the sidebar")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:13px;'>"
    "Built with Python · Scikit-learn · XGBoost · Streamlit &nbsp;|&nbsp; "
    "Data: IPL 2008–2024"
    "</div>",
    unsafe_allow_html=True
)