import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Batsman Predictor", page_icon="🏏", layout="wide")

# ── Load model and data ───────────────────────────────────
@st.cache_resource
def load_model():
    with open('batsman_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('batsman_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    with open('players_data.json') as f:
        players = json.load(f)
    return df, players

model   = load_model()
df, players = load_data()

# ── Page header ───────────────────────────────────────────
st.title("🏏 Batsman Performance Predictor")
st.markdown("Predict whether a batsman will score **30+ runs** in their next match")
st.divider()

# ── Input section ─────────────────────────────────────────
st.subheader("🎯 Select Match Details")

col1, col2, col3 = st.columns(3)

with col1:
    selected_batsman = st.selectbox(
        "Select Batsman",
        options=players['batsmen'],
        index=players['batsmen'].index('V Kohli') 
              if 'V Kohli' in players['batsmen'] else 0
    )

with col2:
    selected_opponent = st.selectbox(
        "Opponent Team (Bowling Team)",
        options=players['teams']
    )

with col3:
    selected_venue = st.selectbox(
        "Venue",
        options=players['venues']
    )

st.divider()

# ── Compute features for selected player ──────────────────
def get_batsman_features(batsman, opponent, venue, df):
    player_df = df[df['batter'] == batsman].sort_values('date')

    if len(player_df) == 0:
        return None

    # Use most recent available stats
    latest = player_df.iloc[-1]

    # avg_runs_last5 — mean of last 5 matches
    last5 = player_df['runs_scored'].tail(5).mean()

    # career_avg
    career_avg = player_df['runs_scored'].mean()

    # past_strike_rate
    if 'past_strike_rate' in player_df.columns:
        strike_rate = player_df['past_strike_rate'].iloc[-1]
    else:
        strike_rate = (career_avg / player_df['balls_faced'].mean() * 100) \
                      if player_df['balls_faced'].mean() > 0 else 0

    # venue_avg
    venue_df = player_df[player_df['venue'] == venue]
    venue_avg = venue_df['runs_scored'].mean() if len(venue_df) > 0 else career_avg

    # vs_team_avg
    opp_df = player_df[player_df['bowling_team'] == opponent]
    vs_team_avg = opp_df['runs_scored'].mean() if len(opp_df) > 0 else career_avg

    # consistency
    consistency = player_df['runs_scored'].std()

    # form_trend
    last3  = player_df['runs_scored'].tail(3).mean()
    last10 = player_df['runs_scored'].tail(10).mean()
    form_trend = last3 - last10

    # matches_played
    matches_played = len(player_df)

    features = {
        'avg_runs_last5'   : round(last5, 2),
        'career_avg'       : round(career_avg, 2),
        'past_strike_rate' : round(strike_rate, 2),
        'venue_avg'        : round(venue_avg, 2),
        'vs_team_avg'      : round(vs_team_avg, 2),
        'consistency'      : round(consistency, 2),
        'form_trend'       : round(form_trend, 2),
        'matches_played'   : matches_played
    }
    return features

# ── Predict button ────────────────────────────────────────
if st.button("🔮 Predict Performance", type="primary", use_container_width=True):

    features = get_batsman_features(
        selected_batsman, selected_opponent, selected_venue, df
    )

    if features is None:
        st.error("No data found for this player!")
    else:
        # Build input dataframe
        input_df = pd.DataFrame([features])
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        prob_under30 = round(probability[0] * 100, 1)
        prob_30plus  = round(probability[1] * 100, 1)

        st.divider()

        # ── Result section ────────────────────────────────
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns([1, 1])

        with col1:
            if prediction == 1:
                st.success(f"### ✅ Likely to Score 30+ Runs!")
                st.markdown(f"**{selected_batsman}** is predicted to have a **strong performance** against {selected_opponent} at {selected_venue}")
            else:
                st.warning(f"### ⚠️ Unlikely to Score 30+ Runs")
                st.markdown(f"**{selected_batsman}** may struggle against {selected_opponent} at {selected_venue}")

            # Probability bar
            st.markdown("#### Probability Breakdown")
            st.metric("Chance of scoring 30+",  f"{prob_30plus}%")
            st.metric("Chance of scoring under 30", f"{prob_under30}%")

        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob_30plus,
                title = {'text': "Probability of 30+ runs (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar' : {'color': "#1D9E75" if prob_30plus >= 50 else "#E24B4A"},
                    'steps': [
                        {'range': [0,  40], 'color': "#FCEBEB"},
                        {'range': [40, 60], 'color': "#FAEEDA"},
                        {'range': [60, 100],'color': "#EAF3DE"}
                    ],
                    'threshold': {
                        'line' : {'color': "#185FA5", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Feature breakdown ──────────────────────────────
        st.subheader("🔍 Player Stats Used for Prediction")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Career Average",    f"{features['career_avg']}")
        col2.metric("Last 5 Match Avg",  f"{features['avg_runs_last5']}")
        col3.metric("Venue Average",     f"{features['venue_avg']}")
        col4.metric("Vs Opponent Avg",   f"{features['vs_team_avg']}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strike Rate",       f"{features['past_strike_rate']}")
        col2.metric("Form Trend",        f"{features['form_trend']:+.1f}")
        col3.metric("Consistency (std)", f"{features['consistency']}")
        col4.metric("Matches Played",    f"{features['matches_played']}")

        st.divider()

        # ── Recent form chart ──────────────────────────────
        st.subheader(f"📈 {selected_batsman} — Recent Form (Last 20 Matches)")

        player_df = df[df['batter'] == selected_batsman].sort_values('date').tail(20)

        fig2 = px.bar(
            player_df,
            x='date',
            y='runs_scored',
            color='runs_scored',
            color_continuous_scale='RdYlGn',
            labels={'runs_scored': 'Runs Scored', 'date': 'Match Date'},
            title=f"{selected_batsman} — Last 20 IPL Innings"
        )
        fig2.add_hline(
            y=30, line_dash="dash",
            line_color="#185FA5",
            annotation_text="30 run mark"
        )
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # ── Venue performance ──────────────────────────────
        st.subheader(f"🏟️ {selected_batsman} — Performance by Venue")

        venue_perf = df[df['batter'] == selected_batsman].groupby(
            'venue'
        )['runs_scored'].agg(['mean', 'count']).reset_index()
        venue_perf.columns = ['venue', 'avg_runs', 'matches']
        venue_perf = venue_perf[venue_perf['matches'] >= 2].sort_values(
            'avg_runs', ascending=True
        ).tail(10)

        fig3 = px.bar(
            venue_perf,
            x='avg_runs',
            y='venue',
            orientation='h',
            color='avg_runs',
            color_continuous_scale='Blues',
            labels={'avg_runs': 'Average Runs', 'venue': 'Venue'},
            title=f"Top venues for {selected_batsman}"
        )
        fig3.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)