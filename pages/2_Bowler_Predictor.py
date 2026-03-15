import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Bowler Predictor", page_icon="🎳", layout="wide")

# ── Load model and data ───────────────────────────────────
@st.cache_resource
def load_model():
    with open('bowler_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv('bowler_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    with open('players_data.json') as f:
        players = json.load(f)
    return df, players

model, (df, players) = load_model(), load_data()

# ── Page header ───────────────────────────────────────────
st.title("🎳 Bowler Performance Predictor")
st.markdown("Predict whether a bowler will take **at least 1 wicket** in their next match")
st.divider()

# ── Input section ─────────────────────────────────────────
st.subheader("🎯 Select Match Details")

col1, col2, col3 = st.columns(3)

with col1:
    selected_bowler = st.selectbox(
        "Select Bowler",
        options=players['bowlers'],
        index=players['bowlers'].index('JJ Bumrah')
              if 'JJ Bumrah' in players['bowlers'] else 0
    )

with col2:
    selected_opponent = st.selectbox(
        "Opponent Team (Batting Team)",
        options=players['teams']
    )

with col3:
    selected_venue = st.selectbox(
        "Venue",
        options=players['venues']
    )

st.divider()

# ── Compute features ──────────────────────────────────────
def get_bowler_features(bowler, opponent, venue, df):
    player_df = df[df['bowler'] == bowler].sort_values('date')

    if len(player_df) == 0:
        return None

    # avg_wickets_last5
    last5 = player_df['wickets_taken'].tail(5).mean()

    # career_wickets_avg
    career_avg = player_df['wickets_taken'].mean()

    # past_economy
    past_economy = player_df['economy_rate'].mean()

    # venue_wickets_avg
    venue_df = player_df[player_df['venue'] == venue]
    venue_wickets_avg = venue_df['wickets_taken'].mean() \
                        if len(venue_df) > 0 else career_avg

    # vs_team_wickets_avg
    opp_df = player_df[player_df['batting_team'] == opponent]
    vs_team_avg = opp_df['wickets_taken'].mean() \
                  if len(opp_df) > 0 else career_avg

    # bowl_consistency
    consistency = player_df['wickets_taken'].std()

    # bowl_form_trend
    last3      = player_df['wickets_taken'].tail(3).mean()
    last10     = player_df['wickets_taken'].tail(10).mean()
    form_trend = last3 - last10

    # matches_bowled
    matches_bowled = len(player_df)

    features = {
        'avg_wickets_last5'   : round(last5, 2),
        'career_wickets_avg'  : round(career_avg, 2),
        'past_economy'        : round(past_economy, 2),
        'venue_wickets_avg'   : round(venue_wickets_avg, 2),
        'vs_team_wickets_avg' : round(vs_team_avg, 2),
        'bowl_consistency'    : round(consistency, 2),
        'bowl_form_trend'     : round(form_trend, 2),
        'matches_bowled'      : matches_bowled
    }
    return features

# ── Predict button ────────────────────────────────────────
if st.button("🔮 Predict Performance", type="primary", use_container_width=True):

    features = get_bowler_features(
        selected_bowler, selected_opponent, selected_venue, df
    )

    if features is None:
        st.error("No data found for this bowler!")
    else:
        input_df   = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        probability= model.predict_proba(input_df)[0]

        prob_no_wicket  = round(probability[0] * 100, 1)
        prob_wicket     = round(probability[1] * 100, 1)

        st.divider()
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns([1, 1])

        with col1:
            if prediction == 1:
                st.success(f"### ✅ Likely to Take a Wicket!")
                st.markdown(
                    f"**{selected_bowler}** is predicted to be **effective** "
                    f"against {selected_opponent} at {selected_venue}"
                )
            else:
                st.warning(f"### ⚠️ May Not Take a Wicket")
                st.markdown(
                    f"**{selected_bowler}** may find it tough "
                    f"against {selected_opponent} at {selected_venue}"
                )

            st.markdown("#### Probability Breakdown")
            st.metric("Chance of taking wicket",    f"{prob_wicket}%")
            st.metric("Chance of no wicket",        f"{prob_no_wicket}%")

        with col2:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob_wicket,
                title = {'text': "Probability of taking wicket (%)"},
                gauge = {
                    'axis'  : {'range': [0, 100]},
                    'bar'   : {'color': "#1D9E75" if prob_wicket >= 50 else "#E24B4A"},
                    'steps' : [
                        {'range': [0,  40], 'color': "#FCEBEB"},
                        {'range': [40, 60], 'color': "#FAEEDA"},
                        {'range': [60, 100],'color': "#EAF3DE"}
                    ],
                    'threshold': {
                        'line'     : {'color': "#185FA5", 'width': 4},
                        'thickness': 0.75,
                        'value'    : 50
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # ── Stats breakdown ────────────────────────────────
        st.subheader("🔍 Bowler Stats Used for Prediction")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Career Wickets Avg",  f"{features['career_wickets_avg']}")
        col2.metric("Last 5 Match Avg",    f"{features['avg_wickets_last5']}")
        col3.metric("Venue Wickets Avg",   f"{features['venue_wickets_avg']}")
        col4.metric("Vs Opponent Avg",     f"{features['vs_team_wickets_avg']}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Economy Rate",        f"{features['past_economy']}")
        col2.metric("Form Trend",          f"{features['bowl_form_trend']:+.2f}")
        col3.metric("Consistency (std)",   f"{features['bowl_consistency']}")
        col4.metric("Matches Bowled",      f"{features['matches_bowled']}")

        st.divider()

        # ── Recent form chart ──────────────────────────────
        st.subheader(f"📈 {selected_bowler} — Recent Form (Last 20 Matches)")

        player_df = df[df['bowler'] == selected_bowler].sort_values('date').tail(20)

        fig2 = px.bar(
            player_df,
            x='date',
            y='wickets_taken',
            color='wickets_taken',
            color_continuous_scale='RdYlGn',
            labels={'wickets_taken': 'Wickets Taken', 'date': 'Match Date'},
            title=f"{selected_bowler} — Last 20 IPL Matches"
        )
        fig2.add_hline(
            y=1, line_dash="dash",
            line_color="#185FA5",
            annotation_text="1 wicket mark"
        )
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # ── Economy rate chart ─────────────────────────────
        st.subheader(f"💰 {selected_bowler} — Economy Rate Trend")

        fig3 = px.line(
            player_df,
            x='date',
            y='economy_rate',
            markers=True,
            labels={'economy_rate': 'Economy Rate', 'date': 'Match Date'},
            title=f"{selected_bowler} — Economy Rate (Last 20 Matches)"
        )
        fig3.add_hline(
            y=8, line_dash="dash",
            line_color="#E24B4A",
            annotation_text="8.0 benchmark"
        )
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

        # ── Venue performance ──────────────────────────────
        st.subheader(f"🏟️ {selected_bowler} — Wickets by Venue")

        venue_perf = df[df['bowler'] == selected_bowler].groupby(
            'venue'
        )['wickets_taken'].agg(['mean', 'count']).reset_index()
        venue_perf.columns = ['venue', 'avg_wickets', 'matches']
        venue_perf = venue_perf[
            venue_perf['matches'] >= 2
        ].sort_values('avg_wickets', ascending=True).tail(10)

        fig4 = px.bar(
            venue_perf,
            x='avg_wickets',
            y='venue',
            orientation='h',
            color='avg_wickets',
            color_continuous_scale='Greens',
            labels={'avg_wickets': 'Avg Wickets', 'venue': 'Venue'},
            title=f"Best venues for {selected_bowler}"
        )
        fig4.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)