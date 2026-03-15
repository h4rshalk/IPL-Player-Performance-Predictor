import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Player Analytics", page_icon="📊", layout="wide")

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    bat_df  = pd.read_csv('batsman_data.csv')
    bowl_df = pd.read_csv('bowler_data.csv')
    bat_df['date']  = pd.to_datetime(bat_df['date'])
    bowl_df['date'] = pd.to_datetime(bowl_df['date'])
    with open('players_data.json') as f:
        players = json.load(f)
    return bat_df, bowl_df, players

bat_df, bowl_df, players = load_data()

# ── Page header ───────────────────────────────────────────
st.title("📊 Player Analytics")
st.markdown("Deep dive into any IPL player's career statistics and performance trends")
st.divider()

# ── Player type selector ──────────────────────────────────
player_type = st.radio(
    "Select Player Type",
    options=["🏏 Batsman", "🎳 Bowler"],
    horizontal=True
)

st.divider()

# ══════════════════════════════════════════════════════════
# BATSMAN ANALYTICS
# ══════════════════════════════════════════════════════════
if player_type == "🏏 Batsman":

    selected_player = st.selectbox(
        "Select Batsman",
        options=players['batsmen'],
        index=players['batsmen'].index('V Kohli')
              if 'V Kohli' in players['batsmen'] else 0
    )

    player_df = bat_df[bat_df['batter'] == selected_player].sort_values('date')

    if len(player_df) == 0:
        st.error("No data found for this player!")
    else:
        # ── Career summary cards ───────────────────────────
        st.subheader(f"🏏 {selected_player} — Career Summary")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Matches",    f"{len(player_df)}")
        col2.metric("Total Runs",       f"{int(player_df['runs_scored'].sum())}")
        col3.metric("Career Average",   f"{player_df['runs_scored'].mean():.1f}")
        col4.metric("Highest Score",    f"{int(player_df['runs_scored'].max())}")
        col5.metric("30+ Scores",
                    f"{(player_df['runs_scored'] >= 30).sum()}")

        st.divider()

        # ── Season wise performance ────────────────────────
        st.subheader("📅 Season-wise Performance")

        season_df = player_df.groupby('season').agg(
            matches    = ('runs_scored', 'count'),
            total_runs = ('runs_scored', 'sum'),
            avg_runs   = ('runs_scored', 'mean'),
            high_score = ('runs_scored', 'max')
        ).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(
                season_df,
                x='season',
                y='total_runs',
                color='total_runs',
                color_continuous_scale='Blues',
                labels={'total_runs': 'Total Runs', 'season': 'Season'},
                title=f"{selected_player} — Total Runs per Season"
            )
            fig1.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.line(
                season_df,
                x='season',
                y='avg_runs',
                markers=True,
                labels={'avg_runs': 'Average Runs', 'season': 'Season'},
                title=f"{selected_player} — Batting Average per Season"
            )
            fig2.add_hline(
                y=player_df['runs_scored'].mean(),
                line_dash="dash",
                line_color="#185FA5",
                annotation_text="Career avg"
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── All innings chart ──────────────────────────────
        st.subheader("📈 All IPL Innings")

        fig3 = px.scatter(
            player_df,
            x='date',
            y='runs_scored',
            color='runs_scored',
            size='runs_scored',
            color_continuous_scale='RdYlGn',
            hover_data=['venue', 'bowling_team', 'season'],
            labels={'runs_scored': 'Runs', 'date': 'Date'},
            title=f"{selected_player} — All IPL Innings"
        )
        fig3.add_hline(
            y=30, line_dash="dash",
            line_color="#185FA5",
            annotation_text="30 run mark"
        )
        fig3.add_hline(
            y=player_df['runs_scored'].mean(),
            line_dash="dot",
            line_color="#BA7517",
            annotation_text="Career avg"
        )
        fig3.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ── Venue and opponent analysis ────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏟️ Best Venues")
            venue_df = player_df.groupby('venue').agg(
                avg_runs = ('runs_scored', 'mean'),
                matches  = ('runs_scored', 'count')
            ).reset_index()
            venue_df = venue_df[
                venue_df['matches'] >= 2
            ].sort_values('avg_runs', ascending=True).tail(8)

            fig4 = px.bar(
                venue_df,
                x='avg_runs',
                y='venue',
                orientation='h',
                color='avg_runs',
                color_continuous_scale='Blues',
                labels={'avg_runs': 'Avg Runs', 'venue': 'Venue'},
                text='avg_runs'
            )
            fig4.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig4.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            st.subheader("⚔️ vs Each Opponent")
            opp_df = player_df.groupby('bowling_team').agg(
                avg_runs = ('runs_scored', 'mean'),
                matches  = ('runs_scored', 'count')
            ).reset_index()
            opp_df = opp_df[
                opp_df['matches'] >= 2
            ].sort_values('avg_runs', ascending=True)

            fig5 = px.bar(
                opp_df,
                x='avg_runs',
                y='bowling_team',
                orientation='h',
                color='avg_runs',
                color_continuous_scale='RdYlGn',
                labels={'avg_runs': 'Avg Runs', 'bowling_team': 'Opponent'},
                text='avg_runs'
            )
            fig5.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig5.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

        st.divider()

        # ── Score distribution ─────────────────────────────
        st.subheader("📊 Score Distribution")

        col1, col2 = st.columns(2)

        with col1:
            fig6 = px.histogram(
                player_df,
                x='runs_scored',
                nbins=30,
                color_discrete_sequence=['#185FA5'],
                labels={'runs_scored': 'Runs Scored'},
                title=f"{selected_player} — Score Distribution"
            )
            fig6.update_layout(height=320)
            st.plotly_chart(fig6, use_container_width=True)

        with col2:
            bucket_counts = pd.DataFrame({
                'Category': ['0–10', '11–30', '31–60', '60+'],
                'Count': [
                    (player_df['runs_scored'] <= 10).sum(),
                    ((player_df['runs_scored'] > 10) & (player_df['runs_scored'] <= 30)).sum(),
                    ((player_df['runs_scored'] > 30) & (player_df['runs_scored'] <= 60)).sum(),
                    (player_df['runs_scored'] > 60).sum()
                ]
            })
            fig7 = px.pie(
                bucket_counts,
                names='Category',
                values='Count',
                color_discrete_sequence=['#E24B4A','#BA7517','#185FA5','#1D9E75'],
                title=f"{selected_player} — Score Breakdown"
            )
            fig7.update_layout(height=320)
            st.plotly_chart(fig7, use_container_width=True)

# ══════════════════════════════════════════════════════════
# BOWLER ANALYTICS
# ══════════════════════════════════════════════════════════
else:
    selected_player = st.selectbox(
        "Select Bowler",
        options=players['bowlers'],
        index=players['bowlers'].index('JJ Bumrah')
              if 'JJ Bumrah' in players['bowlers'] else 0
    )

    player_df = bowl_df[bowl_df['bowler'] == selected_player].sort_values('date')

    if len(player_df) == 0:
        st.error("No data found for this player!")
    else:
        # ── Career summary ─────────────────────────────────
        st.subheader(f"🎳 {selected_player} — Career Summary")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Matches",    f"{len(player_df)}")
        col2.metric("Total Wickets",    f"{int(player_df['wickets_taken'].sum())}")
        col3.metric("Career Avg Wickets", f"{player_df['wickets_taken'].mean():.2f}")
        col4.metric("Best Bowling",     f"{int(player_df['wickets_taken'].max())}")
        col5.metric("Career Economy",   f"{player_df['economy_rate'].mean():.2f}")

        st.divider()

        # ── Season wise ────────────────────────────────────
        st.subheader("📅 Season-wise Performance")

        season_df = player_df.groupby('season').agg(
            matches        = ('wickets_taken', 'count'),
            total_wickets  = ('wickets_taken', 'sum'),
            avg_wickets    = ('wickets_taken', 'mean'),
            avg_economy    = ('economy_rate',  'mean')
        ).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(
                season_df,
                x='season',
                y='total_wickets',
                color='total_wickets',
                color_continuous_scale='Greens',
                labels={'total_wickets': 'Total Wickets', 'season': 'Season'},
                title=f"{selected_player} — Total Wickets per Season"
            )
            fig1.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.line(
                season_df,
                x='season',
                y='avg_economy',
                markers=True,
                labels={'avg_economy': 'Economy Rate', 'season': 'Season'},
                title=f"{selected_player} — Economy Rate per Season"
            )
            fig2.add_hline(
                y=8, line_dash="dash",
                line_color="#E24B4A",
                annotation_text="8.0 benchmark"
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── All matches scatter ────────────────────────────
        st.subheader("📈 All IPL Matches")

        fig3 = px.scatter(
            player_df,
            x='date',
            y='wickets_taken',
            color='wickets_taken',
            size='wickets_taken'.replace,
            color_continuous_scale='RdYlGn',
            hover_data=['venue', 'batting_team', 'economy_rate'],
            labels={'wickets_taken': 'Wickets', 'date': 'Date'},
            title=f"{selected_player} — All IPL Matches"
        )
        fig3.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ── Venue and opponent ─────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏟️ Best Venues")
            venue_df = player_df.groupby('venue').agg(
                avg_wickets = ('wickets_taken', 'mean'),
                matches     = ('wickets_taken', 'count')
            ).reset_index()
            venue_df = venue_df[
                venue_df['matches'] >= 2
            ].sort_values('avg_wickets', ascending=True).tail(8)

            fig4 = px.bar(
                venue_df,
                x='avg_wickets',
                y='venue',
                orientation='h',
                color='avg_wickets',
                color_continuous_scale='Greens',
                labels={'avg_wickets': 'Avg Wickets', 'venue': 'Venue'},
                text='avg_wickets'
            )
            fig4.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig4.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            st.subheader("⚔️ vs Each Opponent")
            opp_df = player_df.groupby('batting_team').agg(
                avg_wickets = ('wickets_taken', 'mean'),
                matches     = ('wickets_taken', 'count')
            ).reset_index()
            opp_df = opp_df[
                opp_df['matches'] >= 2
            ].sort_values('avg_wickets', ascending=True)

            fig5 = px.bar(
                opp_df,
                x='avg_wickets',
                y='batting_team',
                orientation='h',
                color='avg_wickets',
                color_continuous_scale='RdYlGn',
                labels={'avg_wickets': 'Avg Wickets', 'batting_team': 'Opponent'},
                text='avg_wickets'
            )
            fig5.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig5.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

        st.divider()

        # ── Wicket distribution ────────────────────────────
        st.subheader("📊 Wicket Distribution")

        col1, col2 = st.columns(2)

        with col1:
            fig6 = px.histogram(
                player_df,
                x='wickets_taken',
                color_discrete_sequence=['#1D9E75'],
                labels={'wickets_taken': 'Wickets Taken'},
                title=f"{selected_player} — Wicket Distribution"
            )
            fig6.update_layout(height=320)
            st.plotly_chart(fig6, use_container_width=True)

        with col2:
            wicket_counts = pd.DataFrame({
                'Category': ['0 wickets', '1 wicket', '2 wickets', '3+ wickets'],
                'Count': [
                    (player_df['wickets_taken'] == 0).sum(),
                    (player_df['wickets_taken'] == 1).sum(),
                    (player_df['wickets_taken'] == 2).sum(),
                    (player_df['wickets_taken'] >= 3).sum()
                ]
            })
            fig7 = px.pie(
                wicket_counts,
                names='Category',
                values='Count',
                color_discrete_sequence=['#E24B4A','#BA7517','#185FA5','#1D9E75'],
                title=f"{selected_player} — Wicket Breakdown"
            )
            fig7.update_layout(height=320)
            st.plotly_chart(fig7, use_container_width=True)