# 🏏 IPL Player Performance Predictor

A Machine Learning web application that predicts IPL player performance using historical ball-by-ball data from **2008 to 2024**.

> **Live Demo:** [Click here to open the app](https://YOUR_USERNAME-ipl-predictor.streamlit.app)

---

## 📌 Project Overview

This project predicts whether an IPL batsman will score **30+ runs** or a bowler will take **at least 1 wicket** in an upcoming match — based on their historical performance, venue statistics, and head-to-head records against the opponent.

The entire project covers the **full data science pipeline**:
- Raw data ingestion and cleaning
- Feature engineering from ball-by-ball data
- ML model training with Sklearn Pipelines
- Interactive web app deployment using Streamlit

---

## 🎯 Problem Statement

Can we predict IPL player performance using historical data?

- **Batsman:** Will this player score 30+ runs in the next match?
- **Bowler:** Will this player take at least 1 wicket in the next match?

---

## 📊 Dataset

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `matches.csv` | 1,095 | 20 | Match-level data — venue, teams, toss, result |
| `deliveries.csv` | 260,920 | 17 | Ball-by-ball data — every delivery since 2008 |

**Source:** [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)

**Coverage:** IPL seasons 2008 — 2024

---

## 🔧 Features Engineered

All features are computed from **past matches only** using `shift(1)` and expanding means — ensuring zero data leakage.

### Batsman Features

| Feature | Description |
|---------|-------------|
| `avg_runs_last5` | Average runs scored in last 5 matches |
| `career_avg` | Overall career batting average |
| `past_strike_rate` | Historical strike rate (runs per 100 balls) |
| `venue_avg` | Average runs at this specific venue |
| `vs_team_avg` | Average runs vs this specific opponent |
| `consistency` | Standard deviation of scores (lower = more consistent) |
| `form_trend` | Difference between last 3 avg and last 10 avg |
| `matches_played` | Total IPL experience (number of matches) |

### Bowler Features

| Feature | Description |
|---------|-------------|
| `avg_wickets_last5` | Average wickets in last 5 matches |
| `career_wickets_avg` | Overall career wicket average |
| `past_economy` | Historical economy rate |
| `venue_wickets_avg` | Average wickets at this specific venue |
| `vs_team_wickets_avg` | Average wickets vs this specific opponent |
| `bowl_consistency` | Standard deviation of wickets taken |
| `bowl_form_trend` | Difference between last 3 avg and last 10 avg |
| `matches_bowled` | Total IPL bowling experience |

---

## 🤖 Model Details

### Algorithm
- **XGBoost Classifier** inside an **Sklearn Pipeline**
- Pipeline includes: `SimpleImputer` → `StandardScaler` → `XGBClassifier`

### Why Pipeline?
Sklearn Pipelines package preprocessing and model together in one object — preventing data leakage between train/test splits and making deployment cleaner and more production-ready.

### Handling Class Imbalance
The batsman dataset had class imbalance — 70% "Under 30" vs 30% "30+". Fixed using `scale_pos_weight=2.36` in XGBoost which improved minority class recall from **5% to 73%**.

### Model Performance

| Model | Accuracy | ROC-AUC | Notes |
|-------|----------|---------|-------|
| Batsman Predictor | 57.71% | 0.656 | 5-fold CV: 68.99% ± 0.97% |
| Bowler Predictor | 58.89% | 0.560 | 5-fold CV: 59.47% ± 1.46% |

> **Note on accuracy:** Cricket has inherent natural variance — even world-class players are unpredictable match to match. These models provide data-driven probability estimates, not guaranteed predictions. Low R² in sports prediction is normal and expected even in professional sports analytics.

---

## 🚨 Key Data Science Challenges Solved

### 1. Data Leakage Detection and Fix
Initially got suspicious R² of 0.99 on training. Identified that `venue_avg` and `vs_team_avg` were computed including the current match — a classic leakage bug. Fixed by applying `shift(1)` before all rolling and expanding calculations.

### 2. Class Imbalance
Target variable had 70/30 imbalance. Standard model predicted "Under 30" almost always (recall = 0.05). Fixed with `scale_pos_weight` — recall improved to 0.73.

### 3. Zero-Inflated Bowler Target
42.7% of bowler matches had 0 wickets — making regression unreliable. Converted to binary classification (will take wicket or not) which gave more useful and stable predictions.

---

## 🖥️ Streamlit App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, dataset stats, model performance summary |
| 🏏 Batsman Predictor | Select player, opponent, venue → predict 30+ runs probability |
| 🎳 Bowler Predictor | Select bowler, opponent, venue → predict wicket probability |
| 📊 Player Analytics | Full career stats, season trends, venue analysis, score distribution |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Serving | Pickle, Sklearn Pipeline |

---

## 📁 Project Structure

```
ipl_predictor/
├── app.py                          ← Home page
├── pages/
│   ├── 1_Batsman_Predictor.py      ← Batsman prediction page
│   ├── 2_Bowler_Predictor.py       ← Bowler prediction page
│   └── 3_Player_Analytics.py       ← Player analytics page
├── batsman_pipeline.pkl            ← Trained batsman model
├── bowler_pipeline.pkl             ← Trained bowler model
├── batsman_features.json           ← Batsman feature list
├── bowler_features.json            ← Bowler feature list
├── players_data.json               ← Player/venue/team lists
├── batsman_data.csv                ← Processed batsman data
├── bowler_data.csv                 ← Processed bowler data
└── requirements.txt                ← Python dependencies
```

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/h4rshalk/IPL-Player-Performance-Predictor.git
cd IPL-Player-Performance-Predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
plotly
```

---

## 📈 Future Improvements

- Add SHAP explainability to show which features drove each prediction
- Include toss decision and weather data as additional features
- Build a Dream11 team suggester using both batsman and bowler predictions
- Add live IPL 2025 data integration using CricAPI
- Experiment with LSTM for time-series based form prediction

---

## 👨‍💻 About the Author

**Harshal**
- 🎓 B.E. Computer Engineering
- 📊 Data Science Enthusiast
- 🔗 [LinkedIn](www.linkedin.com/in/harshal-kawane-3375a4382)
- 💻 [GitHub](https://github.com/h4rshalk)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project useful, please consider giving it a star on GitHub!
