# Stock Price Predictor

A web app that lets you enter any stock ticker and get next-day price predictions using either an LSTM neural network, ARIMA, or both — side by side.

Built with Flask on the backend and a dark-themed UI on the front. The chart shows the last 90 days of historical closing prices alongside the model's prediction point, so you can visually judge how it fits in context.

---

## What it does

- **LSTM model** — trains on the last year of daily closing prices, uses a 60-day lookback window, and generates SHAP explanations showing which past days influenced the prediction most
- **ARIMA model** — fits a statistical time-series model on the same data for a second opinion
- **Portfolio tracker** — create portfolios, log buy/sell transactions, and track current holdings with live prices
- **Dark/light theme toggle** — persists across sessions via a cookie
- User accounts with signup, login, and bcrypt-hashed passwords

---

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Flask, Flask-Login, Flask-SQLAlchemy, Flask-Bcrypt |
| ML / Stats | TensorFlow (LSTM), statsmodels (ARIMA), scikit-learn |
| Data | yfinance |
| Explainability | SHAP (KernelExplainer) |
| Charts | Plotly |
| Database | SQLite |

---

## Getting started

**1. Clone and set up a virtual environment**

```bash
git clone https://github.com/Kshitijkediya/stock-price-predictor.git
cd stock-price-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the app**

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser. Create an account and start predicting.

> The first prediction for a ticker will take a moment — it's training the LSTM and saving the model locally. Subsequent runs for the same ticker load the saved model and are much faster.

---

## Project structure

```
stock-price-predictor/
├── app.py              # Flask routes and app config
├── stock_data.py       # Data fetching, model training, predictions, charts
├── models.py           # SQLAlchemy models (User, Portfolio, Transaction)
├── templates/          # Jinja2 HTML templates
├── static/             # CSS, JS, static assets
├── models/             # Saved LSTM .keras model files (auto-created)
└── instance/           # SQLite database (auto-created)
```

---

## Notes

- Predictions are for the **next trading day** only — this is not financial advice
- SHAP explanations on the LSTM use `KernelExplainer`, which can be slow on first run
- The ARIMA model tries `(5,1,0)` first and falls back to `(1,1,0)` if it doesn't converge
