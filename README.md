# Multi-Agent Data Science Pipeline

An end-to-end automated data science pipeline built with LangGraph. Upload a CSV, interact with an EDA agent, clean your data with a feature engineering agent, then watch three parallel ML models get tuned by Optuna and evaluated by a Critic agent — all through a Streamlit interface.

## How it works

```
CSV Upload → EDA Agent → Feature Engineering Agent → Column Selection → AutoML → Results
```

- **EDA Agent** — analyses your dataset and answers follow-up questions
- **Feature Engineering Agent** — cleans and prepares the data
- **3 Parallel Modeller Agents** — each proposes Optuna hyperparameter search spaces via an LLM and trains a different algorithm (Logistic Regression / HistGradientBoosting / LightGBM)
- **Critic Agent** — compares all three models, detects overfitting, and sends feedback for the next tuning round
- **Human-in-the-loop** — you stay in control via interrupts between each stage

## Requirements

- Python 3.10+
- A [Groq API key] (free tier works)
- A [Google Gemini API key] (free tier works)

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install uv
uv sync
```

**3. Set your API key**

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

**4. Run the app**
```bash
uv run streamlit run app.py
```

## Usage

1. Upload a cleaned CSV file using the sidebar
2. Click **Start Pipeline**
3. Chat with the EDA agent — ask questions about your data or type *"continue to feature engineering"* to move on
4. The feature engineering agent will clean your data automatically
5. Select your target and feature columns when prompted
6. Watch the three models train and tune in the terminal
7. View results and download any of the three trained models as `.pkl` files

## Configuration

You can adjust these values at the top of `ml_modeller_critic_langgraph.py`:

| Variable | Default | Description |
|---|---|---|
| `MAX_ITERATIONS` | `3` | Number of Critic feedback rounds |
| `OPTUNA_TRIALS` | `20` | Optuna trials per model per round |
| `LLM_PROVIDER` | `"groq"` | `"groq"` or `"gemini"` |

## Models

| Branch | Classification | Regression |
|---|---|---|
| A | Logistic Regression | ElasticNet |
| B | HistGradientBoosting | HistGradientBoosting |
| C | LightGBM | LightGBM |

## Loading a downloaded model

```python
import pickle

with open("model_c_lightgbm.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_new)
```

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [Groq](https://groq.com) — LLM inference (`openai/gpt-oss-120b`)
- [Gemini](https://aistudio.google.com/welcome) - LLM inference(`gemini-3.5-flash-lite`)
- [Optuna](https://optuna.org) — hyperparameter optimisation
- [LightGBM](https://lightgbm.readthedocs.io) — gradient boosting
- [scikit-learn](https://scikit-learn.org) — ML utilities and models
- [Streamlit](https://streamlit.io) — frontend
