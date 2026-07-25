import json
import operator
import warnings
from typing import Annotated, Any, Literal, TypedDict
from dotenv import load_dotenv

import numpy as np
import optuna
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.runtime import Runtime
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from sklearn.utils.multiclass import type_of_target
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression,ElasticNet
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

load_dotenv()

MAX_ITERATIONS = 3
OPTUNA_TRIALS  = 20

MODEL_REGISTRY = {
    "classification": {
        "logistic_regression":    LogisticRegression,
        "hist_gradient_boosting": HistGradientBoostingClassifier,
        "lightgbm":               LGBMClassifier,
    },
    "regression": {
        "elasticnet":             ElasticNet,
        "hist_gradient_boosting": HistGradientBoostingRegressor,
        "lightgbm":               LGBMRegressor,
    },
}

from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start: end + 1])


# Module-level store for fitted models (kept out of state to avoid serialization issues)
_FITTED_MODELS: dict[str, Any] = {}


class BranchResult(TypedDict):
    """Holds everything produced by one modeller branch."""
    algorithm:    str
    search_space: dict
    best_params:  dict
    optuna_score: float
    metrics:      dict


class PipelineState(MessagesState):
    # ── static data (set once at pipeline start) ───────────────────────────
    task_type:    Literal["classification", "regression"]
    target_col:   str
    feature_cols: list[str]
    df_profile:   str   # pre-computed string: describe/info/head/value_counts etc. of target

    # ── loop control ───────────────────────────────────────────────────────
    iteration:      int
    max_iterations: int
    approved:       bool
    winner:         str
    cv_metric:      str   # chosen by metric_selector_node

    # ── per-branch results (dict-union reducer — each branch writes its key)
    results: Annotated[dict[str, BranchResult], operator.or_]


# Valid sklearn scoring strings the LLM may choose from
VALID_METRICS = {
    "classification": [
        "accuracy", "f1_weighted", "f1_macro", "f1_binary",
        "roc_auc", "roc_auc_ovr_weighted", "balanced_accuracy",
        "average_precision",
    ],
    "regression": [
        "r2", "neg_mean_squared_error", "neg_root_mean_squared_error",
        "neg_mean_absolute_error", "neg_mean_absolute_percentage_error",
    ],
}

SYSTEM_PROMPT = """You are an expert ML engineer operating inside an automated
pipeline. You respond ONLY with valid JSON — no preamble, no markdown fences,
no explanation outside the JSON object."""


def metric_selector_node(state: PipelineState) -> dict:
    """
    Passes state["df_profile"] — a pre-computed string containing dataset and target metadata.
     — directly to the LLM and asks it to pick
    the most appropriate CV metric from the valid sklearn list.
    Runs once before any modeller node.
    """
    task_type = state["task_type"]
    valid     = VALID_METRICS[task_type]

    prompt = f"""You are a senior ML engineer selecting the best cross-validation
metric for an automated Optuna hyperparameter search.

Task type     : {task_type}
Target column : {state["target_col"]}
Valid choices : {valid}

Dataset profile:
{state["df_profile"]}

Based on what you observe in the profile above, pick the single best metric
from the valid choices list. Things to consider:
- Classification: prefer balanced_accuracy or f1_macro for imbalanced classes,
  roc_auc for binary problems, accuracy only when classes are balanced.
- Regression: prefer neg_root_mean_squared_error for well-behaved targets,
  neg_mean_absolute_error when outliers are evident, 
  neg_mean_absolute_percentage_error when relative error matters more than absolute.

Respond with JSON only:
{{"metric": "<one value from the valid choices list>", "reason": "<1 sentence>"}}"""

    call_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt, name="metric_selector"),
    ]
    resp    = llm.invoke(call_messages)
    verdict = _extract_json(resp.content)

    chosen = verdict.get("metric", "")
    # Guard: fall back to a safe default if the LLM goes off-list
    if chosen not in valid:
        chosen = "f1_weighted" if task_type == "classification" else "r2"
        verdict["reason"] = f"LLM chose an invalid metric; fell back to {chosen}."

    print(f"[MetricSelector] cv_metric={chosen} | {verdict.get('reason', '')}")

    info_msg = AIMessage(
        content=json.dumps({"cv_metric": chosen, "reason": verdict.get("reason", "")}),
        name="metric_selector",
    )
    return {
        "messages":  [info_msg],
        "cv_metric": chosen,
    }


# ---------------------------------------------------------------------------
# Optuna objective builder
# ---------------------------------------------------------------------------
def _build_objective(algo_name, search_space, task_type, cv_metric, X_train, y_train):
    ModelClass = MODEL_REGISTRY[task_type][algo_name]

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, spec in search_space.items():
            kind = spec.get("type", "float")
            if kind == "int":
                params[name] = trial.suggest_int(
                    name, int(spec["low"]), int(spec["high"]),
                    log=bool(spec.get("log", False)))
            elif kind == "float":
                params[name] = trial.suggest_float(
                    name, float(spec["low"]), float(spec["high"]),
                    log=bool(spec.get("log", False)))
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        try:
            model = ModelClass(**params)
            if algo_name == "lightgbm":
                model.set_params(verbosity=-1, boosting_type="gbdt")
        except (TypeError, ValueError):
            raise optuna.exceptions.TrialPruned()
        scores = cross_val_score(model, X_train, y_train,
                                  cv=3, scoring=cv_metric, n_jobs=-1)
        return float(np.mean(scores))

    return objective


# ---------------------------------------------------------------------------
# Shared branch logic
# ---------------------------------------------------------------------------
def _train_branch(state: PipelineState, branch: str, runtime:Runtime, algo_name: str) -> dict:
    """
    1. Build message history for this branch from `state["messages"]`
    2. LLM proposes a search space
    3. Optuna tunes within it
    4. Refit + evaluate; append result as AIMessage
    """
    if algo_name == "logistic_regression":
        extra_constraints = """
        - LogisticRegression:
        - Include 'solver' parameter and it should only have 'saga' in the search space.
        - penalty choices must be [l1, l2, elasticnet].
        - do not include 'warm_start', 'precompute', 'positive' in the search space
        """

    elif algo_name == "hist_gradient_boosting":
        extra_constraints = """
        - HistGradientBoosting:
        - valid params: learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization, max_bins.
        - l2_regularization must be >= 0.
        - max_bins must be an int between 2 and 255.
        """

    elif algo_name == "lightgbm":
        extra_constraints = """
        - LightGBM:
        - do not include boosting_type, goss, bagging_freq, min_child_weight.
        """

    else:
        extra_constraints = ""
    
    task_type = state["task_type"]

    # ── Collect branch-specific messages from shared history ────────────────
    branch_msgs = [m for m in state["messages"]
                   if getattr(m, "name", None) in (branch, "critic", None)
                   or isinstance(m, SystemMessage)]

    # ── Build the new user turn ─────────────────────────────────────────────
    prev_result  = state.get("results", {}).get(branch)
    prev_metrics = prev_result["metrics"]      if prev_result else {}
    prev_space   = prev_result["search_space"] if prev_result else {}

    user_content = f"""You are Modeller-{branch.upper()} (algorithm: {algo_name}).

Task type     : {task_type}
Dataset profile : {state["df_profile"]}
Iteration     : {state["iteration"] + 1} / {state["max_iterations"]}
Previous test metrics  : {json.dumps(prev_metrics)}
Previous search space  : {json.dumps(prev_space)}

Propose a HYPERPARAMETER SEARCH SPACE for an Optuna study ({OPTUNA_TRIALS} trials).
Use critic feedback above (if any) to shift or widen/narrow ranges intelligently.
Every param must be a valid scikit-learn constructor argument for {algo_name}.
Look at the dataset profile and decide what parameters to tune. Include only necessary parameters 
based on the dataset.
Use "log": true for params spanning orders of magnitude (C, alpha, learning_rate).
Important:

- Every parameter combination must be valid.

{extra_constraints}

Return only parameters guaranteed to be valid.

Respond with JSON only:
{{
  "search_space": {{
    "<param>": {{"type": "int"|"float"|"categorical",
                 "low": <number>, "high": <number>, "log": false}},
    "<param>": {{"type": "categorical", "choices": [...]}}
  }},
  "rationale": "<1-2 sentences>"
}}"""

    call_messages = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + branch_msgs
        + [HumanMessage(content=user_content, name=branch)]
    )

    resp     = llm.invoke(call_messages)
    decision = _extract_json(resp.content)

    search_space = decision.get("search_space", {})
    rationale    = decision.get("rationale", "")

    print(f"[Modeller-{branch}] iter {state['iteration']+1}: {algo_name} | "
          f"tuning {list(search_space.keys())} over {OPTUNA_TRIALS} trials"
          f"Rationale: {rationale}")

    # ── Optuna ──────────────────────────────────────────────────────────────
    study = optuna.create_study(direction="maximize", 
                                sampler=optuna.samplers.TPESampler(seed=42, multivariate=False))
    study.optimize(_build_objective(algo_name, search_space, task_type, state["cv_metric"],
                          runtime.context["X_train"], runtime.context["y_train"]),
                          n_trials=OPTUNA_TRIALS, show_progress_bar=False,
    )
    best_params = study.best_trial.params
    best_cv     = round(study.best_trial.value, 4)

    # ── Refit + test metrics ────────────────────────────────────────────────
    ModelClass = MODEL_REGISTRY[task_type][algo_name]
    try:
        model = ModelClass(**best_params)
        if algo_name == "lightgbm":
            model.set_params(verbosity=-1, boosting_type="gbdt")
    except (TypeError, ValueError):
        model, best_params = ModelClass(), {}

    model.fit(runtime.context["X_train"], runtime.context["y_train"])
    preds   = model.predict(runtime.context["X_test"])
    metrics: dict = {}

    if task_type == "classification":
        metrics["accuracy"]       = round(accuracy_score(runtime.context["y_test"], preds), 4)
        metrics["f1"]             = round(f1_score(runtime.context["y_test"], preds, average="weighted"), 4)
        metrics["train_accuracy"] = round(accuracy_score(
            runtime.context["y_train"], model.predict(runtime.context["X_train"])), 4)
        if hasattr(model, "predict_proba") and len(np.unique(runtime.context["y_train"])) == 2:
            try:
                proba = model.predict_proba(runtime.context["X_test"])[:, 1]
                metrics["roc_auc"] = round(roc_auc_score(runtime.context["y_test"], proba), 4)
            except Exception:
                pass
    else:
        metrics["rmse"]     = round(mean_squared_error(runtime.context["y_test"], preds) ** 0.5, 4)
        metrics["mae"]      = round(mean_absolute_error(runtime.context["y_test"], preds), 4)
        metrics["r2"]       = round(r2_score(runtime.context["y_test"], preds), 4)
        metrics["train_r2"] = round(r2_score(
            runtime.context["y_train"], model.predict(runtime.context["X_train"])), 4)

    print(f"[Modeller-{branch}] best_cv={best_cv} | test={metrics}")

    result: BranchResult = {
        "algorithm":    algo_name,
        "search_space": search_space,
        "best_params":  best_params,
        "optuna_score": best_cv,
        "metrics":      metrics,
    }

    # Store fitted model outside state (sklearn models aren't msgpack-serializable)
    _FITTED_MODELS[branch] = model

    # ── Append result as AIMessage (tagged with branch name) ────────────────
    ai_msg = AIMessage(
        content=json.dumps({
            "branch": branch, "algorithm": algo_name,
            "search_space": search_space, "best_params": best_params,
            "optuna_cv": best_cv, "metrics": metrics, "rationale": rationale,
        }),
        name=branch,
    )

    return {
        "messages": [ai_msg],          # add_messages reducer appends this
        "results":  {branch: result},  # dict-union reducer merges this
    }


# ---------------------------------------------------------------------------
# Modeller nodes
# ---------------------------------------------------------------------------
def modeller_a_node(state: PipelineState, runtime:Runtime) -> dict:
    algo = "logistic_regression" if state["task_type"] == "classification" else "elasticnet"
    return _train_branch(state, "a", runtime, algo)


def modeller_b_node(state: PipelineState, runtime:Runtime) -> dict:
    return _train_branch(state, "b", runtime, "hist_gradient_boosting")


def modeller_c_node(state: PipelineState, runtime:Runtime) -> dict:
    return _train_branch(state, "c", runtime, "lightgbm")

# ---------------------------------------------------------------------------
# Critic node
# ---------------------------------------------------------------------------

def _primary_metric(task_type: str, metrics: dict) -> float:
    return (metrics.get("f1", metrics.get("accuracy", 0))
            if task_type == "classification"
            else metrics.get("r2", -999))


def critic_node(state: PipelineState) -> dict:
    results   = state["results"]
    task_type = state["task_type"]
    iteration = state["iteration"] + 1

    summary = {}
    for branch, r in results.items():
        summary[branch] = {
            "algorithm":    r["algorithm"],
            "best_params":  r["best_params"],
            "optuna_cv":    r["optuna_score"],
            "test_metrics": r["metrics"],
        }

    # ── The critic sees the full shared message history ──────────────────
    critic_prompt = f"""You are the Critic. Review these 3 Optuna-tuned models (iter {iteration}/{state["max_iterations"]}):

{json.dumps(summary, indent=2)}

1. Pick the best branch ("a","b","c") — reward test performance, compare train vs test metrics in each result to detect overfitting.
2. Decide if the winner is good enough to ship.
3. Give SHORT, search-space-level feedback for EACH branch (shift ranges, regularise, etc.).
{"If this is the last iteration, approve unless results are clearly broken." if iteration >= state["max_iterations"] else ""}

Respond with JSON only:
{{
  "winner": "a|b|c",
  "approved": true|false,
  "overall_feedback": "<1-2 sentences>",
  "feedback_a": "<advice on search space for branch a>",
  "feedback_b": "<advice on search space for branch b>",
  "feedback_c": "<advice on search space for branch c>"
}}"""

    call_messages = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + [HumanMessage(content=critic_prompt, name="critic")]
    )

    resp    = llm.invoke(call_messages)
    verdict = _extract_json(resp.content)

    winner = (verdict.get("winner")
              if verdict.get("winner") in results
              else max(results, key=lambda b: _primary_metric(task_type, results[b]["metrics"])))

    approved = bool(verdict.get("approved", False))
    if iteration >= state["max_iterations"]:
        approved = True
        verdict.setdefault("overall_feedback", "Max iterations reached; approving best result.")

    print(f"[Critic] iter {iteration}: winner={winner} approved={approved} "
          f"| {verdict.get('overall_feedback')}")

    # ── Append critic verdict + per-branch feedback as HumanMessages ─────
    # These will be seen by each modeller node on the next iteration.
    new_msgs = [
        AIMessage(content=json.dumps(verdict), name="critic"),
        HumanMessage(content=f"[Feedback for branch a]: {verdict.get('feedback_a', '')}", name="critic"),
        HumanMessage(content=f"[Feedback for branch b]: {verdict.get('feedback_b', '')}", name="critic"),
        HumanMessage(content=f"[Feedback for branch c]: {verdict.get('feedback_c', '')}", name="critic"),
    ]

    return {
        "messages":  new_msgs,
        "winner":    winner,
        "approved":  approved,
        "iteration": iteration,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def route_after_critic(state: PipelineState) -> list[str]:
    if state["approved"] or state["iteration"] >= state["max_iterations"]:
        return [END]
    return ["modeller_a", "modeller_b", "modeller_c"]


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(PipelineState)
 
    g.add_node("metric_selector", metric_selector_node)
    g.add_node("modeller_a", modeller_a_node)
    g.add_node("modeller_b", modeller_b_node)
    g.add_node("modeller_c", modeller_c_node)
    g.add_node("critic",     critic_node)
 
    # metric selector runs once, then fans out to all three modellers
    g.add_edge(START,              "metric_selector")
    g.add_edge("metric_selector",  "modeller_a")
    g.add_edge("metric_selector",  "modeller_b")
    g.add_edge("metric_selector",  "modeller_c")
    g.add_edge("modeller_a", "critic")
    g.add_edge("modeller_b", "critic")
    g.add_edge("modeller_c", "critic")
 
    g.add_conditional_edges(
        "critic", route_after_critic,
        {"modeller_a": "modeller_a", "modeller_b": "modeller_b",
            "modeller_c": "modeller_c", END: END},
    )
    memory = InMemorySaver()
    return g.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    feat_cols:list | None = None,
    task_type: Literal["classification", "regression"] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iterations: int = MAX_ITERATIONS,
    optuna_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Run the parallel Modeller/Optuna/Critic loop on a cleaned dataframe.

    df            : cleaned numeric dataframe (encode categoricals beforehand)
    target_col    : name of the target column
    task_type     : 'classification' or 'regression'; auto-inferred if None
    optuna_trials : Optuna trials per branch per iteration (default 30)
    """
    global OPTUNA_TRIALS
    OPTUNA_TRIALS = optuna_trials

    feature_cols = feat_cols if feat_cols else [c for c in df.columns if c != target_col]
    X, y = df[feature_cols], df[target_col]

    def infer_task_type(y):
        target_type = type_of_target(y)

        if target_type in ("binary", "multiclass"):
            return "classification"

        if target_type == "continuous":
            return "regression"

        return "unsupported"

    if task_type is None:
        task_type = infer_task_type(y)
    
    def build_modeling_profile(X: pd.DataFrame, y: pd.Series, task_type: str) -> str:
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        datetime_cols = [
            col for col in X.columns
            if is_datetime64_any_dtype(X[col])
        ]

        target_series = y
        n_rows        = len(df)
        n_features    = X.shape[1]

        profile = {
            "n_rows":             n_rows,
            "n_features":         n_features,
            "n_numeric_features": len(numeric_cols),
            "n_categorical_features": len(categorical_cols),
            "n_datetime_features":    len(datetime_cols),
            "missing_percentage": round(
                100 * df.isna().sum().sum() / max(df.size, 1), 2),
            "high_cardinality_features": {
                col: int(df[col].nunique(dropna=True))
                for col in categorical_cols
                if df[col].nunique(dropna=True) > 50
            },
            "task_type":           task_type,
            "target_name":         y.name,
            "feature_to_row_ratio": round(n_features / max(n_rows, 1), 4),
            "contains_datetime":   len(datetime_cols) > 0,
            "n_classes": (
                int(target_series.nunique())
                if task_type == "classification" else None
            ),
            "class_distribution": (
                target_series.value_counts(normalize=True).round(4).to_dict()
                if task_type == "classification" else None
            ),
            "majority_class_ratio": (
                round(target_series.value_counts(normalize=True).max(), 4)
                if task_type == "classification" else None
            ),
            "target_skew": (
                round(float(target_series.skew()), 4)
                if task_type == "regression" else None
            ),
        }

        return json.dumps(profile, indent=2)

    df_profile = build_modeling_profile(X, y, task_type)

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    seed_message = SystemMessage(
        content=(
            f"ML pipeline started. Task: {task_type}. "
            f"Target: {target_col}. Features: {feature_cols}. "
            f"Train size: {len(X_train)}, Test size: {len(X_test)}."
        )
    )

    initial_state: PipelineState = {
        "messages":      [seed_message],
        "task_type":     task_type,
        "target_col":    target_col,
        "feature_cols":  feature_cols,
        "df_profile":    df_profile,
        "iteration":      0,
        "max_iterations": max_iterations,
        "approved":       False,
        "winner":         "",
        "cv_metric":      "",
        "results":        {},
    }

    graph       = build_graph()
    final_state = graph.invoke(
        initial_state,
        config  = {"configurable": {"thread_id": "1"}, "recursion_limit": 25},
        context = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
    )

    winner  = final_state["winner"]
    winning = final_state["results"][winner]

    final_state["winning_model"] = _FITTED_MODELS.get(winner)
    final_state["all_models"]    = dict(_FITTED_MODELS)
    return final_state


if __name__ == "__main__":
    df = pd.read_csv('datasets/ai_student.csv')
    run_pipeline(
        df, target_col='Post_Semester_GPA',
        feat_cols=['Pre_Semester_GPA', 'Weekly_GenAI_Hours',
                   'Traditional_Study_Hours', 'Anxiety_Level_During_Exams'],
        max_iterations=3, optuna_trials=20,
    )