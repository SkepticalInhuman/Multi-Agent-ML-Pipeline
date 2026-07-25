import operator
import glob
import os
from dotenv import load_dotenv
from typing import Annotated, List, Literal, TypedDict
from pydantic import BaseModel, Field
import pandas as pd
from langgraph.graph import END, START, StateGraph
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from data_analyst import graph as analyst_graph
from data_engineer import graph as engineer_graph
from model_critic import run_pipeline
from langchain_groq import ChatGroq

load_dotenv()



# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class MainState(TypedDict):
    # Data paths
    dataset_path:      str
    cleaned_data_path: str
    target_col:        str
    feature_cols:      list[str]

    # Flow control
    user_input: str

    # Agent outputs
    eda_summary:   str
    plot_paths:    list[str]    # paths to plots saved by the EDA agent
    cleaning_plan: str
    model_results: dict   
    final_report:  str

    # Agent Messages
    eda_mess: Annotated[list[str], operator.add]
    fe_mess: Annotated[list[str], operator.add]



class RouteDecision(BaseModel):
    next_step: Literal["eda", "feature_engineering"]


class ColumnSelection(BaseModel):
    feature_columns: List[str] = Field(description="Columns used as model inputs")
    target_column:   str       = Field(description="Column to predict")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)
llm_structured = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def run_eda(state: MainState) -> dict:
    result = analyst_graph.invoke({
        "messages": HumanMessage(
            content=state["user_input"] if state.get("user_input")
                    else "Analyze the features and provide information"
        ),
        "dataset_path": state["dataset_path"],
    })

    #print(f"\nEDA report:\n{result['eda_output_to_user'][0]['text']}")
    eda_messages = [
        {"role": type(m).__name__, "name": getattr(m, "name", "") or "", "content": m.content}
        for m in result.get("messages", [])
    ]
    
    PLOTS_DIR = "eda_plots"   
    if os.path.isdir(PLOTS_DIR):
        plot_paths = sorted(
            glob.glob(os.path.join(PLOTS_DIR, "*.png"))  +
            glob.glob(os.path.join(PLOTS_DIR, "*.jpg"))  +
            glob.glob(os.path.join(PLOTS_DIR, "*.jpeg"))
        )
    else:
        plot_paths = []

    if state['eda_summary']:
        return{
            "eda_mess": [eda_messages[-1]],
            "plot_paths":   plot_paths
        }
    return {
        "eda_summary": result["eda_output_to_user"],
        "eda_mess":    [eda_messages[-1]],
        "plot_paths":   plot_paths
        
    }


def eda_router(state: MainState) -> Command[Literal["run_eda", "run_fe"]]:
    user_input = interrupt(
        "EDA complete. Ask another EDA question or tell me to continue to feature engineering."
    )

    decision = llm.with_structured_output(RouteDecision).invoke(
        f"""
        User response:
        {user_input}

        The user can either ask an EDA question or can decide to move on to feature engineering. Only if the user
        asks an EDA question return eda, anything else return feature_engineering.
        Decide whether the user wants:
        - eda
        - feature_engineering
        """
    )

    if decision.next_step == "eda":
        return Command(goto="run_eda", update={"user_input": user_input})

    return Command(goto="run_fe", update={"user_input": ""})


def run_fe(state: MainState) -> dict:
    result = engineer_graph.invoke({
        "messages":           HumanMessage(content="Clean the dataset"),
        "dataset_path":       state["dataset_path"],
        "eda_output_to_user": state["eda_summary"],
    })

    #print(f"\nData cleaning and feature engineering report:\n{result['cleaning_plan']}")

    fe_messages = [
        {"role": type(m).__name__, "name": getattr(m, "name", "") or "", "content": m.content}
        for m in result.get("messages", [])
    ]
    print(fe_messages[-2])

    return {
        "cleaning_plan":      result["cleaning_plan"],
        "fe_mess":            [fe_messages[-1]]
        #"cleaned_data_path":  result["cleaned_data_path"],
    }


def select_columns(state: MainState) -> dict:
    import time
    time.sleep(5)
    df      = pd.read_csv(state["cleaned_data_path"])
    columns = df.columns.tolist()

    user_input = interrupt(
        f"""
    Feature engineering complete. Please specify your modelling columns.

    Available columns: {', '.join(columns)}

    Please specify:
    - which column is the target
    - which columns should be used as features

    Example:
    Target: price
    Features: area, bedrooms, bathrooms
        """
    )

    selection = llm_structured.with_structured_output(ColumnSelection).invoke(
        f"""
    Available columns: {columns}

    User response: {user_input}

    Extract:
    1. feature_columns — only columns that exist in the available columns list
    2. target_column   — only a column that exists in the available columns list
        """
    )

    return {
        "feature_cols": selection.feature_columns,
        "target_col":   selection.target_column,
    }

# Module-level store for models (not msgpack-serializable, kept out of state)
_MAIN_MODELS: dict = {}

def run_mc(state: MainState) -> dict:
    df     = pd.read_csv(state["cleaned_data_path"])
    result = run_pipeline(
        df,
        target_col=state["target_col"],
        feat_cols=state["feature_cols"],
        max_iterations=3,
        optuna_trials=20,
    )
    # Pull out non-serializable objects before storing result in state
    _MAIN_MODELS["winning_model"] = result.pop("winning_model", None)
    _MAIN_MODELS["all_models"]    = result.pop("all_models", {})

    return {"model_results": result}


def display_results(state: MainState) -> dict:
    result = state["model_results"]
    winner = result["winner"]
    winning = result["results"][winner]

    lines = [
        "=" * 60,
        "MODELLING COMPLETE",
        "=" * 60,
        f"  Winning branch  : {winner}  ({winning['algorithm']})",
        f"  CV metric used  : {result['cv_metric']}",
        f"  Optuna CV score : {winning['optuna_score']}",
        f"  Best params     : {winning['best_params']}",
        f"  Test metrics    : {winning['metrics']}",
        "",
        "All branches:",
    ]
    for b, r in result["results"].items():
        marker = " ← winner" if b == winner else ""
        lines.append(f"  [{b}] {r['algorithm']:22s} | {r['metrics']}{marker}")

    report = "\n".join(lines)
    print(report)

    return {"final_report": report}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_main_graph():
    g = StateGraph(MainState)

    g.add_node("run_eda",        run_eda)
    g.add_node("eda_router",     eda_router)
    g.add_node("run_fe",         run_fe)
    g.add_node("select_columns", select_columns)
    g.add_node("run_mc",         run_mc)
    g.add_node("display_results",display_results)

    g.add_edge(START,              "run_eda")
    g.add_edge("run_eda",          "eda_router")
    # eda_router uses Command so its edges are declared implicitly via goto
    g.add_edge("run_fe",           "select_columns")
    g.add_edge("select_columns",   "run_mc")
    g.add_edge("run_mc",           "display_results")
    g.add_edge("display_results",  END)

    memory = InMemorySaver()
    return g.compile(checkpointer=memory)


main_graph = build_main_graph()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_main(dataset_path: str, thread_id: str = "main-1") -> dict:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}

    initial_state: MainState = {
        "dataset_path":      dataset_path,
        "cleaned_data_path": "datasets/cleaned.csv",
        "target_col":        "",
        "feature_cols":      [],
        "user_input":        "",
        "eda_summary":       "",
        "cleaning_plan":     "",
        "model_results":     {},
        "final_report":      "",
    }

    print("Starting pipeline. Resuming after each interrupt with your input.\n")

    # ── First run: execute until first interrupt (after EDA) ─────────────────
    state = main_graph.invoke(initial_state, config=config)

    # ── Resume loop: handle all interrupts until graph finishes ──────────────
    while state.get("__interrupt__"):
        interrupt_msg = state["__interrupt__"][0].value
        print(f"\n[INPUT REQUIRED]\n{interrupt_msg}\n")
        user_response = input("Your response: ").strip()
        state = main_graph.invoke(Command(resume=user_response), config=config)

    print("\nPipeline complete.")
    return state


if __name__ == "__main__":
    final = run_main("datasets/ai_student.csv")
    print(f"\nFinal report:\n{final['final_report']}")
    print(f"\nWinning model object: {_MAIN_MODELS.get('winning_model')}")
    print(f"All models: { {b: type(m).__name__ for b, m in _MAIN_MODELS.get('all_models', {}).items()} }")





