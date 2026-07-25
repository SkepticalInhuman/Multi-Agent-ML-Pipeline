from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pprint import pprint
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import json
from typing import Dict, Any, Literal
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import seaborn as sns
import os

from kernel_tool import sandbox


load_dotenv()

class InteractiveDataScienceState(MessagesState):
    # Data Paths
    dataset_path: str
    
    # Flow Control Flags
    user_input: str
    
    # Internal agent history
    eda_output_to_user: str


@tool
def run_python_code(code: str) -> str:
    """Executes Python code in an isolated local Jupyter kernel sandbox and returns the stdout or errors. 
    Use this to inspect data, clean datasets, train models, and print evaluation metrics."""
    return sandbox.execute_code(code)
 
@tool
def load_and_profile_data(dataset_path: str) -> str:
    """
    Loads a dataset and generates a comprehensive profile for EDA and preprocessing.

    Creates:
    - Dataset overview (rows, columns, column names)
    - Data types for all columns
    - Missing value counts and percentages
    - Unique value counts for each column
    - Duplicate row count
    - Lists of numerical, categorical, and datetime columns
    - Constant column detection
    - Numerical summary statistics using df.describe()
    - Categorical summary statistics using df.describe()
    - Top 10 correlated numerical feature
    - Sample rows for data inspection

    Returns:
    - JSON string with all the necessary information of the dataframe

    """

    df = pd.read_csv(dataset_path)

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
    corr = df[numeric_cols].corr() if len(numeric_cols) > 1 else None

    profile = {
        # Dataset overview
        "rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),

        # Schema
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },

        # Missing values
        "null_counts": df.isnull().sum().to_dict(),
        "null_percentages": (
            df.isnull().mean() * 100
        ).round(2).to_dict(),

        # Cardinality
        "unique_values": df.nunique().to_dict(),

        # Duplicates
        "duplicate_rows": int(df.duplicated().sum()),

        # Column groups
        "numeric_columns": list(numeric_cols),
        "categorical_columns": list(categorical_cols),
        "datetime_columns": list(datetime_cols),

        # Constant columns
        "constant_columns": [
            col for col in df.columns
            if df[col].nunique(dropna=False) <= 1
        ],

        # Numeric summary statistics
        "numeric_summary": (
            df.describe()
            .round(4)
            .to_dict()
            if len(numeric_cols) > 0
            else {}
        ),

        # Categorical summary statistics
        "categorical_summary": (
            df[categorical_cols]
            .describe()
            .to_dict()
            if len(categorical_cols) > 0
            else {}
        ),

        #Top 10 correlated pairs
        "top_correlated_pairs": (
        sorted(
            [
                {
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "correlation": round(corr.iloc[i, j], 4),
                }
                for i in range(len(corr.columns))
                for j in range(i + 1, len(corr.columns))
            ],
            key=lambda x: abs(x["correlation"]),
            reverse=True,
        )[:10]
        if corr is not None
        else []
    ),

        # Sample rows for context
        "sample_rows": (
            df.head(5)
            .fillna("NULL")
            .to_dict(orient="records")
        ),

        # Percentage of missing values per column
        "null_percentage": (
            (df.isnull().sum() / len(df)) * 100
        ).round(2).to_dict(),

        # Skewness of numerical columns
        "skewness": (
            df.select_dtypes(include="number")
            .skew()
            .round(4)
            .to_dict()
        )
    }

    return json.dumps(profile, default=str)

@tool
def generate_eda_visualizations(
    dataset_path: str
) -> str:
    """
    Generate essential EDA visualizations and save them as PNG files.

    Creates:
    - Histograms for numerical columns
    - Boxplots for numerical columns
    - Correlation heatmap
    - Countplots for categorical columns
    - Pairplot (if number of numerical columns <= 10)

    Returns:
        'Visualization tool executed plots saved successfully'
    """

    df = pd.read_csv(dataset_path)
    output_dir = 'eda_plots'

    os.makedirs(output_dir, exist_ok=True)

    generated_plots = {}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # ---------------------------
    # Histograms
    # ---------------------------
    histogram_paths = []

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=30)

        plt.title(f"Histogram - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        path = os.path.join(
            output_dir,
            f"histogram_{col}.png"
        )

        plt.savefig(path)
        plt.close()

        histogram_paths.append(path)

    generated_plots["histograms"] = histogram_paths

    # ---------------------------
    # Boxplots
    # ---------------------------
    boxplot_paths = []

    for col in numeric_cols:
        plt.figure(figsize=(6, 3))

        sns.boxplot(x=df[col])

        plt.title(f"Boxplot - {col}")
        plt.tight_layout()

        path = os.path.join(
            output_dir,
            f"boxplot_{col}.png"
        )

        plt.savefig(path)
        plt.close()

        boxplot_paths.append(path)

    generated_plots["boxplots"] = boxplot_paths

    # ---------------------------
    # Correlation Heatmap
    # ---------------------------
    if len(numeric_cols) > 1:

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Heatmap")
        plt.tight_layout()

        path = os.path.join(
            output_dir,
            "correlation_heatmap.png"
        )

        plt.savefig(path)
        plt.close()

        generated_plots["correlation_heatmap"] = path

    # ---------------------------
    # Countplots
    # ---------------------------
    countplot_paths = []

    for col in categorical_cols:

        if df[col].nunique() <=20:

            plt.figure(figsize=(8, 4))

            sns.countplot(
                data=df,
                x=col,
                order=df[col].value_counts().index
            )

            plt.xticks(rotation=45)
            plt.title(f"Countplot - {col}")
            plt.tight_layout()

            path = os.path.join(
                output_dir,
                f"countplot_{col}.png"
            )

            plt.savefig(path)
            plt.close()

            countplot_paths.append(path)

    generated_plots["countplots"] = countplot_paths

    # ---------------------------
    # Pairplot
    # ---------------------------
    if 1 < len(numeric_cols) <= 10:

        pairplot_df = df[numeric_cols].dropna()

        if len(pairplot_df) > 5000:
            pairplot_df = pairplot_df.sample(
                5000,
                random_state=42
            )

        pairplot = sns.pairplot(pairplot_df)

        path = os.path.join(
            output_dir,
            "pairplot.png"
        )

        pairplot.savefig(path)
        plt.close("all")

        generated_plots["pairplot"] = path

    return 'Visualization tool executed plots saved successfully'


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
llm_with_tools = llm.bind_tools([run_python_code,load_and_profile_data,generate_eda_visualizations])


def interactive_eda_node(state: InteractiveDataScienceState) -> Dict[str, Any]:
    print("\n[Node: EDA Agent] Processing your dataset...")
    
    # If the user has just provided custom instructions, execute them
    user_request = state.get("user_input", "Run the load_and_profile_data and generate_eda_visualizations tools and give me the summary.")
    
    sys_msg = SystemMessage(content=(f"""You are an EDA Agent. The dataset is at '{state['dataset_path']}'.
    The user wants to see: '{user_request}'
    Use the load_and_profile_data and generate_eda_visualizations for EDA. Use the run_python_code only if 
    the user wants information about the csv or a plot that can't be provided by the load_and_profile_data
    and the generate_eda_visualizations tools. Show the entire output from the load_and_profile_data to the user beautifully.
    Do not provide any information about the plots.

    IF run_python_code IS USED:
    To generate a plot then save it locally using `plt.savefig('eda_plots/name.png')`. Replace name in the file location of plt.savefig() with the plots name.
    If a matrix or list is large, print only the length or slice it (e.g., print(matrix[:5])).
    Only use pandas,numpy,seaborn and matplotlib. Do not use any other modules. Do not access any other file
    other than the dataset.
    Do not modify the csv. Only do EDA and visualization. If user requests anything else, do not do that task.
    Never print an entire DataFrame. Always use .head(), .info(), .describe(), or .shape to inspect data.
  
    """))

    #messages = state["messages"].copy()

    # Find the last AIMessage (the summary)
    #for i in range(len(messages) - 1, -1, -1):
        #if isinstance(messages[i], AIMessage):
            #ai_idx = i
            #break
    #else:
        #ai_idx = None

    #if ai_idx is not None:
        # Remove all consecutive ToolMessages immediately before it
        #j = ai_idx - 1
        #while j >= 0 and isinstance(messages[j], ToolMessage):
            #del messages[j]
            #ai_idx -= 1  # AI shifts left after deletion
            #j -= 1

    response = llm_with_tools.invoke([sys_msg] + state['messages'])
    
    #print(response)
    # (Execute tool calls here as shown in the previous framework)
    
    # We stay in the "eda" stage until the user explicitly says "move to cleaning"
    return {
        "eda_output_to_user": response.content,
        "messages": [response]
    }

g = StateGraph(InteractiveDataScienceState)
g.add_node('llm',interactive_eda_node)
g.add_node('tools',ToolNode([run_python_code,load_and_profile_data,generate_eda_visualizations]))
g.add_edge(START,'llm')
g.add_conditional_edges("llm", tools_condition)
g.add_edge("tools", "llm")
#memory = InMemorySaver()                #Uncomment if this file is run
#graph = g.compile(checkpointer=memory)
graph = g.compile()

if __name__=='__main__':
    initial_input = {"messages": HumanMessage(content="Analyze the features and provide information"),
                 'dataset_path':'datasets/ai_student.csv'}

    thread = {"configurable": {"thread_id": "1"}}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

    sandbox.shutdown()