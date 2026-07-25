from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from pprint import pprint
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver 
from typing import Dict, Any, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import interrupt, Command

from kernel_tool import sandbox


load_dotenv()

class InteractiveDataScienceState(MessagesState):
    # Data Paths
    dataset_path: str
    cleaned_data_path: str
    
    # Flow Control Flags
    user_input: str
    
    # Internal agent history
    eda_output_to_user: str
    cleaning_plan: str

@tool
def run_python_code(code: str) -> str:
    """Executes Python code in an isolated local Jupyter kernel sandbox and returns the stdout or errors. 
    Use this to inspect data, clean datasets, train models, and print evaluation metrics."""
    return sandbox.execute_code(code)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
llm_with_tools = llm.bind_tools([run_python_code])

llm_planner = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

def cleaning_plan_node(state: InteractiveDataScienceState) -> Dict[str, Any]:
    print("\n[Node: Cleaning Agent] Processing your dataset...")

    
    sys_msg = SystemMessage(content=(f"""You are a Data Cleaning Expert.

    You have been provided with Exploratory Data Analysis (EDA) results for a dataset.

    Your task is to carefully analyze the EDA findings and create a comprehensive data cleaning plan before any modeling is performed.

    EDA Results:
    {state['eda_output_to_user']}

    Analyze the EDA results and identify:

    1. Missing value issues
    - Columns with missing values
    - Severity of missingness
    - Recommended treatment (drop, mean/median/mode imputation, forward fill, backward fill, interpolation, etc.)
    - Justification for each recommendation

    2. Duplicate records
    - Presence of duplicate rows
    - Recommended action

    3. Data type issues
    - Columns with potentially incorrect data types
    - Recommended conversions

    4. Outliers
    - Numerical columns containing significant outliers
    - Whether to keep, cap, transform, or remove them
    - Justification based on the EDA findings

    5. Categorical data issues
    - High-cardinality columns
    - Rare categories
    - Inconsistent labels, casing, spelling, or formatting problems

    6. Feature quality issues
    - Constant or near-constant columns
    - Columns with extremely high missingness
    - Columns with little predictive value
    - Potential data leakage columns

    7. Distribution-related issues
    - Highly skewed numerical features
    - Recommended transformations if needed

    8. Correlation and redundancy
    - Highly correlated features
    - Redundant columns that may require removal

    9. Data consistency issues
    - Invalid values
    - Impossible values
    - Formatting inconsistencies
    - Potential anomalies

    10. Cleaning execution order
        - Provide the recommended sequence of cleaning steps

    Return your response as a structured cleaning plan.

    Only make recommendations that are supported by the provided EDA results. Do not invent issues that are not evident from the analysis.
  
    """))



    response = llm_planner.invoke([sys_msg] + state['messages'])
    
    return {
        "cleaning_plan": response.content,
        "messages": [response]
    }

def cleaning_code_executor(state: InteractiveDataScienceState) -> Dict[str, Any]:

    """Pause and ask the user to approve, revise, or no cleaning."""
    if not state.get("user_input"):
      user_review = interrupt({
        "question": "Here's the proposed cleaning plan. Approve, suggest changes, or no cleaning?",
        "proposed_steps": state["cleaning_plan"][0]['text'],
    })
    else:
      user_review = state["user_input"]
    
    
    sys_msg = SystemMessage(content=(f"""You are a Senior Data Cleaning Execution Agent.

    Your job is to review the proposed cleaning plan, consider the user's feedback, generate Python code to perform the required cleaning, and execute that code using the available `run_python_code` tool.

    Available Information:

    Dataset Path:
    {state['dataset_path']}

    Cleaning Plan:
    {state['cleaning_plan']}

    User Review / Feedback:
    {user_review}

    Instructions:

    1. Read the cleaning plan carefully.
    2. Read the user's feedback carefully.
    3. Determine how to proceed:

    A. If the user APPROVED the cleaning plan:
        - Execute the cleaning plan as specified.

    B. If the user requested MODIFICATIONS:
        - Follow the user's instructions.
        - Override the corresponding cleaning methods from the original plan.
        - Keep all other approved cleaning steps unchanged.

    C. If the user explicitly requested NOT to clean the dataset:
        - Do not execute any cleaning code.
        - Return a summary explaining that no cleaning was performed.

    4. The dataset is located at:
    state["dataset_path"]

    5. Generate Python code that:
    - Loads the dataset.
    - Applies the approved cleaning operations.
    - Saves the cleaned dataset to a new file as 'datasets/cleaned.csv'.
    - Produces useful execution output describing what was changed.

    6. Execute the generated code using the `run_python_code` tool.

    7. Before executing code:
    - Verify that the code aligns with the approved cleaning plan and user feedback.
    - Ensure all referenced columns exist before applying transformations.
    - Handle potential errors gracefully.

    8. Only perform safe data-cleaning operations such as:
    - Missing value treatment
    - Duplicate removal
    - Datatype conversions
    - Outlier treatment
    - Category standardization
    - Feature removal
    - Basic feature transformations

    9. Strict Safety Requirements:

    Never generate or execute code that:
    - Uses `eval()`
    - Uses `exec()`
    - Uses `compile()`
    - Uses `subprocess`
    - Uses `os.system`
    - Uses `shutil.rmtree`
    - Deletes files
    - Modifies files outside the dataset workflow
    - Accesses the network
    - Installs packages
    - Executes shell commands
    - Reads arbitrary system files
    - Performs actions unrelated to dataset cleaning

    10. Save the cleaned dataset as a new file.
        Never overwrite the original dataset.

    11. If code execution fails:
        - Analyze the error.
        - Generate a corrected version of the code.
        - Retry using the `run_python_code` tool.
        - Repeat until the task succeeds or a safe resolution is reached.

    AFTER RUNNING THE run_python_code TOOL AND GETTING 'code executed successfully' TOOL OUTPUT:
    - Confirm whether cleaning succeeded.
    - Summarize the performed actions.
    - Report the location of the cleaned dataset.

    Output Format:

    "cleaning_performed": true/false,
    "user_changes_applied": [...],
    "cleaning_actions": [...],
    "cleaned_dataset_path": "...",
    "execution_status": "success | failed",
    "summary": "Short summary of what was done."

    Always use the `run_python_code` tool to perform the actual cleaning work. Do not merely describe the cleaning steps. The final response should be based on the actual execution results produced by the tool.
  
    """))


    response = llm_with_tools.invoke([sys_msg]+state['messages'])
    print(response.content)
    
    return {
        "messages": [response],
        "user_input": "user_review",
        "cleaning_plan": response.content
    }

g = StateGraph(InteractiveDataScienceState)
g.add_node('clean_plan_llm',cleaning_plan_node)
g.add_node('clean_code_llm',cleaning_code_executor)
g.add_node('tools',ToolNode([run_python_code]))
g.add_edge(START,'clean_plan_llm')
g.add_edge('clean_plan_llm','clean_code_llm')
g.add_conditional_edges("clean_code_llm", tools_condition)
g.add_edge("tools", "clean_code_llm")
#memory = InMemorySaver()
#graph = g.compile(checkpointer=memory)
graph = g.compile()


