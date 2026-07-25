import os
import streamlit as st
from langgraph.types import Command
from main_graph import build_main_graph, MainState

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ML Pipeline", page_icon="🤖", layout="wide")
st.title("🤖 Automated ML Pipeline")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
# Persists across Streamlit reruns within the same browser session
if "graph"         not in st.session_state:
    st.session_state.graph         = build_main_graph()
if "thread_id"     not in st.session_state:
    st.session_state.thread_id     = "session-1"
if "chat_history"  not in st.session_state:
    st.session_state.chat_history  = []   # list of {"role": "user"|"assistant", "content": str}
if "graph_started" not in st.session_state:
    st.session_state.graph_started = False
if "graph_done"    not in st.session_state:
    st.session_state.graph_done    = False
if "interrupted"   not in st.session_state:
    st.session_state.interrupted   = False   # True when graph is paused at interrupt()
if "final_state"   not in st.session_state:
    st.session_state.final_state   = None
if "eda_messages"  not in st.session_state:
    st.session_state.eda_messages  = []
if "fe_messages"   not in st.session_state:
    st.session_state.fe_messages   = []
if "plot_paths"    not in st.session_state:
    st.session_state.plot_paths    = []


def get_config():
    return {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": 50,
    }


def add_message(role: str, content):
    st.session_state.chat_history.append({"role": role, "content": content})


def run_until_interrupt(invoke_input):
    """
    Calls graph.invoke() and handles the result.
    If the graph hits an interrupt(), stores the interrupt message and sets
    interrupted=True so the UI can collect user input.
    If the graph finishes, sets graph_done=True.
    """
    graph = st.session_state.graph
    config = get_config()

    with st.spinner("Running pipeline..."):
        state = graph.invoke(invoke_input, config=config)
    
    # Capture message threads as soon as they appear in state
    if state.get("eda_mess"):
        st.session_state.eda_messages = state["eda_mess"]
    if state.get("fe_mess"):
        st.session_state.fe_messages  = state["fe_mess"]
    if state.get("plot_paths"):
        st.session_state.plot_paths   = state["plot_paths"]

    interrupts = state.get("__interrupt__")

    if interrupts:
        # Graph paused — surface the interrupt message as an assistant message
        interrupt_msg = interrupts[0].value
        add_message("assistant", interrupt_msg)
        st.session_state.interrupted = True
    else:
        # Graph finished
        st.session_state.interrupted  = False
        st.session_state.graph_done   = True
        st.session_state.final_state  = state
        add_message("assistant", _format_results(state))


def _format_results(state: dict) -> str:
    if not state.get("model_results"):
        return "Pipeline complete."

    result  = state["model_results"]
    winner  = result["winner"]
    winning = result["results"][winner]

    lines = [
        "✅ **Modelling complete!**\n",
        f"**Winning branch:** {winner} — `{winning['algorithm']}`",
        f"**CV metric:** `{result['cv_metric']}`",
        f"**Optuna CV score:** `{winning['optuna_score']}`",
        f"**Best params:** `{winning['best_params']}`",
        f"**Test metrics:** `{winning['metrics']}`",
        "\n**All branches:**",
    ]
    for b, r in result["results"].items():
        marker = " ⬅ winner" if b == winner else ""
        lines.append(f"- `[{b}]` {r['algorithm']} | {r['metrics']}{marker}")

    return "\n".join(lines)

def render_message_thread(messages, title):
    if messages:
        with st.expander(title, expanded=False):
            for msg in messages:
                role = msg["role"]
                name = msg["name"]
                content = msg["content"]

                icon = "🤖" if "AI" in role else "👤"
                label = f"{icon} **{role}** ({name})" if name else f"{icon} **{role}**"

                st.markdown(label)

                if isinstance(content, str):
                    st.markdown(content)

                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                st.markdown(block.get("text", ""))
                            else:
                                st.write(block)
                        else:
                            st.write(block)

                else:
                    st.write(content)

                st.divider()


# ---------------------------------------------------------------------------
# Sidebar — dataset upload & pipeline start
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Dataset")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded and not st.session_state.graph_started:
        os.makedirs("datasets", exist_ok=True)
        dataset_path = f"datasets/{uploaded.name}"
        with open(dataset_path, "wb") as f:
            f.write(uploaded.read())

        if st.button("🚀 Start Pipeline"):
            st.session_state.graph_started = True
            add_message("assistant", f"Dataset `{uploaded.name}` loaded. Starting EDA...")

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
            run_until_interrupt(initial_state)
            st.rerun()

    if st.session_state.graph_done:
        st.success("Pipeline complete!")

    if st.button("🔄 Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------
# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        content = msg["content"]

        # Normal text message
        if isinstance(content, str):
            st.markdown(content)

        # LLM content blocks
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        st.markdown(block.get("text", ""))
                    else:
                        st.write(block)
                else:
                    st.write(block)

        # Structured interrupt message
        elif isinstance(content, dict):
            if "question" in content and "proposed_steps" in content:
                st.info(content["question"])

                with st.expander("📋 Proposed Cleaning Plan", expanded=True):
                    st.markdown(content["proposed_steps"])

            else:
                st.json(content)

        # Fallback
        else:
            st.write(content)

if st.session_state.plot_paths:
    #st.divider()
    with st.expander("📈 EDA Plots", expanded=False):
        #st.subheader("📈 EDA Plots")
        # Show plots in a responsive grid (2 columns)
        plots = [p for p in st.session_state.plot_paths if os.path.isfile(p)]
        for i in range(0, len(plots), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(plots):
                    path = plots[i + j]
                    with col:
                        caption = (
                            os.path.splitext(os.path.basename(path))[0]
                            .replace("_", " ")
                            .title()
                        )
                        st.image(
                            path,
                            caption=caption,
                            use_container_width=True,
                        )

# Results panel
if st.session_state.graph_done and st.session_state.final_state:
    state   = st.session_state.final_state
    result  = state.get("model_results", {})
    winner  = result.get("winner")

    if winner:
        st.divider()
        st.subheader("📊 Model Results")

        col1, col2, col3 = st.columns(3)
        winning = result["results"][winner]
        metrics = winning["metrics"]

        with col1:
            st.metric("Algorithm",    winning["algorithm"])
        with col2:
            st.metric("CV Metric",    result["cv_metric"])
        with col3:
            st.metric("Optuna Score", winning["optuna_score"])

        st.subheader("All Branches")
        branch_data = []
        for b, r in result["results"].items():
            row = {"branch": b, "algorithm": r["algorithm"],
                   "optuna_cv": r["optuna_score"], "winner": b == winner}
            row.update(r["metrics"])
            branch_data.append(row)
        st.dataframe(branch_data, use_container_width=True)

        # ── Model downloads ──────────────────────────────────────────────
        st.subheader("⬇️ Download Models")
        from main_graph import _MAIN_MODELS
        import pickle
 
        all_models = _MAIN_MODELS.get("all_models", {})
        if all_models:
            algo_names = {
                b: result["results"][b]["algorithm"]
                for b in all_models
            }
            cols = st.columns(len(all_models))
            for col, (branch, model) in zip(cols, all_models.items()):
                algo  = algo_names.get(branch, branch)
                label = f"{'🏆 ' if branch == winner else ''}{algo} (branch {branch})"
                buf   = pickle.dumps(model)
                col.download_button(
                    label     = label,
                    data      = buf,
                    file_name = f"model_{branch}_{algo}.pkl",
                    mime      = "application/octet-stream",
                    key       = f"download_{branch}",
                )

# EDA and FE message threads (shown in expanders once available)
eda_msgs = st.session_state.eda_messages
fe_msgs  = st.session_state.fe_messages

render_message_thread(eda_msgs, "🔍 EDA Agent Message Thread")
render_message_thread(fe_msgs, "🔧 Feature Engineering Agent Message Thread")


# Chat input — only active when graph is running and waiting at an interrupt
if st.session_state.graph_started and not st.session_state.graph_done:
    if st.session_state.interrupted:
        user_msg = st.chat_input("Your response...")
        if user_msg:
            add_message("user", user_msg)
            # Resume the graph from the interrupt with the user's response
            run_until_interrupt(Command(resume=user_msg))
            st.rerun()
    else:
        st.chat_input("Waiting for pipeline...", disabled=True)

elif not st.session_state.graph_started:
    st.chat_input("Upload a dataset and click Start to begin.", disabled=True)