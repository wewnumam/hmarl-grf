
import solara
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mesa_model import FootballModel
import threading
import time

# Global Model Instance
model = FootballModel(training_mode=True)
training_thread = None
is_training = solara.reactive(False)
episode_count = solara.reactive(0)
reward_history = solara.reactive([])

def train_job():
    global model
    while is_training.value:
        model.train_batch(episodes=1)
        # Update reactive variables
        reward_history.set(list(model.reward_history))
        episode_count.set(len(model.reward_history))
        time.sleep(0.1) # Yield a bit

def toggle_training():
    global training_thread
    if is_training.value:
        is_training.set(False)
        if training_thread:
            training_thread.join()
    else:
        is_training.set(True)
        training_thread = threading.Thread(target=train_job)
        training_thread.start()

@solara.component
def Page():
    with solara.Column(style={"padding": "20px", "max-width": "1000px", "margin": "0 auto"}):
        solara.Title("HMARL Football Trainer (PPO)")
        
        with solara.Card("Control Panel"):
            solara.Text(f"Total Episodes: {episode_count.value}")
            solara.Button(
                label="Stop Training" if is_training.value else "Start Training",
                on_click=toggle_training,
                color="primary" if not is_training.value else "green"
            )
            
        with solara.GridFixed(columns=2):
            with solara.Card("Reward History"):
                if len(reward_history.value) > 0:
                    df = pd.DataFrame({
                        "Episode": range(len(reward_history.value)),
                        "Reward": reward_history.value
                    })
                    # Rolling average
                    df["Average Reward (10)"] = df["Reward"].rolling(10).mean()
                    
                    fig = px.line(df, x="Episode", y=["Reward", "Average Reward (10)"], 
                                  title="Training Progress", 
                                  labels={"value": "Score", "variable": "Metric"})
                    fig.update_layout(template="plotly_dark")
                    solara.FigurePlotly(fig)
                else:
                    solara.Text("No training data yet.")
            
            with solara.Card("Agent Hierarchy Status"):
                solara.Markdown("### Policy Architecture")
                solara.Markdown("- **High Level**: Strategy Selection (Interval: 20 steps)")
                solara.Markdown("- **Mid Level**: Sub-goal Targeting (Interval: 5 steps)")
                solara.Markdown("- **Low Level**: Action Execution (PPO, Every step)")
                
                solara.Markdown("### Current Stats")
                if model.controller.low_level.values:
                    last_val = model.controller.low_level.values[-1]
                    solara.Text(f"Last Value Estimate: {last_val:.4f}")
                
                # Visualize Action Probabilities
                if hasattr(model.controller.low_level, 'last_probs') and model.controller.low_level.last_probs is not None:
                    probs = model.controller.low_level.last_probs
                    # GRF has 19 actions
                    action_labels = [str(i) for i in range(len(probs))]
                    fig_probs = px.bar(x=action_labels, y=probs, labels={'x': 'Action', 'y': 'Probability'}, title="Low Level Policy Dist")
                    fig_probs.update_layout(template="plotly_dark", height=250)
                    solara.FigurePlotly(fig_probs)
                
                with solara.Row():
                    solara.VBox([
                        solara.Text("High Level Command:"),
                        solara.Text(str(model.controller.current_high_action), style="font-size: 24px; font-weight: bold")
                    ])
                    solara.VBox([
                        solara.Text("Mid Level Command:"),
                        solara.Text(str(model.controller.current_mid_action), style="font-size: 24px; font-weight: bold")
                    ])

        with solara.Card("Visualization Info"):
            solara.Markdown("Environment running in `simple115` vector mode for PPO efficiency. Spatial visualization is disabled to favor training speed.")
