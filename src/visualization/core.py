import json
import plotly.graph_objects as go
import os

from src.utils.paths import RESULTS_PATH


class Visualizer:

    def __init__(self):
        self.logs_file_path = RESULTS_PATH / "training_monitoring" / "lora.json"
        self.logs_plots_path = RESULTS_PATH / "training_monitoring" / "plots"
        os.makedirs(self.logs_plots_path, exist_ok=True)
        os.makedirs(self.logs_file_path.parent, exist_ok=True)

    def execute(self):
        self.load_log()
        self.plot()

    def load_log(self):
        with open(self.logs_file_path, "r") as f:
            self.data = json.load(f)["data"]

    def plot(self):

        epochs = [d["epoch"] for d in self.data]
        loss = [d["loss"] for d in self.data]
        grad_norm = [d["grad_norm"] for d in self.data]
        learning_rate = [d["learning_rate"] for d in self.data]

        # 1. Loss vs Epoch
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=epochs,
            y=loss,
            mode="lines+markers"
        ))
        fig_loss.update_layout(
            title="Loss vs Epoch",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )
        fig_loss.write_image(self.logs_plots_path / "loss_vs_epoch.png")

        # 2. Grad Norm vs Epoch
        fig_grad = go.Figure()
        fig_grad.add_trace(go.Scatter(
            x=epochs,
            y=grad_norm,
            mode="lines+markers"
        ))
        fig_grad.update_layout(
            title="Grad Norm vs Epoch",
            xaxis_title="Epoch",
            yaxis_title="Grad Norm"
        )
        fig_grad.write_image(self.logs_plots_path / "grad_norm_vs_epoch.png")

        # 3. Learning Rate vs Epoch
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(
            x=epochs,
            y=learning_rate,
            mode="lines+markers"
        ))
        fig_lr.update_layout(
            title="Learning Rate vs Epoch",
            xaxis_title="Epoch",
            yaxis_title="Learning Rate"
        )
        fig_lr.write_image(self.logs_plots_path / "learning_rate_vs_epoch.png")

        print("Saved:")
        print("- loss_vs_epoch.html")
        print("- grad_norm_vs_epoch.html")
        print("- learning_rate_vs_epoch.html")
