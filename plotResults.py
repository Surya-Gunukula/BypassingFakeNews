import os 
import json
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def parseMetrics(path, run_name):
    with open(path) as f:
        metrics = json.load(f)
        loss = []
        acc = []
        for log in metrics["log_history"]:
            if "loss" in log and "epoch" in log:
                loss.append({
                    "epoch": log["epoch"],
                    "loss" : log["loss"],
                    "run": run_name
                })
            if "eval_accuracy" in log and "epoch" in log:
                acc.append({
                    "epoch": log["epoch"],
                    "accuracy" : log["eval_accuracy"],
                    "run": run_name
                })
    
    return pd.DataFrame(loss), pd.DataFrame(acc)


loss_r16, acc_r16 = parseMetrics("./resultsLoRA-r16/checkpoint-4500/trainer_state.json", "LoRA-r16")
loss_r4, acc_r4 = parseMetrics("./resultsLoRA-r4/checkpoint-4500/trainer_state.json", "LoRA-r4")
loss_complete, acc_complete = parseMetrics("./resultsCompleteFineTuning/checkpoint-4500/trainer_state.json", "Complete Fine Tuning")
loss_r8, acc_r8 = parseMetrics("./resultsLoRA/checkpoint-4500/trainer_state.json", "LoRA-r8")
loss_head, acc_head = parseMetrics("./resultsClassificationHead/checkpoint-4500/trainer_state.json", "Classification Head")

df_loss_all = pd.concat([loss_r16, loss_r8, loss_r4, loss_complete, loss_head])
df_acc_all = pd.concat([acc_r16, acc_r8, acc_r4, acc_complete, acc_head])

plt.figure(figsize=(10, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

runs = df_loss_all['run'].unique()
colors = plt.cm.tab10.colors
loss_marker = "o"     # Circle for loss
acc_marker = "s"      # Square for accuracy

for idx, run_name in enumerate(runs):
    color = colors[idx % len(colors)]

    # Plot loss on left axis
    loss_group = df_loss_all[df_loss_all['run'] == run_name]
    ax1.plot(loss_group["epoch"], loss_group["loss"], label=run_name, color=color, marker=loss_marker)

    # Plot accuracy on right axis
    acc_group = df_acc_all[df_acc_all['run'] == run_name]
    ax2.plot(acc_group["epoch"], acc_group["accuracy"], color=color, marker=acc_marker)

# Axis labels and coloring
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="tab:red")
ax2.set_ylabel("Accuracy", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:blue")

# Legend
ax1.legend(
    title="Run",
    loc="center right",           # Inside the plot
    bbox_to_anchor=(1, 0.7),   # Adjust horizontal position (closer to center)
    frameon=True,
    shadow=False,
    fontsize="medium"
)

plt.title("Training Loss & Eval Accuracy Across Runs")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()







