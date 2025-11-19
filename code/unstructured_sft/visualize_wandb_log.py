import ast
import matplotlib.pyplot as plt
import os

log_path = "/localdisk1/PARK/park_vlm_finetuning/code/unstructured_sft/wandb/run-20251113_213917-bzfkgwl1/files/output.log"  # change this to your log file

train_epochs, train_losses = [], []
eval_epochs, eval_losses = [], []

with open(log_path, "r") as f:
    for line in f:
        line = line.strip()
        # Skip non dict lines
        if not line.startswith("{") or not line.endswith("}"):
            continue

        try:
            record = ast.literal_eval(line)
        except Exception:
            continue  # skip malformed lines

        # Eval (test) loss
        if "eval_loss" in record:
            eval_losses.append(record["eval_loss"])
            eval_epochs.append(record.get("epoch", None))

        # Training loss
        elif "loss" in record:
            train_losses.append(record["loss"])
            train_epochs.append(record.get("epoch", None))

# Quick sanity check
print(f"Training points: {len(train_losses)}, Eval points: {len(eval_losses)}")

plt.figure(figsize=(8, 5))

if train_losses:
    plt.plot(train_epochs, train_losses, label="Train loss")
if eval_losses:
    plt.plot(eval_epochs, eval_losses, marker="o", linestyle="--", label="Eval loss")

os.makedirs("plots", exist_ok=True)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Eval loss from log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/loss_plot.png")
