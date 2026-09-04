"""Génère les figures du rapport à partir des run_summary.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).parent
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

CLASS_COLORS = {
    "LCR": "#4C72B0",
    "matière grise": "#8172B2",
    "matière blanche": "#DD8452",
}


def load(name):
    with open(REPORT_DIR / name) as f:
        return json.load(f)


def plot_learning_curves(summary, title, out_name):
    history = summary["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    mean_dice = [h["mean_dice_fg"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_loss, label="train_loss", color="#4C72B0")
    ax1.plot(epochs, val_loss, label="val_loss", color="#C44E52")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (CrossEntropy)")
    ax1.set_title("Perte")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, mean_dice, label="Dice moyen (fg)", color="#55A868", linewidth=2)
    for cls, color in CLASS_COLORS.items():
        vals = [h["dice_per_class"][cls] for h in history]
        ax2.plot(epochs, vals, label=cls, color=color, linestyle="--", alpha=0.8)
    best_epoch = max(history, key=lambda h: h["mean_dice_fg"])["epoch"]
    ax2.axvline(best_epoch, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_ylim(0, 1)
    ax2.set_title("Score Dice par classe")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"saved {out_name}")


baseline = load("baseline_bc16_summary.json")
plot_learning_curves(
    baseline,
    "CompactUNet (base_channels=16) — 20 epochs — 99 503 paramètres",
    "baseline_20ep_learning_curves.png",
)
