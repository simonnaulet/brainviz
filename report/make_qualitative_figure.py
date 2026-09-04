"""Figure qualitative : T1 / vérité terrain / prédiction, sur une tranche de validation."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

from brainviz.data.dataset import CLASS_NAMES, BrainSliceDataset
from brainviz.models import CompactUNet
from brainviz.visualize import find_nonempty_indices, _to_numpy_2d

REPORT_DIR = Path(__file__).parent

device = torch.device("cpu")

ds = BrainSliceDataset("dataset/train", axis=2, modality="T1T2", scaling="padding", min_foreground_ratio=0.0)
val_mask = ds.subject_ids == "subject-9"
val_indices = [i for i in range(len(ds)) if val_mask[i]]

candidates = set(find_nonempty_indices(ds, min_ratio=0.5)) & set(val_indices)
idx = sorted(candidates)[len(candidates) // 2]

model = CompactUNet(in_channels=ds.num_channels, num_classes=ds.num_classes, base_channels=16, depth=3)
model.load_state_dict(torch.load(REPORT_DIR / "baseline_bc16_best_model.pt", map_location=device))
model.eval()

x, y = ds[idx]
with torch.no_grad():
    logits = model(x.unsqueeze(0))
    pred = logits.argmax(dim=1).squeeze(0)

t1 = _to_numpy_2d(x[0])
label = _to_numpy_2d(y)
pred_np = _to_numpy_2d(pred)

n_classes = len(CLASS_NAMES)
cmap = plt.get_cmap("viridis", n_classes)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
axes[0].imshow(t1, cmap="gray")
axes[0].set_title("T1 (entrée)")
axes[0].axis("off")

axes[1].imshow(label, cmap=cmap, interpolation="nearest", vmin=0, vmax=n_classes - 1)
axes[1].set_title("Vérité terrain")
axes[1].axis("off")

axes[2].imshow(pred_np, cmap=cmap, interpolation="nearest", vmin=0, vmax=n_classes - 1)
axes[2].set_title("Prédiction du modèle")
axes[2].axis("off")

handles = [Patch(color=cmap(i), label=name) for i, name in enumerate(CLASS_NAMES)]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

dice_line = ", ".join(
    f"{name}={float(((pred_np == i) & (label == i)).sum() * 2 / max((pred_np == i).sum() + (label == i).sum(), 1)):.3f}"
    for i, name in enumerate(CLASS_NAMES) if i > 0
)
fig.suptitle(f"subject-9, tranche idx={idx} (validation, non vue à l'entraînement)\nDice sur cette tranche : {dice_line}")
fig.tight_layout(rect=[0, 0.05, 1, 0.92])
fig.savefig(REPORT_DIR / "figures" / "qualitative_prediction_subject9.png", dpi=150)
print("saved qualitative_prediction_subject9.png")
