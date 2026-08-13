"""
Model Interpretability: GradCAM, GradCAM++, and Layer Comparison
Visualises what ResNet50 learns at different depths using Captum.
Results are logged to Weights & Biases.
"""

import matplotlib
matplotlib.use('agg')
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
import logging
import wandb

from captum.attr import LayerGradCam, LayerAttribution
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# ── paths ──────────────────────────────────────────────────────────────────
_src_dir  = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_src_dir)
sys.path.insert(0, _src_dir)

from load_date import load_config
from train import create_model

config     = load_config(os.path.join(_root_dir, "configs", "params.yaml"))
output_dir = os.path.join(_root_dir, "interpretation_results")
os.makedirs(output_dir, exist_ok=True)

# ── W&B init ───────────────────────────────────────────────────────────────
wandb.init(
    project="flower-classification-interpretability",
    name=f"gradcam_layers_lr_{config['training']['optimizer']['lr']}",
    config={
        "learning_rate": config["training"]["optimizer"],
        "dataset":       "Flowers-102",
        "epochs":        config["training"]["epochs"],
        "experiments":   ["GradCAM per layer", "GradCAM vs GradCAM++"],
    }
)

logger = logging.getLogger(__name__)


def load_model_and_data(config, n_samples: int = 20):
    """Load trained model and a small test subset."""
    transform = transforms.Compose([
        transforms.Resize((225, 225)),
        transforms.ToTensor(),
    ])
    data_dir = os.path.join(config["data"]["local_dir"], "jpg")
    dataset  = datasets.Flowers102(
        root=data_dir, split="test", transform=transform, download=True
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = create_model()
    model.load_state_dict(
        torch.load(
            os.path.join(config["artifacts"]["output_dir"], "best_model.pth"),
            map_location=device,
        )
    )
    model.to(device).eval()
    return model, loader, device


def to_heatmap(attr: np.ndarray) -> np.ndarray:
    """Normalise attribution to a jet heatmap (H×W×3, uint8)."""
    a = attr.squeeze()
    if a.ndim == 3:
        a = a.mean(axis=0)          # collapse channel dim if present
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return np.uint8(cm.jet(a)[..., :3] * 255)


def gradcam_pp(model, layer, images, labels):
    """Simple GradCAM++ via weighted sum of positive gradient channels."""
    images.requires_grad_(True)
    activations, gradients = {}, {}

    def fwd_hook(m, i, o):
        activations["v"] = o.detach()

    def bwd_hook(m, gi, go):
        gradients["v"] = go[0].detach()

    h_fwd = layer.register_forward_hook(fwd_hook)
    h_bwd = layer.register_full_backward_hook(bwd_hook)

    output = model(images)
    model.zero_grad()
    output[0, labels[0].item()].backward()

    h_fwd.remove()
    h_bwd.remove()

    grads = gradients["v"]                          # (1, C, H, W)
    acts  = activations["v"]                        # (1, C, H, W)

    # GradCAM++ weights
    grads_sq  = grads ** 2
    grads_cub = grads ** 3
    denom     = 2 * grads_sq + acts * grads_cub.sum(dim=(2, 3), keepdim=True)
    alpha     = grads_sq / (denom + 1e-8)
    weights   = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)

    cam = (weights * acts).sum(dim=1, keepdim=True)      # (1, 1, H, W)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam, size=(225, 225), mode="bilinear", align_corners=False
    )
    return cam[0, 0].cpu().detach().numpy()


# ── Experiment 1: GradCAM across layers ────────────────────────────────────
def experiment_layer_comparison(model, loader, device, n_samples=10):
    """Visualise GradCAM at layer1, layer2, layer3, layer4."""
    layers = {
        "layer1": model.layer1[-1],
        "layer2": model.layer2[-1],
        "layer3": model.layer3[-1],
        "layer4": model.layer4[-1],
    }
    wandb_images = []

    for idx, (images, labels) in enumerate(loader):
        if idx >= n_samples:
            break
        images, labels = images.to(device), labels.to(device)

        fig, axes = plt.subplots(1, len(layers) + 1, figsize=(5 * (len(layers) + 1), 5))
        axes[0].imshow(images[0].cpu().permute(1, 2, 0))
        axes[0].set_title("Original")
        axes[0].axis("off")

        for ax, (name, layer) in zip(axes[1:], layers.items()):
            gc    = LayerGradCam(model, layer)
            attr  = gc.attribute(images, target=labels)
            inter = LayerAttribution.interpolate(attr, (225, 225))
            heat  = to_heatmap(inter[0].cpu().detach().numpy())
            ax.imshow(heat)
            ax.set_title(name)
            ax.axis("off")

        plt.suptitle(f"Sample {idx+1} — GradCAM by layer", fontsize=14)
        path = os.path.join(output_dir, f"layers_{idx+1}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        wandb_images.append(
            wandb.Image(path, caption=f"Sample {idx+1} | label {labels[0].item()}")
        )
        os.remove(path)

    wandb.log({"Layer comparison (GradCAM)": wandb_images})
    print(f"Layer comparison: logged {len(wandb_images)} images to W&B")


# ── Experiment 2: GradCAM vs GradCAM++ ─────────────────────────────────────
def experiment_gradcam_vs_pp(model, loader, device, n_samples=10):
    """Side-by-side: original | GradCAM | GradCAM++"""
    layer      = model.layer4[-1]
    gc         = LayerGradCam(model, layer)
    wandb_images = []

    for idx, (images, labels) in enumerate(loader):
        if idx >= n_samples:
            break
        images, labels = images.to(device), labels.to(device)

        # GradCAM
        attr  = gc.attribute(images, target=labels)
        inter = LayerAttribution.interpolate(attr, (225, 225))
        heat_gc = to_heatmap(inter[0].cpu().detach().numpy())

        # GradCAM++
        heat_pp = to_heatmap(gradcam_pp(model, layer, images.clone(), labels))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(images[0].cpu().permute(1, 2, 0))
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(heat_gc)
        axes[1].set_title("GradCAM")
        axes[1].axis("off")
        axes[2].imshow(heat_pp)
        axes[2].set_title("GradCAM++")
        axes[2].axis("off")

        plt.suptitle(f"Sample {idx+1} — GradCAM vs GradCAM++", fontsize=14)
        path = os.path.join(output_dir, f"gcpp_{idx+1}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        wandb_images.append(
            wandb.Image(path, caption=f"Sample {idx+1} | label {labels[0].item()}")
        )
        os.remove(path)

    wandb.log({"GradCAM vs GradCAM++": wandb_images})
    print(f"GradCAM vs GradCAM++: logged {len(wandb_images)} images to W&B")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    model, loader, device = load_model_and_data(config, n_samples=20)

    print("Running Experiment 1: Layer comparison...")
    experiment_layer_comparison(model, loader, device, n_samples=10)

    print("Running Experiment 2: GradCAM vs GradCAM++...")
    experiment_gradcam_vs_pp(model, loader, device, n_samples=10)

    wandb.finish()
    print("Done. Results logged to W&B.")


if __name__ == "__main__":
    main()
