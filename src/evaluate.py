"""
Model evaluation on the test set.
Saves metrics to artifacts/metrics.json.
"""
import os
import sys
import json
import torch
from torch.utils.data import DataLoader

# ── paths ──────────────────────────────────────────────────────────────────
_src_dir  = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_src_dir)
sys.path.insert(0, _src_dir)

from load_date import load_config, load_and_split_data
from train import create_model

config = load_config(os.path.join(_root_dir, "configs", "params.yaml"))


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return {"test_accuracy": round(accuracy / 100.0, 6)}


def main():
    config   = load_config(os.path.join(_root_dir, "configs", "params.yaml"))
    data_dir = config["data"]["local_dir"]

    _, _, test_dataset = load_and_split_data(data_dir)
    test_loader = DataLoader(
        test_dataset, batch_size=config["training"]["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = create_model()
    model.load_state_dict(
        torch.load(
            os.path.join(config["artifacts"]["output_dir"], "best_model.pth"),
            map_location=device,
        )
    )
    model.to(device)

    metrics     = evaluate(model, test_loader, device)
    output_dir  = config["artifacts"]["output_dir"]
    metrics_file = os.path.join(output_dir, "metrics.json")

    os.makedirs(output_dir, exist_ok=True)

    # Append to existing metrics if file exists
    if os.path.isfile(metrics_file):
        with open(metrics_file, "r") as f:
            existing = json.load(f)
        existing.update(metrics)
        metrics = existing

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
