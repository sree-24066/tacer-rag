"""
Confidence calibration for the TACER-RAG hallucination detector.

Implements:
1. Temperature Scaling (Guo et al., ICML 2017)
2. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
3. Multi-class Brier Score
4. Reliability Diagrams (Confidence vs. Accuracy calibration plots)
5. Validation-based confidence threshold selection (never tuned on test set)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from tacer_rag.utils import get_logger

logger = get_logger("calibration")


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling for probability calibration.
    Calibrates logits: p_scaled = softmax(logits / T).
    Temperature T > 0 is learned by minimizing negative log likelihood on validation set.
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(initial_temperature)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature.clamp(min=1e-4)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Fit temperature T on validation logits and labels using L-BFGS.
        Only valid labels (not -100) are used.
        """
        # Filter out ignore_index (-100)
        mask = val_labels != -100
        filtered_logits = val_logits[mask]
        filtered_labels = val_labels[mask]

        if len(filtered_labels) == 0:
            logger.warning("No valid labels provided to TemperatureScaler.fit")
            return 1.0

        device = filtered_logits.device
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(filtered_logits), filtered_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        final_temp = float(self.temperature.item())
        logger.info("Optimal temperature learned: T = %.4f", final_temp)
        return final_temp

    def scale_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probabilities after temperature scaling."""
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            return torch.softmax(scaled_logits, dim=-1)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probs: Array of predicted probabilities shape (N, num_classes) or (N,)
        labels: Array of integer true labels shape (N,)
        n_bins: Number of equal-width confidence bins

    Returns:
        dict with ece, mce, bin_accuracies, bin_confidences, bin_counts
    """
    if probs.ndim == 2:
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
    else:
        confidences = probs
        predictions = (confidences >= 0.5).astype(int)

    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        count = int(np.sum(in_bin))
        bin_counts.append(count)

        if count > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            diff = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            bin_accs.append(round(accuracy_in_bin, 4))
            bin_confs.append(round(avg_confidence_in_bin, 4))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bin_lower + bin_upper) / 2))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 9) -> float:
    """
    Compute multi-class Brier score: (1/N) * sum_{i,k} (p_{i,k} - y_{i,k})^2.
    Lower is better (0.0 is perfect).
    """
    N = len(labels)
    if N == 0:
        return 0.0

    one_hot = np.zeros((N, num_classes), dtype=float)
    for i, l in enumerate(labels):
        if 0 <= l < num_classes:
            one_hot[i, l] = 1.0

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
    return round(float(brier), 4)


def plot_reliability_diagram(
    uncal_ece_dict: Dict[str, Any],
    cal_ece_dict: Dict[str, Any],
    save_path: str | Path,
    title_suffix: str = "",
) -> None:
    """
    Plot side-by-side reliability diagrams before and after temperature scaling.
    Uses dark-theme compatible styling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor="#161b22")

    for ax, data, label in zip(
        axes,
        [uncal_ece_dict, cal_ece_dict],
        ["Uncalibrated", "Calibrated (Temperature Scaled)"]
    ):
        ax.set_facecolor("#0d1117")
        n_bins = data["n_bins"]
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = 1.0 / n_bins

        accs = data["bin_accuracies"]
        confs = data["bin_confidences"]
        ece = data["ece"]

        # Bar plot of accuracy vs confidence with vibrant cyan/blue theme
        bar_color = "#388bfd" if "Uncalibrated" in label else "#2ea043"
        ax.bar(bin_centers, accs, width=width * 0.88, alpha=0.85, color=bar_color,
               edgecolor="#30363d", label="Accuracy in Bin")

        # Ideal calibration line
        ax.plot([0, 1], [0, 1], "--", color="#f85149", linewidth=2.0, label="Perfect Calibration")

        # Gap indicator
        for c, a in zip(confs, accs):
            if c > 0 and a > 0:
                ax.plot([c, c], [a, c], color="#d29922", linestyle=":", linewidth=1.5)

        ax.set_title(f"{label}\nECE = {ece:.4f}", fontsize=12, color="#f0f6fc", fontweight="bold", pad=10)
        ax.set_xlabel("Confidence Score", fontsize=10, color="#c9d1d9")
        ax.set_ylabel("Empirical Accuracy", fontsize=10, color="#c9d1d9")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, linestyle="--", alpha=0.2, color="#8b949e")
        legend = ax.legend(loc="upper left", framealpha=0.6, facecolor="#161b22", edgecolor="#30363d")
        for text in legend.get_texts():
            text.set_color("#c9d1d9")

    full_title = f"Reliability Diagrams {title_suffix}".strip()
    plt.suptitle(full_title, fontsize=14, fontweight="bold", color="#f0f6fc", y=0.98)
    plt.tight_layout()

    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info("Saved reliability diagram to %s", out_file)


def select_confidence_thresholds(
    calibrated_probs: np.ndarray,
    labels: np.ndarray,
    hallucination_label_ids: list[int],
) -> Dict[str, float]:
    """
    Select detection confidence operating thresholds on validation data:
    1. high_confidence_threshold: threshold where precision >= 0.80 or 80th percentile
    2. low_confidence_threshold: threshold below which detector should abstain
    """
    # Max probability across hallucination classes
    hal_probs = np.max(calibrated_probs[:, hallucination_label_ids], axis=-1)
    is_gold_hal = np.isin(labels, hallucination_label_ids).astype(int)

    # Sort thresholds to find empirical precision operating points
    thresholds = np.linspace(0.1, 0.95, 86)
    best_high = 0.75
    best_low = 0.35

    for t in thresholds:
        pred_pos = hal_probs >= t
        if np.sum(pred_pos) > 0:
            prec = np.sum(is_gold_hal[pred_pos]) / np.sum(pred_pos)
            if prec >= 0.80 and t < best_high:
                best_high = round(float(t), 2)
            if prec >= 0.50 and t < best_low:
                best_low = round(float(t), 2)

    return {
        "high_confidence_threshold": best_high,
        "low_confidence_threshold": best_low,
        "abstain_below_threshold": round(best_low * 0.7, 2),
    }
