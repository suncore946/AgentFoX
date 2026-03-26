"""
Calibration Visualization Module - Simplified Version
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set font and plotting style
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
sns.set_style("whitegrid")


class CalibrationVisualizer:
    """
    Calibration Effect Visualizer - Simplified Version
    """
    
    def __init__(self, figsize=(16, 12), dpi=300, n_bins=15):
        """Initialize visualizer with basic parameters"""
        self.figsize = figsize
        self.dpi = dpi
        self.n_bins = n_bins
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
    def _calculate_reliability_data(
        self, true_labels: np.ndarray, pred_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate reliability diagram data"""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            if bin_lower == 0:
                in_bin = (pred_probs >= bin_lower) & (pred_probs <= bin_upper)
            else:
                in_bin = (pred_probs > bin_lower) & (pred_probs <= bin_upper)
            
            bin_count = in_bin.sum()
            
            if bin_count > 0:
                acc_in_bin = true_labels[in_bin].mean()
                avg_conf_in_bin = pred_probs[in_bin].mean()
                bin_accs.append(acc_in_bin)
                bin_confs.append(avg_conf_in_bin)
                bin_counts.append(bin_count)
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_counts.append(0)
        
        return bin_boundaries, np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)

    def plot_reliability_diagram(
        self, 
        original_data: pd.DataFrame, 
        calibration_result: Dict[str, Any], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Draw reliability diagram comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Get data
        original_preds = original_data["pred_label"].values
        true_labels = original_data["gt_label"].values
        calibrated_preds = calibration_result.get("calibration_probs", original_preds)
        
        # Calculate reliability data
        _, orig_accs, orig_confs, orig_counts = self._calculate_reliability_data(true_labels, original_preds)
        _, cal_accs, cal_confs, cal_counts = self._calculate_reliability_data(true_labels, calibrated_preds)
        
        # Plot
        self._plot_single_reliability(ax1, orig_confs, orig_accs, orig_counts, "Before Calibration", self.colors[0])
        method_name = calibration_result.get("method", "Unknown")
        self._plot_single_reliability(ax2, cal_confs, cal_accs, cal_counts, f"After Calibration ({method_name})", self.colors[1])
        
        # Add title
        original_ece = calibration_result.get("original_ece", 0)
        calibrated_ece = calibration_result.get("ece", 0)
        improvement = abs(original_ece - calibrated_ece)
        
        fig.suptitle(
            f"Calibration Effect Comparison\n"
            f"Original ECE: {original_ece:.4f} → Calibrated ECE: {calibrated_ece:.4f} "
            f"(Improvement: {improvement:.4f})",
            fontsize=16, fontweight="bold"
        )
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        
        return fig

    def _plot_single_reliability(
        self, ax: plt.Axes, bin_confs: np.ndarray, bin_accs: np.ndarray, 
        bin_counts: np.ndarray, title: str, color: str
    ) -> None:
        """Draw single reliability diagram"""
        # Filter valid bins
        valid_mask = (bin_counts > 0) & (~np.isnan(bin_accs)) & (~np.isnan(bin_confs))
        
        if not np.any(valid_mask):
            ax.text(0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontweight="bold")
            return
        
        valid_indices = np.where(valid_mask)[0]
        valid_confs = bin_confs[valid_mask]
        valid_accs = bin_accs[valid_mask]
        valid_counts = bin_counts[valid_mask]
        
        # Draw bars and line
        ax.bar(valid_indices, valid_accs, alpha=0.7, color=color, edgecolor="black", label="Accuracy")
        
        # Add count annotations
        for idx, acc, count in zip(valid_indices, valid_accs, valid_counts):
            ax.text(idx, acc + 0.02, f"{int(count)}", ha="center", va="bottom", fontsize=8)
        
        # Draw confidence line
        ax2 = ax.twinx()
        ax2.plot(valid_indices, valid_confs, "ro-", alpha=0.8, label="Average Confidence")
        
        # Perfect calibration line
        if len(valid_indices) > 1:
            ax2.plot(valid_indices, valid_confs, "k--", alpha=0.5, label="Perfect Calibration")
        
        # Set labels and limits
        ax.set_xlabel("Confidence Bins")
        ax.set_ylabel("Accuracy", color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.1)
        
        ax2.set_ylabel("Average Confidence", color='red')
        ax2.set_ylim(0, 1.1)
        
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels
        ax.set_xticks(valid_indices)
        x_labels = [f"{conf:.2f}" for conf in valid_confs]
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

    def plot_calibration_curve_comparison(
        self, 
        models_data: Dict[str, pd.DataFrame], 
        calibration_results: Dict[str, Dict[str, Any]], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Draw multi-model calibration curve comparison"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.8, label="Perfect Calibration", linewidth=2)
        
        plotted_models = 0
        
        for model_name, model_data in models_data.items():
            if model_name not in calibration_results:
                continue
                
            calibration_result = calibration_results[model_name]
            if not calibration_result.get("success", False):
                continue
            
            # Get data
            calibrated_preds = calibration_result.get("calibration_probs", model_data["pred_label"].values)
            true_labels = model_data["gt_label"].values
            
            # Calculate calibration curve
            _, bin_accs, bin_confs, bin_counts = self._calculate_reliability_data(true_labels, calibrated_preds)
            valid_mask = (bin_counts > 0) & (~np.isnan(bin_accs)) & (~np.isnan(bin_confs))
            
            if not np.any(valid_mask):
                continue
            
            valid_confs = bin_confs[valid_mask]
            valid_accs = bin_accs[valid_mask]
            
            # Plot
            color = self.colors[plotted_models % len(self.colors)]
            method = calibration_result.get("method", "Unknown")
            ece = calibration_result.get("ece", 0)
            
            ax.plot(valid_confs, valid_accs, "o-", color=color, alpha=0.8,
                   label=f"{model_name} ({method}, ECE: {ece:.3f})")
            plotted_models += 1
        
        ax.set_xlabel("Average Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Multi-Model Calibration Curve Comparison", fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        
        return fig

    def plot_metrics_comparison(
        self, 
        calibration_results: Dict[str, Dict[str, Any]], 
        metrics: List[str] = None, 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Draw metrics comparison"""
        if metrics is None:
            # metrics = ["ece", "mce", "brier_score", "accuracy"]
            metrics = ["ece", "brier_score", "f1_score", "accuracy"]
        
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows), dpi=self.dpi)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        model_names = [name for name, result in calibration_results.items() 
                      if result.get("success", False)]
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Collect data
            original_values = []
            calibrated_values = []
            model_labels = []
            
            for model_name in model_names:
                result = calibration_results[model_name]
                
                if metric == "ece":
                    orig_val = result.get("original_ece", 0)
                    cal_val = result.get("ece", 0)
                else:
                    cal_val = result.get(metric, 0)
                    orig_val = result.get(f"original_{metric}", cal_val)
                
                original_values.append(orig_val)
                calibrated_values.append(cal_val)
                model_labels.append(model_name)
            
            if not original_values:
                ax.text(0.5, 0.5, f"No {metric} Data", ha="center", va="center", transform=ax.transAxes)
                continue
            
            # Plot bars
            x = np.arange(len(model_labels))
            width = 0.35
            
            ax.bar(x - width/2, original_values, width, label="Before Calibration", 
                   color=self.colors[0], alpha=0.7)
            ax.bar(x + width/2, calibrated_values, width, label="After Calibration", 
                   color=self.colors[1], alpha=0.7)
            
            ax.set_xlabel("Models")
            ax.set_ylabel(f"{metric.upper()}")
            ax.set_title(f"{metric.upper()} Comparison", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        
        return fig

    def plot_confidence_distribution(
        self, 
        original_data: pd.DataFrame, 
        calibration_result: Dict[str, Any], 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Draw confidence distribution comparison"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), dpi=self.dpi)
        
        original_preds = original_data["pred_label"].values
        calibrated_preds = calibration_result.get("calibration_probs", original_preds)
        
        # Plot histograms
        ax1.hist(original_preds, bins=30, alpha=0.7, color=self.colors[0], 
                density=True, edgecolor="black")
        ax1.set_title("Before Calibration", fontweight="bold")
        ax1.set_xlabel("Prediction Probability")
        ax1.set_ylabel("Density")
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(calibrated_preds, bins=30, alpha=0.7, color=self.colors[1], 
                density=True, edgecolor="black")
        ax2.set_title("After Calibration", fontweight="bold")
        ax2.set_xlabel("Prediction Probability")
        ax2.set_ylabel("Density")
        ax2.grid(True, alpha=0.3)
        
        # Comparison
        ax3.hist(original_preds, bins=30, alpha=0.5, color=self.colors[0], 
                label="Before", density=True, edgecolor="black")
        ax3.hist(calibrated_preds, bins=30, alpha=0.5, color=self.colors[1], 
                label="After", density=True, edgecolor="black")
        ax3.set_title("Comparison", fontweight="bold")
        ax3.set_xlabel("Prediction Probability")
        ax3.set_ylabel("Density")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Sample data
    true_labels = np.random.binomial(1, 0.3, n_samples)
    pred_probs = np.random.beta(2, 2, n_samples) * 0.9 + 0.05
    
    original_data = pd.DataFrame({
        "pred_label": pred_probs,
        "gt_label": true_labels
    })
    
    calibration_result = {
        "method": "Platt Scaling",
        "success": True,
        "ece": 0.05,
        "original_ece": 0.15,
        "calibration_probs": pred_probs * 0.8 + 0.1
    }
    
    # Create visualizer and plot
    visualizer = CalibrationVisualizer()
    fig = visualizer.plot_reliability_diagram(original_data, calibration_result)
    plt.show()
