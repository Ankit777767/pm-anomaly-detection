from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
PALETTE = {"normal": "#4C72B0", "failure": "#CA17A3"}#sort of legends

def _save(fig, name: str):#the underscore is juast for internal calling
  path = FIGURES_DIR / name
  fig.savefig(path, dpi=150, bbox_inches="tight")
  return path

def plot_class_imbalance(labels: pd.DataFrame,save_as="01_class_imbalance.png"):
  counts = labels["machine_failure"].value_counts().sort_index()
  pct = 100*counts/counts.sum()#there are two labels-0 and 1, so it will convert into percentage of machine failure contributions
  fig, ax = plt.subplots(figsize=(5, 4))
  bars = ax.bar(["Normal", "Failure"], counts.values, color=[PALETTE["normal"], PALETTE["failure"]])
  for bar, p in zip(bars, pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{p:.1f}%", ha="center", va="bottom", fontweight="bold")
  ax.set_title("Class Balance: Machine Failure (label used for eval only)")
  ax.set_ylabel("Count")
  _save(fig, save_as)     
  return fig



def plot_failure_mode_breakdown(labels: pd.DataFrame, save_as="02_failure_modes.png"):
  modes = ["twf", "hdf", "pwf", "osf", "rnf"]
  counts = labels[modes].sum().sort_values(ascending=False)
  #by deafult sum() adds column wise with axis = 0
  fig, ax = plt.subplots(figsize=(6, 4))
  ax.barh(counts.index.str.upper(), counts.values, color="#C44E52")
  ax.set_title("Failure Mode Frequency (TWF/HDF/PWF/OSF/RNF)")
  ax.set_xlabel("Count")
  _save(fig, save_as)
  return fig


def plot_feature_distributions(features: pd.DataFrame, labels: pd.DataFrame, cols: list, save_as="03_feature_distributions.png"):
  merged = features.merge(labels[["udi", "machine_failure"]], on="udi")
  merged["status"] = merged["machine_failure"].map({0: "normal", 1: "failure"})

  n = len(cols)
  fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
  if n == 1:
      axes = [axes]
  for ax, col in zip(axes, cols):
      sns.kdeplot(data=merged, x=col, hue="status", ax=ax,
                  palette=PALETTE, fill=True, alpha=0.3, common_norm=False)
      ax.set_title(col)
  fig.suptitle("Feature Distributions: Normal vs. Failure", y=1.03)
  _save(fig, save_as)
  return fig


def plot_correlation_heatmap(features: pd.DataFrame, cols: list, save_as="04_correlation_heatmap.png"):
  fig, ax = plt.subplots(figsize=(7, 6))
  corr = features[cols].corr()
  sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
              square=True, ax=ax, cbar_kws={"shrink": 0.8})
  ax.set_title("Feature Correlation Matrix")
  _save(fig, save_as)
  return fig


def plot_pca_scree(explained_variance_ratio, cumulative_variance, save_as="05_pca_scree.png"):
  fig, ax1 = plt.subplots(figsize=(6, 4))
  x = np.arange(1, len(explained_variance_ratio) + 1)
  ax1.bar(x, explained_variance_ratio, color="#4C72B0", alpha=0.7,
          label="Individual")
  ax1.set_xlabel("Principal Component")
  ax1.set_ylabel("Explained Variance Ratio")
  ax1.set_xticks(x)

  ax2 = ax1.twinx()
  ax2.plot(x, cumulative_variance, color="#C44E52", marker="o", label="Cumulative")
  ax2.axhline(0.9, color="gray", linestyle="--", linewidth=1)
  ax2.set_ylabel("Cumulative Explained Variance")
  ax2.set_ylim(0, 1.05)

  fig.suptitle("PCA Scree Plot")
  fig.legend(loc="center right", bbox_to_anchor=(0.9, 0.5))
  _save(fig, save_as)
  return fig


def plot_2d_embedding(embedding: np.ndarray, labels: pd.DataFrame, title: str, save_as: str):
  status = labels["machine_failure"].map({0: "normal", 1: "failure"}).values

  fig, ax = plt.subplots(figsize=(6, 5))
  for label, color in PALETTE.items():
      mask = status == label
      # Plot normal points first (background), failures on top with more emphasis
      size = 12 if label == "normal" else 35
      alpha = 0.35 if label == "normal" else 0.9
      zorder = 1 if label == "normal" else 2
      ax.scatter(embedding[mask, 0], embedding[mask, 1], c=color, label=label,
                  s=size, alpha=alpha, zorder=zorder, edgecolors="none")
  ax.set_title(title)
  ax.set_xlabel("Component 1")
  ax.set_ylabel("Component 2")
  ax.legend()
  _save(fig, save_as)
  return fig





