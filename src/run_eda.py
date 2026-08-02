import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))#copy paste -just

from data_loader import load_raw, split_features_labels, basic_sanity_checks, FEATURE_COLUMNS
from features import add_feature_engg, encode_type, FEATURE_SETS
from dimensionality_reduction import run_pca, run_umap
import visualize as viz

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "eda_summary.md"
 
def main():
  log_lines = ["# EDA Summary — Predictive Maintenance Anomaly Detection\n"]
  
  # 1. Load + split (labels quarantined from the start)
  data = load_raw()
  features, labels = split_features_labels(data)
  checks = basic_sanity_checks(data)
  print("Sanity checks:", checks)
  log_lines.append("## Dataset sanity checks\n")

  for k,v in checks.items():
    log_lines.append(f"- **{k}**: {v}")
  
  # 2. Feature engineering
  features_eng = add_feature_engg(features)
  features_encoded = encode_type(features_eng)

  # 3. Visual EDA
  viz.plot_class_imbalance(labels)#we will plot the machine_failure imbalance 
  viz.plot_failure_mode_breakdown(labels)
  viz.plot_feature_distributions(
      features_eng, labels,
      cols=["power_w", "temp_diff_k", "wear_torque_product", "tool_wear_min"]
  )
  corr_cols = FEATURE_SETS["engineered"]
  viz.plot_correlation_heatmap(features_eng, cols=corr_cols)
  #new features accomodated

  #Dimensionality Reduction Step--Most Important
  X = features_encoded[[c for c in features_encoded.columns if c not in ("udi","product_id")]]
  pca_full = run_pca(X,n_components=None)
  viz.plot_pca_scree(pca_full.explained_variance_ratio, pca_full.cumulative_variance)
  #this scree plot will tell the variance from all principle components and then a plot to explain cumulative variance

  #so i think 3 will be good, if we take num_components =3
  #we will select the num_principle_components acc to the 90percent cumulative variance

  n_90 = int((pca_full.cumulative_variance<0.90).sum()+1)
  print(n_90)
  #cumulative_variance is the array of cumvar , it will be truwe upto pc
  log_lines.append(f"\n## Dimensionality reduction\n")
  log_lines.append(f"- Components needed for 90% variance: **{n_90}** / {X.shape[1]}")
  pca_2d = run_pca(X, n_components=2)
  viz.plot_2d_embedding(pca_2d.embedding, labels,
                        title="PCA (2D) — colored by true failure label (EDA only)",
                        save_as="06_pca_2d_embedding.png")

  
  umap_2d = run_umap(X)
  viz.plot_2d_embedding(umap_2d, labels,
                          title="UMAP (2D) — colored by true failure label (EDA only)",
                          save_as="07_umap_2d_embedding.png")

  umap_2d = run_umap(X)
  viz.plot_2d_embedding(umap_2d, labels,
                          title="UMAP (2D) — colored by true failure label (EDA only)",
                          save_as="07_umap_2d_embedding.png")

  log_lines.append(f"- PC1 explained variance: {pca_full.explained_variance_ratio[0]:.1%}")
  log_lines.append(f"- PC2 explained variance: {pca_full.explained_variance_ratio[1]:.1%}")

  REPORT_PATH.parent.mkdir(exist_ok=True)
  REPORT_PATH.write_text("\n".join(log_lines))
  print(f"\nSaved summary to {REPORT_PATH}")
  print("Saved figures to reports/figures/")


if __name__ == "__main__":
  main()