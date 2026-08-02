"""
dimensionality_reduction.py
----------------------------
PCA and UMAP utilities used two ways in this project:

1. EDA / diagnostic use (this phase): project the *engineered feature
   space* down to 2D and color by the true failure label (which we have,
   but only use for looking, never for fitting) to sanity-check whether
   failures are even separable from normal operation before we commit to
   modeling them as anomalies.

2. Modeling use (Phase 3 ablation): PCA as a *preprocessing* step feeding
   into distance-based detectors (LOF, One-Class SVM), and as its own
   anomaly detector via reconstruction error. That's implemented here too
   (`pca_reconstruction_error`) so the same module serves both phases.
"""

from dataclasses import dataclass# wrapper-decorator

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA #good point to remeberr
from sklearn.preprocessing import StandardScaler

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

@dataclass
class PCAresult:
  embedding:np.ndarray
  explained_variance_ratio: np.ndarray
  cumulative_variance: np.ndarray
  components: np.ndarray
  feature_names: list
  model: PCA

def run_pca(X:pd.DataFrame,n_components:int=None)-> PCAresult:
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  pca = PCA(n_components=n_components,random_state = 42)
  embeddings = pca.fit_transform(X_scaled)

  return PCAresult(
    embedding = embeddings,
    explained_variance_ratio = pca.explained_variance_ratio_,
    cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
    components=pca.components_,
    feature_names=list(X.columns),
    model=pca,
  )

def run_umap(X: pd.DataFrame, n_neighbors: int = 15, min_dist: float = 0.1,
             n_components: int = 2, random_state: int = 42) -> np.ndarray:

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=n_components,
    random_state=random_state,
  )

  return reducer.fit_transform(X_scaled)










