"""Project held-out slides onto a trained atlas."""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False


class AtlasProjector:
    """Project new data onto a trained atlas."""

    def __init__(self):
        self.scaler_ = None
        self.pca_ = None
        self.umap_reducer_ = None
        self.adata_train_ = None
        self.cluster_centroids_ = None  # {cluster_id: centroid_vector}
        self.knn_pseudotime_ = None     # KNN regressor for DPT fallback
        self.is_fitted_ = False

    @classmethod
    def from_training(
        cls,
        scaler,
        pca,
        umap_reducer,
        adata_train: "ad.AnnData",
        cluster_centroids: Dict,
    ) -> "AtlasProjector":
        """Build projector from training artifacts."""
        proj = cls()
        proj.scaler_ = scaler
        proj.pca_ = pca
        proj.umap_reducer_ = umap_reducer
        proj.adata_train_ = adata_train

        # project() applies scaler+PCA to get raw PCA features, so both centroid
        # matching and KNN pseudotime must be trained in that same space.
        # When Harmony or scVI was used, the pre-correction PCA is in X_pca_original.
        if "X_pca_original" in adata_train.obsm:
            train_pca = adata_train.obsm["X_pca_original"]
            print("  Using X_pca_original (pre-Harmony/scVI) for KNN to match projection input.")
        else:
            train_pca = adata_train.X

        # With scVI the centroids are in the 30-dim latent space but project()
        # outputs PCA-dim features (e.g. 242).  Detect the mismatch and recompute
        # centroids in PCA space so kneighbors() sees consistent dimensions.
        sample_centroid = next(iter(cluster_centroids.values()))[0]
        if sample_centroid.shape[0] != train_pca.shape[1]:
            print(f"  Centroid dim ({sample_centroid.shape[0]}) != PCA dim "
                  f"({train_pca.shape[1]}); recomputing centroids in PCA space.")
            cluster_arr = adata_train.obs["cluster"].values
            proj.cluster_centroids_ = {
                label: (train_pca[cluster_arr == label].mean(axis=0), None)
                for label in sorted(set(cluster_arr))
            }
        else:
            proj.cluster_centroids_ = cluster_centroids

        # Train a KNN pseudotime regressor as the fallback path.
        print("  Training KNN pseudotime regressor (fallback)...")
        train_pt = adata_train.obs["pseudotime"].values
        proj.knn_pseudotime_ = KNeighborsRegressor(
            n_neighbors=min(15, len(train_pca) - 1),
            weights="distance",
            n_jobs=-1,
        )
        proj.knn_pseudotime_.fit(train_pca, train_pt)

        proj.is_fitted_ = True
        return proj

    def project(
        self,
        raw_features: np.ndarray,
        slide_ids: Optional[np.ndarray] = None,
        method: str = "knn",
    ) -> "ad.AnnData":
        """
        Project new features onto the training atlas.

        Args:
            raw_features: (N, 768) Phikon embeddings from test slides.
            slide_ids: (N,) slide identifiers.
            method: "ingest" for scanpy ingest, "knn" for KNN regression.

        Returns:
            AnnData with projected cluster labels, UMAP coords, and pseudotime.
        """
        if not self.is_fitted_:
            raise RuntimeError("Projector not fitted. Use from_training() or load().")

        # Apply the same preprocessing as training.
        print(f"  Projecting {len(raw_features)} test patches...")
        X_scaled = self.scaler_.transform(raw_features)
        X_pca = self.pca_.transform(X_scaled)

        # Build test AnnData.
        adata_test = ad.AnnData(X=X_pca.astype(np.float32))
        if slide_ids is not None:
            adata_test.obs["slide_id"] = slide_ids.astype(str)

        if method == "ingest" and SCANPY_AVAILABLE:
            return self._project_ingest(adata_test)
        else:
            return self._project_knn(adata_test, X_pca)

    def _project_ingest(self, adata_test: "ad.AnnData") -> "ad.AnnData":
        """Use scanpy.tl.ingest for cluster + UMAP transfer."""
        print("  Using scanpy ingest for projection...")
        try:
            sc.tl.ingest(adata_test, self.adata_train_, obs="cluster")
            print(f"  Ingest transferred clusters to {len(adata_test)} patches.")

            # Pseudotime via the KNN fallback.
            pt = self.knn_pseudotime_.predict(adata_test.X)
            adata_test.obs["pseudotime"] = np.clip(pt, 0.0, 1.0)
            return adata_test

        except Exception as exc:
            print(f"  WARNING: Ingest failed ({exc}), falling back to KNN.")
            return self._project_knn(adata_test, adata_test.X)

    def _project_knn(self, adata_test: "ad.AnnData", X_pca: np.ndarray) -> "ad.AnnData":
        """KNN-based projection: nearest centroid for cluster, KNN for pseudotime."""
        print("  Using KNN projection...")

        # Assign clusters by nearest centroid in PCA space.
        centroids_ids = sorted(self.cluster_centroids_.keys())
        centroid_matrix = np.array([self.cluster_centroids_[c][0] for c in centroids_ids])

        # scVI stores centroids in its 30-dim latent space, but project() produces
        # PCA-dim features (e.g. 242).  On the load() path the saved JSON centroids
        # still carry scVI dims, so detect and fix the mismatch here too.
        if centroid_matrix.shape[1] != X_pca.shape[1]:
            if (self.adata_train_ is not None
                    and "X_pca_original" in self.adata_train_.obsm):
                print(f"  Centroid dim ({centroid_matrix.shape[1]}) != projection dim "
                      f"({X_pca.shape[1]}); recomputing centroids from X_pca_original.")
                train_pca = self.adata_train_.obsm["X_pca_original"]
                cluster_arr = self.adata_train_.obs["cluster"].values
                self.cluster_centroids_ = {
                    label: (train_pca[cluster_arr == label].mean(axis=0), None)
                    for label in sorted(set(cluster_arr))
                }
                centroids_ids = sorted(self.cluster_centroids_.keys())
                centroid_matrix = np.array(
                    [self.cluster_centroids_[c][0] for c in centroids_ids]
                )
            else:
                raise ValueError(
                    f"Centroid dim ({centroid_matrix.shape[1]}) != projection dim "
                    f"({X_pca.shape[1]}).  Projector was saved with scVI centroids "
                    f"but adata_train_ is unavailable.  Re-run LOO training to rebuild."
                )

        nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nbrs.fit(centroid_matrix)
        _, indices = nbrs.kneighbors(X_pca)
        cluster_labels = np.array([str(centroids_ids[i[0]]) for i in indices])
        adata_test.obs["cluster"] = cluster_labels

        # Predict pseudotime with KNN regression.
        pt = self.knn_pseudotime_.predict(X_pca)
        adata_test.obs["pseudotime"] = np.clip(pt, 0.0, 1.0)

        # UMAP projection.
        if self.umap_reducer_ is not None:
            try:
                X_umap = self.umap_reducer_.transform(X_pca)
                adata_test.obsm["X_umap"] = X_umap.astype(np.float32)
            except Exception:
                print("  WARNING: UMAP transform failed for test data.")

        return adata_test

    # Persistence

    def save(self, output_dir: str):
        """Save all projector components to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler_, f)
        with open(out / "pca.pkl", "wb") as f:
            pickle.dump(self.pca_, f)
        with open(out / "umap_reducer.pkl", "wb") as f:
            pickle.dump(self.umap_reducer_, f)
        with open(out / "knn_pseudotime.pkl", "wb") as f:
            pickle.dump(self.knn_pseudotime_, f)

        # Save cluster centroids
        centroids_serializable = {
            str(k): v[0].tolist() for k, v in self.cluster_centroids_.items()
        }
        with open(out / "cluster_centroids.json", "w") as f:
            json.dump(centroids_serializable, f, indent=2)

        # Save training AnnData
        if self.adata_train_ is not None:
            self.adata_train_.write(out / "adata_train.h5ad")

        # Cast numpy scalars to plain Python types; json.dump cannot serialize
        # np.float32 or np.int64.
        metadata = {
            "n_training_samples": int(len(self.adata_train_)) if self.adata_train_ is not None else 0,
            "pca_components": int(self.pca_.n_components_) if self.pca_ else None,
            "pca_variance_explained": float(np.sum(self.pca_.explained_variance_ratio_)) if self.pca_ else None,
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Projector saved to: {out}")

    @classmethod
    def load(cls, load_dir: str) -> "AtlasProjector":
        """Load a previously saved projector."""
        ld = Path(load_dir)
        proj = cls()

        with open(ld / "scaler.pkl", "rb") as f:
            proj.scaler_ = pickle.load(f)
        with open(ld / "pca.pkl", "rb") as f:
            proj.pca_ = pickle.load(f)
        with open(ld / "umap_reducer.pkl", "rb") as f:
            proj.umap_reducer_ = pickle.load(f)
        with open(ld / "knn_pseudotime.pkl", "rb") as f:
            proj.knn_pseudotime_ = pickle.load(f)

        with open(ld / "cluster_centroids.json") as f:
            raw = json.load(f)
            proj.cluster_centroids_ = {
                int(k) if k.lstrip("-").isdigit() else k: (np.array(v), None)
                for k, v in raw.items()
            }

        h5ad_path = ld / "adata_train.h5ad"
        if h5ad_path.exists() and SCANPY_AVAILABLE:
            proj.adata_train_ = sc.read_h5ad(h5ad_path)

        proj.is_fitted_ = True
        print(f"  Projector loaded from: {ld}")
        return proj
