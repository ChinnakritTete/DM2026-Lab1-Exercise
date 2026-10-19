import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Literal, Dict, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> dict:
        """Reduces document-term matrix dimensions to 2D coordinates and plots a scatter plot colored by category.

        Args:
            method: Dimensionality reduction technique: 'pca', 'tsne', or 'umap'. Default is 'pca'.
            n_components: Number of components/dimensions. Default is 2.
            random_state: Random seed for reproducible output (used by TSNE and UMAP). Default is 42.
            perplexity: Perplexity parameter for t-SNE (must be less than n_samples). Default is 30.0.
            n_neighbors: Number of neighbors parameter for UMAP (must be less than n_samples). Default is 15.

        Returns:
            dict with result_id, method, coordinates, and explained_variance_ratio (PCA only).
        """
        if session.feature_matrix is None:
            return {"error": "Document-term matrix has not been built yet. Run build_dtm_tool first."}

        X = session.feature_matrix
        X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        n_samples = X_dense.shape[0]

        # Standardize categories/labels for plotting
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            categories = list(session.labels)
        else:
            categories = ["Unassigned"] * n_samples

        explained_variance_ratio = None

        # Execute reduction algorithm
        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X_dense)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()

        elif method == "tsne":
            # Adjust perplexity if larger than or equal to n_samples
            eff_perplexity = perplexity if perplexity < n_samples else max(1.0, float(n_samples - 1))
            model = TSNE(
                n_components=n_components,
                perplexity=eff_perplexity,
                random_state=random_state
            )
            coords = model.fit_transform(X_dense)

        elif method == "umap":
            # Adjust n_neighbors if larger than or equal to n_samples
            eff_n_neighbors = n_neighbors if n_neighbors < n_samples else max(2, n_samples - 1)
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=eff_n_neighbors,
                random_state=random_state
            )
            coords = model.fit_transform(X_dense)

        else:
            return {"error": f"Unsupported reduction method '{method}'."}

        # Build scatter plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter_df = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1],
            "Category": categories
        })
        sns.scatterplot(data=scatter_df, x="Dim 1", y="Dim 2", hue="Category", style="Category", ax=ax, s=70)
        ax.set_title(f"Dimensionality Reduction using {method.upper()}")
        plt.tight_layout()

        session.pending_figure = fig
        session.artifacts[f"reduced_coords_{method}"] = coords.tolist()

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "coordinates": np.round(coords, 4).tolist(),
            "explained_variance_ratio": [round(v, 4) for v in explained_variance_ratio] if explained_variance_ratio else None
        }

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors
            },
            summary=summary
        )

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarizes document category labels into a 1-hot matrix using LabelBinarizer.

        Reads category_name from session.dataframe (or session.labels), fits sklearn.preprocessing.LabelBinarizer,
        ensures one 0/1 column per category (even for binary targets), stores the fitted binarizer and binary array
        in session.artifacts, and returns a summary. Leaves session.dataframe and session.labels unchanged.

        Returns:
            dict with result_id, status, n_samples, classes, and binarized_matrix.
        """
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.labels is not None:
            labels = session.labels
        else:
            return {
                "status": "error",
                "message": "No category_name found in session.dataframe or session.labels. Run load_dataset_tool first."
            }

        lb = LabelBinarizer()
        binarized = lb.fit_transform(labels)
        classes = lb.classes_.tolist()

        # Handle binary classification case where LabelBinarizer defaults to 1 column 1D array
        if len(classes) == 2 and binarized.shape[1] == 1:
            binarized = np.hstack([1 - binarized, binarized])

        # Store in session.artifacts without modifying session.dataframe or session.labels
        session.artifacts["label_binarizer"] = lb
        session.artifacts["binarized_labels"] = binarized.tolist()

        result_id = session.next_result_id("binarize_labels")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_samples": len(labels),
            "classes": classes,
            "shape": list(binarized.shape),
            "binarized_matrix": binarized.tolist()
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
