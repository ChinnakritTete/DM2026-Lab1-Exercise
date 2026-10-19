import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index_1: int, doc_index_2: int) -> dict:
        """Computes the cosine similarity between two documents in the global document-term matrix.

        Args:
            doc_index_1: Integer row index of the first document (0-indexed).
            doc_index_2: Integer row index of the second document (0-indexed).

        Returns:
            dict with result_id, doc_index_1, doc_index_2, and cosine_similarity.
        """
        if session.feature_matrix is None:
            return {"error": "DTM has not been built yet. Run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if not (0 <= doc_index_1 < n_docs) or not (0 <= doc_index_2 < n_docs):
            return {"error": f"Document index out of bounds. DTM has {n_docs} documents."}

        vec1 = session.feature_matrix[doc_index_1]
        vec2 = session.feature_matrix[doc_index_2]

        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim_matrix = cosine_similarity(vec1, vec2)
            sim = float(sim_matrix[0, 0])

        sim = round(float(np.nan_to_num(sim, nan=0.0)), 4)

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": sim
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary=summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Computes Pearson feature-vs-feature correlation matrix and heatmap over top terms by variance.

        Operates on global DTM session.feature_matrix and session.feature_names.
        Selects top 20 terms by variance, computes Pearson correlation matrix,
        generates seaborn heatmap on session.pending_figure, and stores summary.

        Returns:
            dict with result_id, terms (list of top term names), and correlation_matrix (2D list of floats).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "DTM has not been built yet. Run build_dtm_tool first."}

        dtm = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Compute term variance without converting entire sparse matrix to dense array
        means = np.asarray(dtm.mean(axis=0)).ravel()
        sq_means = np.asarray(dtm.power(2).mean(axis=0)).ravel()
        variances = sq_means - (means ** 2)

        # Get indices of top 20 terms by variance (descending order)
        n_terms = len(feature_names)
        top_k = min(20, n_terms)
        top_indices = np.argsort(variances)[::-1][:top_k]

        top_terms = feature_names[top_indices].tolist()

        # Extract only the top-k columns and convert to dense
        top_cols_dense = dtm[:, top_indices].toarray()

        # Compute Pearson correlation matrix across top columns (rows=vars, cols=obs for np.corrcoef)
        corr_matrix = np.corrcoef(top_cols_dense, rowvar=False)

        # Handle edge case where corrcoef returns scalar for 1 term
        if top_k == 1:
            corr_matrix = np.array([[1.0]])

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title("Top Term Correlation Heatmap")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": np.round(corr_matrix, 4).tolist()
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={},
            summary=summary
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
