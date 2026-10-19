import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter Document-Term Matrix features based on variance across documents.

        Calculates variance for each term across all documents in session.feature_matrix.
        Terms with variance below the threshold are flagged for removal. Stores a full_report
        DataFrame formatted for visualize_result_tool.

        Args:
            threshold: Minimum variance threshold. Terms with variance >= threshold are kept.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No feature_matrix or feature_names found in session. Run build_dtm_tool first."
            }

        matrix = session.feature_matrix.toarray()
        variances = np.var(matrix, axis=0, ddof=0)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = df_report[df_report["variance"] >= threshold]
        removed_df = df_report[df_report["variance"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("var_filter")

        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "top_kept_terms": kept_terms[:10],
            "top_removed_terms": removed_terms[:10]
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter Document-Term Matrix features based on Pearson correlation with a target class.

        Calculates Pearson correlation between each term's frequency and a binary target class indicator.
        Filters and sorts by absolute correlation strength |r|, while preserving the signed pearson_r
        column in full_report so strong negative relationships remain visible.

        Args:
            target_class: The target category name to correlate against (one-vs-rest).
            threshold: Minimum absolute correlation threshold |r| to keep a term.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {
                "status": "error",
                "message": "Missing feature_matrix, feature_names, or labels in session. Run build_dtm_tool first."
            }

        matrix = session.feature_matrix.toarray()
        y = (session.labels == target_class).astype(float)

        if np.all(y == y[0]):
            return {
                "status": "error",
                "message": f"Target class '{target_class}' is either present in all documents or none."
            }

        pearson_rs = []
        for col_idx in range(matrix.shape[1]):
            x = matrix[:, col_idx]
            if np.all(x == x[0]):
                r = 0.0
            else:
                r, _ = pearsonr(x, y)
                if np.isnan(r):
                    r = 0.0
            pearson_rs.append(float(r))

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": pearson_rs,
            "abs_r": np.abs(pearson_rs)
        }).sort_values(by="abs_r", ascending=False).reset_index(drop=True)

        kept_df = df_report[df_report["abs_r"] >= threshold]
        removed_df = df_report[df_report["abs_r"] < threshold]

        full_report = df_report[["term", "pearson_r"]].copy()

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_df),
            "n_removed": len(removed_df),
            "top_kept_terms": kept_df["term"].tolist()[:10],
            "top_removed_terms": removed_df["term"].tolist()[:10]
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=full_report
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter Document-Term Matrix features based on Spearman rank correlation with a target class.

        Calculates Spearman rank correlation between each term's frequency and a binary target class indicator.
        Filters and sorts by absolute correlation strength |rho|, while preserving the signed spearman_r
        column in full_report for visualize_result_tool.

        Args:
            target_class: The target category name to correlate against (one-vs-rest).
            threshold: Minimum absolute rank correlation threshold |rho| to keep a term.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {
                "status": "error",
                "message": "Missing feature_matrix, feature_names, or labels in session. Run build_dtm_tool first."
            }

        matrix = session.feature_matrix.toarray()
        y = (session.labels == target_class).astype(float)

        if np.all(y == y[0]):
            return {
                "status": "error",
                "message": f"Target class '{target_class}' is either present in all documents or none."
            }

        spearman_rs = []
        for col_idx in range(matrix.shape[1]):
            x = matrix[:, col_idx]
            if np.all(x == x[0]):
                r = 0.0
            else:
                r, _ = spearmanr(x, y)
                if np.isnan(r):
                    r = 0.0
            spearman_rs.append(float(r))

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": spearman_rs,
            "abs_r": np.abs(spearman_rs)
        }).sort_values(by="abs_r", ascending=False).reset_index(drop=True)

        kept_df = df_report[df_report["abs_r"] >= threshold]
        removed_df = df_report[df_report["abs_r"] < threshold]

        full_report = df_report[["term", "spearman_r"]].copy()

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_df),
            "n_removed": len(removed_df),
            "top_kept_terms": kept_df["term"].tolist()[:10],
            "top_removed_terms": removed_df["term"].tolist()[:10]
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=full_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
