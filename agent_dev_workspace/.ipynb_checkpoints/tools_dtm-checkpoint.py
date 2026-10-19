import os
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: float = 1,
        max_df: float = 1.0,
        max_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """Build a document-term matrix (DTM) from session.dataframe text.

        Constructs a CountVectorizer on session.dataframe['text'], fits and transforms it into
        session.feature_matrix, updates session.feature_names, labels, and categories, and stores
        the fitted vectorizer under session.artifacts['count_vectorizer']. Calculates and reports
        matrix sparsity.

        Args:
            ngram_range: Lower and upper boundary of range of n-values for n-grams, as a list [min_n, max_n].
            min_df: Minimum document frequency for terms to be included.
            max_df: Maximum document frequency for terms to be included.
            max_features: Maximum number of features to extract.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataframe loaded in session."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "'text' column missing from session dataframe."}

        # Convert list to tuple for CountVectorizer schema safety
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features
        )

        texts = session.dataframe["text"].fillna("").tolist()
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session attributes
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_sample": feature_names[:10]
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={
                "ngram_range": ngram_range,
                "min_df": min_df,
                "max_df": max_df,
                "max_features": max_features
            },
            summary=summary
        )

        return summary

    return [build_dtm_tool]
