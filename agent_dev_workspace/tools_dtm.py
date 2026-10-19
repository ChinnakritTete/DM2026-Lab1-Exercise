import os
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
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

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in session.feature_matrix.

        Calculates total occurrences of each term across all documents in the corpus and returns
        them sorted by frequency in descending order.

        Args:
            top_n: Optional integer to limit the returned frequencies to the top N terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No DTM found in session. Please run build_dtm_tool first."}

        # Calculate total occurrences of each term across all documents (column sums)
        term_counts = session.feature_matrix.sum(axis=0).A1.tolist()
        
        # Pair feature names with their total counts
        term_freqs = list(zip(session.feature_names, term_counts))
        
        # Sort terms descending by frequency
        term_freqs.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None and top_n > 0:
            term_freqs = term_freqs[:top_n]

        term_frequencies_dict = {term: count for term, count in term_freqs}

        result_id = session.next_result_id("tf")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_terms": len(session.feature_names),
            "returned_terms": len(term_frequencies_dict),
            "term_frequencies": term_frequencies_dict
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Generate a heatmap visualization of a slice of the document-term matrix.

        Slices the raw document-term matrix for up to n_documents and n_terms, creates an annotated heatmap,
        sets it on session.pending_figure, and returns the matrix slice and labels.

        Args:
            n_terms: Number of terms (columns) to include from the start of the vocabulary.
            n_documents: Number of documents (rows) to include from the start of the corpus.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No DTM found in session. Please run build_dtm_tool first."}

        # Extract positional slice
        matrix_slice_sparse = session.feature_matrix[:n_documents, :n_terms]
        dense_slice = matrix_slice_sparse.toarray()

        actual_n_docs, actual_n_terms = dense_slice.shape
        term_labels = session.feature_names[:actual_n_terms]
        doc_labels = [f"doc_{i}" for i in range(actual_n_docs)]

        # Generate plot
        fig, ax = plt.subplots(figsize=(max(6, actual_n_terms * 0.8), max(4, actual_n_docs * 0.5)))
        sns.heatmap(
            dense_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=term_labels,
            yticklabels=doc_labels,
            ax=ax
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Attach figure to session for rendering/saving
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": actual_n_docs,
            "n_terms": actual_n_terms,
            "term_labels": term_labels,
            "matrix_slice": dense_slice.tolist()
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
