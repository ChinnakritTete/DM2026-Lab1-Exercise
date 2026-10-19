import os
from typing import Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool

# Ensure nltk tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass


def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """Lists files and directories in the workspace or a specified subdirectory.

        Args:
            subdirectory: Subdirectory path relative to workspace root. Default is '.' (workspace root).

        Returns:
            dict with result_id, subdirectory, directories, and files.
        """
        # Check for '..' path traversal attempts before normpath
        if ".." in subdirectory.split("/") or ".." in subdirectory.split("\\"):
            return {"error": f"Access denied: Relative path navigation '..' is not allowed."}

        if os.path.isabs(subdirectory):
            return {"error": f"Access denied: Absolute path '{subdirectory}' is not allowed."}

        target_dir = os.path.abspath(subdirectory)
        workspace_root = os.path.abspath(".")

        # Ensure target directory is inside workspace root
        if not (target_dir == workspace_root or target_dir.startswith(workspace_root + os.sep)):
            return {"error": f"Access denied: Path '{subdirectory}' is outside the repository workspace."}

        if not os.path.exists(target_dir):
            return {"error": f"Subdirectory '{subdirectory}' does not exist."}

        if not os.path.isdir(target_dir):
            return {"error": f"Path '{subdirectory}' is not a directory."}

        # List exactly one level of contents and sort
        entries = sorted(os.listdir(target_dir))
        directories = []
        files = []

        for entry in entries:
            if entry.startswith(".") or entry in ("__pycache__", "tests"):
                continue
            full_path = os.path.join(target_dir, entry)
            if os.path.isdir(full_path):
                directories.append(entry)
            else:
                files.append(entry)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )

        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        columns: Optional[List[str]] = None,
        max_chars: int = 200
    ) -> dict:
        """Inspects the first few rows of the working DataFrame, with capped row count and character truncation for long text.

        Args:
            n: Number of rows to inspect (capped at 20). Default is 5.
            columns: Optional list of column names to include. If None, includes all columns.
            max_chars: Maximum characters to display per text field before truncating. Default is 200.

        Returns:
            dict with result_id, total_rows, columns, and rows (list of row records).
        """
        if session.dataframe is None:
            return {"error": "DataFrame has not been loaded yet. Run load_dataset_tool first."}

        df = session.dataframe

        # Cap n at 20 rows
        n_capped = max(1, min(n, 20))

        if columns:
            valid_cols = [c for c in columns if c in df.columns]
            if not valid_cols:
                return {"error": f"None of the requested columns {columns} exist in DataFrame."}
            selected_df = df[valid_cols].head(n_capped).copy()
        else:
            selected_df = df.head(n_capped).copy()

        # Truncate string columns to max_chars
        for col in selected_df.columns:
            if selected_df[col].dtype == "object":
                selected_df[col] = selected_df[col].apply(
                    lambda val: (str(val)[:max_chars] + "...") if isinstance(val, str) and len(val) > max_chars else val
                )

        rows = selected_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "columns": list(selected_df.columns),
            "rows": rows
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n": n, "columns": columns, "max_chars": max_chars},
            summary=summary
        )

        return summary

    @tool
    def tokenize_tool(text_column: str = "text") -> dict:
        """Tokenizes text in each document into unigram tokens using lowercasing and nltk.word_tokenize.

        Args:
            text_column: Column name containing document text to tokenize. Default is 'text'.

        Returns:
            dict with result_id, n_documents, total_tokens, and sample_tokens (tokens for first 3 docs).
        """
        if session.dataframe is None:
            return {"error": "DataFrame has not been loaded yet. Run load_dataset_tool first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in DataFrame."}

        df = session.dataframe

        def _tokenize(val):
            if pd.isna(val) or not str(val).strip():
                return []
            text_lower = str(val).lower()
            return word_tokenize(text_lower)

        tokenized_series = df[text_column].apply(_tokenize)
        df["unigrams"] = tokenized_series
        session.artifacts["unigrams"] = tokenized_series.tolist()

        n_documents = len(df)
        total_tokens = sum(len(toks) for toks in tokenized_series)
        sample_tokens = tokenized_series.head(3).tolist()

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "sample_tokens": sample_tokens
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={"text_column": text_column},
            summary=summary
        )

        return summary

    @tool
    def check_missing_tool(text_column: str = "text") -> dict:
        """Checks for missing, null, or whitespace-only text values in the working DataFrame.

        Args:
            text_column: Name of the text column to inspect for missing values. Default is 'text'.

        Returns:
            dict with result_id, n_missing (count of missing/empty rows), and missing_indices (list of int indices).
        """
        if session.dataframe is None:
            return {"error": "DataFrame has not been loaded yet. Run load_dataset_tool first."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in DataFrame."}

        col = session.dataframe[text_column]

        # Identify missing: null/NaN or empty/whitespace-only strings
        is_missing = col.isna() | col.astype(str).str.strip().eq("")

        missing_indices = session.dataframe.index[is_missing].tolist()
        n_missing = len(missing_indices)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "n_missing": n_missing,
            "missing_indices": missing_indices
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={"text_column": text_column},
            summary=summary
        )

        return summary

    @tool
    def check_duplicates_tool(text_column: str = "text", drop: bool = False) -> dict:
        """Checks for duplicate documents and optionally drops all copies matching Master's keep=False across all columns.

        Args:
            text_column: Name of the text column to check for duplicates. Default is 'text'.
            drop: If True, removes all copies of duplicated rows using keep=False. Default is False.

        Returns:
            dict with result_id, n_duplicates, duplicate_indices (up to 10), drop, n_removed, and n_remaining.
        """
        if session.dataframe is None:
            return {"error": "DataFrame has not been loaded yet. Run load_dataset_tool first."}

        df = session.dataframe

        # pandas duplicated(keep='first') marks extra copies
        is_dup = df.duplicated(keep="first")
        all_dup_indices = df.index[is_dup].tolist()
        n_duplicates = len(all_dup_indices)
        duplicate_indices = all_dup_indices[:10]

        n_initial = len(df)

        if drop:
            # Drop ALL copies across all columns matching Master's X.drop_duplicates(keep=False, inplace=True)
            df.drop_duplicates(keep=False, inplace=True)
            session.dataframe = df

        n_remaining = len(session.dataframe)
        n_removed = n_initial - n_remaining

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "text_column": text_column,
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "drop": drop,
            "n_removed": n_removed,
            "n_remaining": n_remaining
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"text_column": text_column, "drop": drop},
            summary=summary
        )

        return summary

    @tool
    def describe_data_tool(text_column: str = "text", category_column: str = "category_name") -> dict:
        """Computes text length describe stats overall and per category, and creates a box plot on pending_figure.

        Args:
            text_column: Name of the text column to analyze. Default is 'text'.
            category_column: Name of the category column to group by. Default is 'category_name'.

        Returns:
            dict with result_id, overall (describe stats), and by_category (dict per category stats).
        """
        if session.dataframe is None:
            return {"error": "DataFrame has not been loaded yet. Run load_dataset_tool first."}

        df = session.dataframe

        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in DataFrame."}

        # Add text_length column on session.dataframe like Master
        df["text_length"] = df[text_column].fillna("").apply(len)

        # Overall stats
        overall_stats = df["text_length"].describe().to_dict()

        # Per category stats
        by_category_stats = {}
        if category_column in df.columns:
            for cat_name, group in df.groupby(category_column):
                by_category_stats[str(cat_name)] = group["text_length"].describe().to_dict()

        # Create box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_column in df.columns:
            sns.boxplot(data=df, x=category_column, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")

        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "by_category": by_category_stats
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={"text_column": text_column, "category_column": category_column},
            summary=summary
        )

        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> dict:
        """Subsamples rows from the working DataFrame using pandas sample.

        Args:
            n: Number of items from axis to return. Cannot be used with frac.
            frac: Fraction of axis items to return. Cannot be used with n.
            random_state: Seed for the random number generator for reproducible sampling.

        Returns:
            dict with result_id, sample_size, and sampled_indices.
        """
        if session.dataframe is None:
            return {"error": "DataFrame has not been loaded yet. Run load_dataset_tool first."}

        df = session.dataframe

        if n is None and frac is None:
            n = min(5, len(df))

        sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sample_size": len(sampled_df),
            "sampled_indices": sampled_indices
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "frac": frac, "random_state": random_state},
            summary=summary
        )

        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        describe_data_tool,
        sample_data_tool,
    ]
