import os
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, Any, List
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as DF2DB
import PAMI.frequentPattern.basic.FPGrowth as FPGrowth
import PAMI.frequentPattern.topk.FAE as FAE
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mines frequent word patterns for a single category using PAMI algorithms after vocabulary filtering.

        Args:
            category_name: Target category to filter session.dataframe rows.
            filtering_method: Vocabulary filter method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Default is 'fpgrowth'.
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k count of highest support patterns (required for 'topk').

        Returns:
            dict with result_id, category_name, filtering_method, algorithm, and patterns list.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "No dataframe found in session. Run load_dataset_tool first."}

        df = session.dataframe
        if "category_name" in df.columns:
            cat_df = df[df["category_name"] == category_name].copy()
        elif "category" in df.columns:
            cat_df = df[df["category"] == category_name].copy()
        else:
            return {"status": "error", "message": "No category column found in session dataframe."}

        if cat_df.empty:
            return {"status": "error", "message": f"No documents found for category '{category_name}'."}

        texts = cat_df["text"].fillna("").tolist()

        # Step 1: Fresh category-local CountVectorizer(stop_words="english")
        vectorizer = CountVectorizer(stop_words="english")
        dtm_sparse = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result(
                tool_name="mine_patterns_tool",
                args={"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm},
                summary=summary
            )
            return summary

        dtm_dense = dtm_sparse.toarray()

        # Step 2: Calculate filtering metric across columns
        if filtering_method == "variance":
            scores = np.var(dtm_dense, axis=0, ddof=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores >= p5) & (scores <= p95)
        elif filtering_method == "term_frequency":
            scores = np.sum(dtm_dense, axis=0)
            p5, p95 = np.percentile(scores, 5), np.percentile(scores, 95)
            keep_mask = (scores >= p5) & (scores <= p95)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_sparse = tfidf_transformer.fit_transform(dtm_sparse)
            scores = np.asarray(tfidf_sparse.mean(axis=0)).ravel()
            p20 = np.percentile(scores, 20)
            keep_mask = scores >= p20
        else:
            return {"status": "error", "message": f"Unsupported filtering_method '{filtering_method}'."}

        surviving_indices = np.where(keep_mask)[0]
        surviving_terms = feature_names[surviving_indices]

        # Step 3: Handle zero surviving terms
        if len(surviving_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result(
                tool_name="mine_patterns_tool",
                args={"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm},
                summary=summary
            )
            return summary

        # Step 4: Build counts DataFrame (raw counts for surviving terms)
        filtered_dtm = dtm_dense[:, surviving_indices]
        counts_df = pd.DataFrame(filtered_dtm, columns=surviving_terms)

        # Remove documents with no surviving terms
        counts_df = counts_df[counts_df.sum(axis=1) > 0]

        if counts_df.empty:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no non-empty transactions remaining"
            }
            session.store_result(
                tool_name="mine_patterns_tool",
                args={"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm},
                summary=summary
            )
            return summary

        # Write to temporary files for PAMI DB conversion and algorithm output
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f_db, \
             tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f_out:
            db_path = f_db.name
            out_path = f_out.name

        try:
            # Convert counts DataFrame using DF2DB(df) and convert2TransactionalDatabase(db_path, '>=', 1)
            converter = DF2DB.DF2DB(counts_df)
            converter.convert2TransactionalDatabase(db_path, '>=', 1)

            # Step 5: Mine with PAMI algorithm
            if algorithm == "fpgrowth":
                effective_min_sup = min_sup if min_sup is not None else 1
                miner = FPGrowth.FPGrowth(db_path, minSup=effective_min_sup, sep="\t")
                miner.mine()
                miner.save(out_path)
            elif algorithm == "topk":
                effective_k = k if k is not None else 10
                miner = FAE.FAE(db_path, k=effective_k, sep="\t")
                miner.mine()
                miner.save(out_path)
            elif algorithm == "maxfpgrowth":
                effective_min_sup = min_sup if min_sup is not None else 1
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=effective_min_sup, sep="\t")
                miner.mine()
                miner.save(out_path)
            else:
                return {"status": "error", "message": f"Unsupported algorithm '{algorithm}'."}

            # Parse mined patterns output
            patterns = []
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.rsplit(":", 1)
                        if len(parts) == 2:
                            itemset_str, sup_str = parts[0].strip(), parts[1].strip()
                            itemset = itemset_str.split("\t") if "\t" in itemset_str else itemset_str.split()
                            patterns.append({
                                "pattern": itemset,
                                "support": int(float(sup_str))
                            })

        finally:
            for p in [db_path, out_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "status": "success",
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns),
            "patterns": patterns
        }

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            },
            summary=summary
        )

        return summary

    return [mine_patterns_tool]
