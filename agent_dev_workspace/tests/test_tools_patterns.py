import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_known_answers_17_and_20():
    session = SessionState()

    # Load premade dataset tool and load fixture
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    pattern_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # --- Known Answer 17: variance / term_frequency filtering keeps ZERO terms ---
    res_var_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var_catA["status"] == "success"
    assert res_var_catA["patterns"] == []
    assert "no terms survived filtering" in res_var_catA["note"]

    res_tf_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf_catB["status"] == "success"
    assert res_tf_catB["patterns"] == []
    assert "no terms survived filtering" in res_tf_catB["note"]

    # --- Known Answer 17: tfidf filtering + fpgrowth ---
    res_tfidf_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tfidf_catA["status"] == "success"
    assert len(res_tfidf_catA["patterns"]) == 1
    assert res_tfidf_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_tfidf_catA["patterns"][0]["support"] == 3

    res_tfidf_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tfidf_catB["status"] == "success"
    assert len(res_tfidf_catB["patterns"]) == 1
    assert res_tfidf_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_tfidf_catB["patterns"][0]["support"] == 4

    # --- Known Answer 20: algorithm="topk" (FAE) ---
    res_topk_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catA["status"] == "success"
    assert len(res_topk_catA["patterns"]) == 1
    assert res_topk_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_catA["patterns"][0]["support"] == 3

    res_topk_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 3
    })
    assert res_topk_catB["status"] == "success"
    assert len(res_topk_catB["patterns"]) == 1
    assert res_topk_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_topk_catB["patterns"][0]["support"] == 4

    # --- Known Answer 20: algorithm="maxfpgrowth" (returns zero patterns) ---
    res_maxfp_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_maxfp_catA["status"] == "success"
    assert res_maxfp_catA["patterns"] == []

    res_maxfp_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 4
    })
    assert res_maxfp_catB["status"] == "success"
    assert res_maxfp_catB["patterns"] == []
