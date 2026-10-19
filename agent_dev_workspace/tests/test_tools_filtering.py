import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # 1. Instantiate state and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate and invoke variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Call with threshold=0.15 (Known Answer #5)
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # 5. Assertions against Known Answer #5
    assert result["status"] == "success"
    assert result["total_terms"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1
    assert set(result["top_kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["top_removed_terms"] == ["always"]

    # Verify variances in the stored full_report
    stored_res = session.results_store[result["result_id"]]
    full_report = stored_res["full_report"]
    var_dict = dict(zip(full_report["term"], full_report["variance"]))

    assert np.isclose(var_dict["alpha"], 1.1875, atol=1e-3)
    assert np.isclose(var_dict["gamma"], 1.109375, atol=1e-3)
    assert np.isclose(var_dict["delta"], 0.25, atol=1e-3)
    assert np.isclose(var_dict["beta"], 0.234375, atol=1e-3)
    assert np.isclose(var_dict["always"], 0.109375, atol=1e-3)


def test_pearson_filter_tool():
    # 1. Instantiate state and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate and invoke pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Call with target_class="catB" (Known Answer #6)
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert result["status"] == "success"
    assert result["total_terms"] == 5

    # Check Pearson r values from full_report against Known Answer #6
    stored_res = session.results_store[result["result_id"]]
    full_report = stored_res["full_report"]
    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))

    assert np.isclose(r_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(r_dict["gamma"], 0.8307, atol=1e-3)
    assert np.isclose(r_dict["beta"], -0.7746, atol=1e-3)
    assert np.isclose(r_dict["alpha"], -0.6882, atol=1e-3)
    assert np.isclose(r_dict["always"], 0.3780, atol=1e-3)


def test_spearman_filter_tool():
    # 1. Instantiate state and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate and invoke spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Call with target_class="catB" (Known Answer #6)
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert result["status"] == "success"
    assert result["total_terms"] == 5

    # Check Spearman r values from full_report against Known Answer #6
    stored_res = session.results_store[result["result_id"]]
    full_report = stored_res["full_report"]
    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))

    assert np.isclose(r_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(r_dict["gamma"], 0.9363, atol=1e-3)
    assert np.isclose(r_dict["beta"], -0.7746, atol=1e-3)
    assert np.isclose(r_dict["alpha"], -0.7500, atol=1e-3)
    assert np.isclose(r_dict["always"], 0.3780, atol=1e-3)
