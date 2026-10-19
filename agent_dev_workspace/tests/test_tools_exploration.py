import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_feature_correlation_matrix_tool_known_answer_19():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student's build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate exploration tools
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Run feature_correlation_matrix_tool
    res = feature_corr_tool.invoke({})

    # Known Answer 19 assertions
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    expected_corr_matrix = np.array([
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ])

    actual_corr_matrix = np.array(res["correlation_matrix"])

    assert np.allclose(actual_corr_matrix, expected_corr_matrix, atol=1e-3)
    assert session.pending_figure is not None


def test_cosine_similarity_tool_known_answer_16():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student's build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer 16 assertions
    # Doc 0 vs 1: 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert pytest.approx(res_0_1["cosine_similarity"], abs=1e-3) == 0.9847

    # Doc 0 vs 3: 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert pytest.approx(res_0_3["cosine_similarity"], abs=1e-3) == 0.0909

    # Doc 3 vs 4: 0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert pytest.approx(res_3_4["cosine_similarity"], abs=1e-3) == 0.9847

    # Doc 0 vs 7 (empty text document): 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["cosine_similarity"] == 0.0
