import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # 1. Instantiate state and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 2. Load the fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Import and instantiate build_dtm_tool
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # 4. Run build_dtm_tool with defaults
    result = build_dtm_tool.invoke({})

    # 5. Assert against Known Answer #4 in TEST_FIXTURE.md
    assert result["status"] == "success"
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert np.isclose(result["sparsity_pct"], 47.5, atol=1e-3)

    # Check session updates
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    # 1. Instantiate shared session
    session = SessionState()

    # 2. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Load tools_dtm tools
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # 4. Execute build_dtm_tool
    build_dtm_res = build_dtm_tool.invoke({})
    assert build_dtm_res.get("status") == "success"

    # 5. Execute term_frequency_tool
    tf_res = term_frequency_tool.invoke({})
    assert tf_res.get("status") == "success"

    # 6. Assert expected frequencies according to known answer 14
    term_freqs = tf_res["term_frequencies"]
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }

    for term, expected_count in expected_freqs.items():
        assert term_freqs.get(term) == pytest.approx(expected_count), f"Mismatch for term {term}"


def test_dtm_heatmap_tool_known_answer_21():
    # 1. Instantiate shared session
    session = SessionState()

    # 2. Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Load tools_dtm tools
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    # 4. Execute build_dtm_tool first
    build_dtm_res = build_dtm_tool.invoke({})
    assert build_dtm_res.get("status") == "success"

    # 5. Test default call (n_terms=20, n_documents=20)
    default_res = dtm_heatmap_tool.invoke({})
    assert default_res.get("status") == "success"
    assert default_res["n_documents"] == 8
    assert default_res["n_terms"] == 5
    assert default_res["term_labels"] == ["alpha", "always", "beta", "delta", "gamma"]
    
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert default_res["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # 6. Test slice call (n_terms=2, n_documents=3)
    slice_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert slice_res.get("status") == "success"
    assert slice_res["n_documents"] == 3
    assert slice_res["n_terms"] == 2
    assert slice_res["term_labels"] == ["alpha", "always"]
    assert slice_res["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
