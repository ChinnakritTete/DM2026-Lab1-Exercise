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
