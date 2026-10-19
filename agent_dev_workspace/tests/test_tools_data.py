import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools


def test_inspect_data_tool():
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

    # Instantiate student's data tools
    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Run inspect_data_tool with n=3, max_chars=10
    res = inspect_data_tool.invoke({"n": 3, "max_chars": 10})

    assert "error" not in res, f"Tool returned error: {res.get('error')}"
    assert res["total_rows"] == 8
    assert len(res["rows"]) == 3

    # Check truncation (first row text is 'always alpha alpha alpha beta', 29 chars -> truncated to 'always alp...')
    first_row_text = res["rows"][0]["text"]
    assert len(first_row_text) == 13  # 10 chars + '...'
    assert first_row_text.endswith("...")


def test_tokenize_tool_known_answer_10():
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

    # Instantiate student's data tools
    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Run tokenize_tool
    res = tokenize_tool.invoke({"text_column": "text"})

    assert "error" not in res, f"Tool returned error: {res.get('error')}"
    assert res["n_documents"] == 8

    # Verify unigrams in session.dataframe
    unigrams = session.dataframe["unigrams"].tolist()

    # Known answer #10 in TEST_FIXTURE.md
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert unigrams == expected_tokens


def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_data_tools(session)
    list_files = next(t for t in tools if t.name == "list_files_tool")

    # Known answer #11 in TEST_FIXTURE.md:
    # list_files_tool(subdirectory="newdataset") should return directories: [], files: ["Reddit-stock-sentiment.csv"]
    res = list_files.invoke({"subdirectory": "newdataset"})

    assert "error" not in res, f"Tool returned an error: {res.get('error')}"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_refuses_outside_path():
    session = SessionState()
    tools = make_data_tools(session)
    list_files = next(t for t in tools if t.name == "list_files_tool")

    # Path traversal with .. should be refused
    res = list_files.invoke({"subdirectory": "../"})
    assert "error" in res
    assert "Access denied" in res["error"]


def test_list_files_tool_nonexistent_directory():
    session = SessionState()
    tools = make_data_tools(session)
    list_files = next(t for t in tools if t.name == "list_files_tool")

    # Non-existent directory should report clear error
    res = list_files.invoke({"subdirectory": "nonexistent_folder_xyz"})
    assert "error" in res
    assert "does not exist" in res["error"]


def test_check_missing_tool_known_answer_2():
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

    # Instantiate student's data tools
    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Run check_missing_tool
    res = check_missing_tool.invoke({})

    # Known Answer 2 assertions
    assert res["n_missing"] == 1
    assert res["missing_indices"] == [7]


def test_check_duplicates_tool_known_answer_3():
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

    # Instantiate student's data tools
    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Test drop=False: check duplicate detection
    res_no_drop = check_duplicates_tool.invoke({"drop": False})
    assert res_no_drop["n_duplicates"] == 1
    assert res_no_drop["duplicate_indices"] == [6]
    assert res_no_drop["n_remaining"] == 8

    # Test drop=True: keep=False drops both copies (rows 5 and 6), leaving 6 rows
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["n_removed"] == 2
    assert res_drop["n_remaining"] == 6
    assert len(session.dataframe) == 6


def test_describe_data_tool_known_answer_18():
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

    # Instantiate student's data tools
    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # Run describe_data_tool
    res = describe_data_tool.invoke({})

    # Overall stats assertions
    overall = res["overall"]
    assert pytest.approx(overall["count"]) == 8
    assert pytest.approx(overall["mean"], abs=1e-3) == 19.875
    assert pytest.approx(overall["std"], abs=1e-3) == 9.4330
    assert pytest.approx(overall["min"]) == 0
    assert pytest.approx(overall["25%"], abs=1e-3) == 17.75
    assert pytest.approx(overall["50%"], abs=1e-3) == 20.5
    assert pytest.approx(overall["75%"], abs=1e-3) == 25.25
    assert pytest.approx(overall["max"]) == 30

    # Per category stats assertions
    by_cat = res["by_category"]

    cat_a = by_cat["catA"]
    assert pytest.approx(cat_a["count"]) == 4
    assert pytest.approx(cat_a["mean"], abs=1e-3) == 17.25
    assert pytest.approx(cat_a["std"], abs=1e-3) == 12.5
    assert pytest.approx(cat_a["min"]) == 0
    assert pytest.approx(cat_a["25%"], abs=1e-3) == 12.75
    assert pytest.approx(cat_a["50%"], abs=1e-3) == 20.0
    assert pytest.approx(cat_a["75%"], abs=1e-3) == 24.5
    assert pytest.approx(cat_a["max"]) == 29

    cat_b = by_cat["catB"]
    assert pytest.approx(cat_b["count"]) == 4
    assert pytest.approx(cat_b["mean"], abs=1e-3) == 22.5
    assert pytest.approx(cat_b["std"], abs=1e-3) == 5.7446
    assert pytest.approx(cat_b["min"]) == 18
    assert pytest.approx(cat_b["25%"], abs=1e-3) == 18.0
    assert pytest.approx(cat_b["50%"], abs=1e-3) == 21.0
    assert pytest.approx(cat_b["75%"], abs=1e-3) == 25.5
    assert pytest.approx(cat_b["max"]) == 30

    # Verify pending figure plot
    assert session.pending_figure is not None


def test_sample_data_tool_known_answer_13():
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

    # Instantiate student's data tools
    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Run sample_data_tool with n=4, random_state=42
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Known Answer 13 assertions
    assert res["sample_size"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
