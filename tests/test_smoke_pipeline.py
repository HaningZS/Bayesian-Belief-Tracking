from csbf.smoke import run_offline_smoke


def test_offline_smoke_pipeline_returns_beliefs_and_metrics_without_external_services():
    result = run_offline_smoke()

    assert result["num_traces"] == 3
    assert len(result["beliefs"]) == 3
    assert all(len(trace_beliefs) > 0 for trace_beliefs in result["beliefs"])
    assert 0.0 <= result["metrics"]["brier"] <= 1.0
