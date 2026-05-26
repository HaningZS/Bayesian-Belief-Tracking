from csbf.segmentation import fixed_token_chunks, rule_based_steps


def test_rule_based_steps_splits_reasoning_markers_without_empty_steps():
    trace = "Step 1: compute 2+2.\nTherefore 4.\nWait check: 2 plus 2 is 4."

    steps = rule_based_steps(trace)

    assert steps == [
        "Step 1: compute 2+2.",
        "Therefore 4.",
        "Wait check: 2 plus 2 is 4.",
    ]


def test_fixed_token_chunks_preserves_token_order_and_chunk_size():
    trace = "one two three four five six seven"

    chunks = fixed_token_chunks(trace, chunk_size=3)

    assert chunks == ["one two three", "four five six", "seven"]
