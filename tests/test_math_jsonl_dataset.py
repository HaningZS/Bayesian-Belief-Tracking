from pathlib import Path

import pytest

from csbf.math_jsonl_dataset import load_math_jsonl_examples


def test_load_math_jsonl_examples_detects_common_fields(tmp_path: Path):
    path = tmp_path / "hard_math.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":"aime-2025-i-1","problem":"Compute 40+2.","answer":"\\\\boxed{42}","level":"aime"}',
                '{"question_id":"olym-2","question":"Find x.","gold_answer":"7","level":"olympiad"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_math_jsonl_examples(path)

    assert [example.question_id for example in examples] == ["aime-2025-i-1", "olym-2"]
    assert [example.question for example in examples] == ["Compute 40+2.", "Find x."]
    assert [example.gold_answer for example in examples] == ["42", "7"]
    assert [example.level for example in examples] == ["aime", "olympiad"]


def test_load_math_jsonl_examples_filters_offset_limit_and_level(tmp_path: Path):
    path = tmp_path / "hard_math.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":"p0","problem":"too easy","answer":"1","level":"warmup"}',
                '{"id":"p1","problem":"hard 1","answer":"2","level":"Level 5"}',
                '{"id":"p2","problem":"hard 2","answer":"3","level":"5"}',
                '{"id":"p3","problem":"hard 3","answer":"4","level":"Level 5"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_math_jsonl_examples(path, levels=["Level 5"], offset=1, limit=1)

    assert len(examples) == 1
    assert examples[0].question_id == "p2"
    assert examples[0].level == "Level 5"


def test_load_math_jsonl_examples_supports_explicit_fields(tmp_path: Path):
    path = tmp_path / "custom.jsonl"
    path.write_text(
        '{"uid":"rimo-1","prompt":"What is 6 times 7?","target":"42","difficulty":"RIMO-N"}\n',
        encoding="utf-8",
    )

    examples = load_math_jsonl_examples(
        path,
        id_field="uid",
        question_field="prompt",
        answer_field="target",
        level_field="difficulty",
    )

    assert examples[0].question_id == "rimo-1"
    assert examples[0].question == "What is 6 times 7?"
    assert examples[0].gold_answer == "42"
    assert examples[0].level == "RIMO-N"


def test_load_math_jsonl_examples_rejects_missing_fields(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"bad","problem":"No answer."}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="answer"):
        load_math_jsonl_examples(path)
