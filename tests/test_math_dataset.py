from csbf.math_dataset import load_math500_examples


class FakeMATHDataset:
    def __init__(self) -> None:
        self.rows = {
            "test": [
                {
                    "problem": "Find the value of $\\frac{3}{4} + \\frac{1}{4}$.",
                    "answer": "Adding the fractions: $\\frac{3}{4} + \\frac{1}{4} = \\boxed{1}$",
                    "subject": "Algebra",
                    "level": "Level 1",
                },
                {
                    "problem": "What is $\\sqrt{144}$?",
                    "answer": "We know $12^2 = 144$, so $\\sqrt{144} = \\boxed{12}$",
                    "subject": "Algebra",
                    "level": "Level 1",
                },
                {
                    "problem": "Simplify $\\frac{6}{8}$.",
                    "answer": "Dividing numerator and denominator by 2: $\\boxed{\\frac{3}{4}}$",
                    "subject": "Prealgebra",
                    "level": "Level 2",
                },
                {
                    "problem": "Solve a hard olympiad-style geometry problem.",
                    "answer": "The final value is $\\boxed{42}$.",
                    "subject": "Geometry",
                    "level": "Level 4",
                },
                {
                    "problem": "Solve a very hard number theory problem.",
                    "answer": "The residue is $\\boxed{17}$.",
                    "subject": "Number Theory",
                    "level": 5,
                },
            ]
        }

    def __getitem__(self, split: str):
        return self.rows[split]


def test_load_math500_examples_uses_correct_dataset():
    calls = []

    def fake_loader(name: str):
        calls.append(name)
        return FakeMATHDataset()

    examples = load_math500_examples(limit=2, loader=fake_loader)

    assert calls == ["HuggingFaceH4/MATH-500"]
    assert len(examples) == 2
    assert examples[0].question_id == "math500-0"
    assert examples[0].gold_answer == "1"
    assert examples[1].question_id == "math500-1"
    assert examples[1].gold_answer == "12"


def test_load_math500_examples_extracts_boxed_latex():
    def fake_loader(name: str):
        return FakeMATHDataset()

    examples = load_math500_examples(loader=fake_loader)

    assert len(examples) == 5
    assert examples[2].gold_answer == "\\frac{3}{4}"


def test_load_math500_examples_no_limit():
    def fake_loader(name: str):
        return FakeMATHDataset()

    examples = load_math500_examples(loader=fake_loader)
    assert len(examples) == 5


def test_load_math500_examples_filters_by_level_before_limit():
    def fake_loader(name: str):
        return FakeMATHDataset()

    examples = load_math500_examples(limit=1, levels=["Level 4", "Level 5"], loader=fake_loader)

    assert len(examples) == 1
    assert examples[0].question_id == "math500-3"
    assert examples[0].gold_answer == "42"
    assert examples[0].level == "Level 4"


def test_load_math500_examples_applies_offset_after_level_filtering():
    def fake_loader(name: str):
        return FakeMATHDataset()

    examples = load_math500_examples(limit=1, offset=1, levels=["Level 1"], loader=fake_loader)

    assert len(examples) == 1
    assert examples[0].question_id == "math500-1"
    assert examples[0].gold_answer == "12"


def test_load_math500_examples_normalizes_numeric_levels_for_filtering():
    def fake_loader(name: str):
        return FakeMATHDataset()

    examples = load_math500_examples(limit=1, levels=["Level 5"], loader=fake_loader)

    assert len(examples) == 1
    assert examples[0].question_id == "math500-4"
    assert examples[0].level == "Level 5"


def test_load_math500_examples_negative_limit_raises():
    def fake_loader(name: str):
        return FakeMATHDataset()

    try:
        load_math500_examples(limit=-1, loader=fake_loader)
        assert False, "should have raised"
    except ValueError:
        pass
