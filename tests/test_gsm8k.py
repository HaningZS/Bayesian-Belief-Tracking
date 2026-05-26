from csbf.gsm8k import load_gsm8k_examples


class FakeDataset:
    def __init__(self) -> None:
        self.rows = {
            "test": [
                {
                    "question": "Jan has 2 apples and buys 3 more. How many?",
                    "answer": "Jan has 2+3=5 apples.\n#### 5",
                },
                {
                    "question": "A pack has 4 cards. Two packs have how many?",
                    "answer": "Two packs have 4*2=8 cards.\n#### 8",
                },
            ]
        }

    def __getitem__(self, split: str):
        return self.rows[split]


def test_load_gsm8k_examples_uses_openai_gsm8k_main_config():
    calls = []

    def fake_loader(name: str, config: str):
        calls.append((name, config))
        return FakeDataset()

    examples = load_gsm8k_examples(split="test", limit=1, loader=fake_loader)

    assert calls == [("openai/gsm8k", "main")]
    assert examples[0].question_id == "gsm8k-test-0"
    assert examples[0].gold_answer == "5"


def test_load_gsm8k_examples_applies_offset_before_limit():
    def fake_loader(name: str, config: str):
        return FakeDataset()

    examples = load_gsm8k_examples(split="test", offset=1, limit=1, loader=fake_loader)

    assert len(examples) == 1
    assert examples[0].question_id == "gsm8k-test-1"
    assert examples[0].gold_answer == "8"
