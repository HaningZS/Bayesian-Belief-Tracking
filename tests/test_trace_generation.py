from csbf.gsm8k import GSM8KExample
from csbf.trace_generation import build_gsm8k_trace_records


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def chat_json(self, messages, model, temperature, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return {
            "final_answer": "5",
            "steps": [
                {"text": "Jan starts with 2 apples.", "score": 0.8},
                {"text": "Buying 3 more gives 5.", "score": 0.9},
            ],
        }


def test_build_gsm8k_trace_records_returns_schema_records_with_observations():
    client = FakeClient()
    examples = [
        GSM8KExample(
            question_id="gsm8k-test-0",
            question="Jan has 2 apples and buys 3 more. How many?",
            gold_answer="5",
        )
    ]

    records = build_gsm8k_trace_records(
        examples,
        client=client,
        traces_per_question=2,
        model="deepseek-v4-flash",
        temperature=0.6,
        max_tokens=512,
    )

    assert len(records) == 2
    assert records[0].trace_id == "gsm8k-test-0-t0"
    assert records[0].correct is True
    assert records[0].observations[0].score == 0.8
    assert records[0].observations[0].concept_code is not None
    assert client.calls[0]["model"] == "deepseek-v4-flash"
