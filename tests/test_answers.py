from csbf.answers import extract_final_answer, is_correct_answer, normalize_answer


def test_extract_final_answer_prefers_gsm8k_hash_marker():
    text = "We compute carefully.\n#### 42"

    assert extract_final_answer(text) == "42"


def test_extract_final_answer_handles_boxed_math_answer():
    text = "The final result is \\boxed{\\frac{3}{4}}."

    assert extract_final_answer(text) == "\\frac{3}{4}"


def test_normalize_answer_removes_commas_units_and_outer_spacing():
    assert normalize_answer(" $1,024.0 apples ") == "1024"


def test_is_correct_answer_uses_normalized_exact_match():
    assert is_correct_answer("#### $1,024", "1024")


def test_extract_boxed_fraction():
    assert extract_final_answer("So the answer is \\boxed{\\frac{3}{4}}.") == "\\frac{3}{4}"


def test_extract_boxed_sqrt():
    assert extract_final_answer("The result is \\boxed{\\sqrt{2}}.") == "\\sqrt{2}"


def test_extract_boxed_tuple():
    assert extract_final_answer("The solution is \\boxed{(2, 5)}") == "(2, 5)"


def test_extract_final_answer_ignores_empty_boxed_instruction():
    text = "End with the final answer in \\boxed{}.\nReasoning done. The final answer is 51."

    assert extract_final_answer(text) == "51"


def test_normalize_preserves_latex_fraction():
    assert normalize_answer("\\frac{3}{4}") == "\\frac{3}{4}"


def test_normalize_preserves_latex_sqrt():
    assert normalize_answer("\\sqrt{2}") == "\\sqrt{2}"


def test_is_correct_latex_fraction():
    assert is_correct_answer("The answer is \\boxed{\\frac{3}{4}}", "\\frac{3}{4}")


def test_is_correct_latex_with_whitespace():
    assert is_correct_answer("\\boxed{ \\frac{3}{4} }", "\\frac{3}{4}")


def test_symbolic_checker_matches_latex_fraction_to_decimal_without_changing_default():
    assert not is_correct_answer("\\boxed{\\frac{1}{2}}", "0.5")
    assert is_correct_answer("\\boxed{\\frac{1}{2}}", "0.5", checker="symbolic")


def test_symbolic_checker_matches_tuple_components():
    assert is_correct_answer("\\boxed{(\\frac{1}{2}, \\sqrt{4})}", "(0.5, 2)", checker="symbolic")


def test_symbolic_checker_matches_pi_approximation():
    assert is_correct_answer("\\boxed{2\\pi}", "6.28318530718", checker="symbolic")
