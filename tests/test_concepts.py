from csbf.concepts import self_verification_concept_code, text_concept_code, stable_concept_code


def test_stable_concept_code_still_works():
    code = stable_concept_code("hello world", buckets=16)
    assert 0 <= code < 16


def test_text_concept_code_detects_verification_pattern():
    assert text_concept_code("Let me check: 2+2=4, which is correct") == "verification"


def test_text_concept_code_detects_correction_pattern():
    assert text_concept_code("Wait, actually I made a mistake") == "correction"


def test_text_concept_code_returns_other_for_generic_text():
    assert text_concept_code("The number 42") == "other"


def test_text_concept_code_detects_conclusion():
    assert text_concept_code("Therefore the answer is 5") == "conclusion"


def test_text_concept_code_detects_setup():
    assert text_concept_code("Let us define x to be 3") == "setup"


def test_text_concept_code_detects_calculation():
    assert text_concept_code("We calculate 3 * 4 = 12") == "calculation"


def test_text_concept_code_detects_exploration():
    assert text_concept_code("Alternatively, consider another way") == "exploration"


def test_self_verification_concept_code_detects_verification_marker():
    assert self_verification_concept_code("Let me double-check the result to confirm it.") == "sv_verification"


def test_self_verification_concept_code_prefers_correction_marker():
    assert self_verification_concept_code("Wait, this is wrong; I need to fix the algebra.") == "sv_correction"


def test_self_verification_concept_code_detects_alternative_marker():
    assert self_verification_concept_code("Alternatively, solve it another way from the symmetry.") == "sv_alternative"


def test_self_verification_concept_code_returns_none_for_generic_step():
    assert self_verification_concept_code("The sum is 42.") == "sv_none"
