"""answer.py: a model's declination must not go out stamped "grounded".

Found by the L0-L6 verification sweep. Asked a question the corpus cannot
answer, the model correctly said the passages do not contain the information -
and then cited a passage anyway, because it is also told every claim needs a
citation. That citation resolved, grounding passed, and the refusal was
returned as `ok: true, "grounded in 1 cited passage(s)"`.

The empty-retrieval guard cannot catch this: RRF always returns its top k
however weak the match, and after OCR tripled this corpus some passage always
ranks. A non-answer wearing the badge of an answer is the exact failure this
codebase refuses everywhere else.
"""

from __future__ import annotations

import pytest

from kavach.knowledge.answer import _DECLINED


@pytest.mark.parametrize("text", [
    "The passages provided do not contain any information regarding the torque spec.",
    "I cannot provide an answer based on the given documents.",
    "The documents do not mention this equipment.",
    "The retrieved context does not specify a value for this.",
    "Unable to determine the setpoint from the provided passages.",
])
def test_a_declination_is_recognised(text):
    assert _DECLINED.search(text), f"declination not detected: {text!r}"


@pytest.mark.parametrize("text", [
    # The negation is about the SUBJECT, not about the retrieved material.
    "The vessel does not contain hydrogen at any stage [SOP_02 p.1].",
    "P-101A does not include a spare coupling [Technical_Procedure p.1].",
    "The trip setpoint is 0.8 barg [SOP_01 p.1 s.2].",
    "The pump does not run below minimum flow [SOP_01 p.1 s.3].",
])
def test_a_real_answer_containing_a_negation_is_not_a_declination(text):
    assert not _DECLINED.search(text), f"false positive on a real answer: {text!r}"
