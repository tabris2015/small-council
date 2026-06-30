"""Deterministic, offline tests of the council loop using Pydantic-AI FunctionModels.

Each role gets a FunctionModel that returns scripted text, so the whole plan->code->verify->retry
control flow is exercised with no network and no real model. The LocalCodeRunner runs real (tiny)
candidate code against the worked examples.
"""

from __future__ import annotations

import json

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from small_council.models import CouncilTask
from small_council.orchestrator import Council

ADD_PROMPT = "Implement add(a, b) that returns the sum.\n\n```python\nassert add(1, 2) == 3\n```"
GOOD_CODE = "```python\ndef add(a, b):\n    return a + b\n```"
WRONG_CODE = "```python\ndef add(a, b):\n    return a - b\n```"
APPROVE = json.dumps({"decision": "APPROVE", "reason": ""})
REVISE = json.dumps({"decision": "REVISE", "reason": "handle negatives"})


def scripted(*responses: str) -> FunctionModel:
    """A FunctionModel that returns each response in turn, repeating the last when exhausted."""
    box = {"i": 0}

    def reply(messages, info):  # noqa: ANN001 - pydantic-ai callback signature
        i = box["i"]
        box["i"] = i + 1
        content = responses[i] if i < len(responses) else responses[-1]
        return ModelResponse(parts=[TextPart(content=content)])

    return FunctionModel(reply)


def make_council(*, planner="plan", coder=GOOD_CODE, verifier=APPROVE, max_retries=2) -> Council:
    return Council(
        planner_model=scripted(planner),
        coder_model=coder if isinstance(coder, FunctionModel) else scripted(coder),
        verifier_model=verifier if isinstance(verifier, FunctionModel) else scripted(verifier),
        max_retries=max_retries,
    )


def test_happy_path_extracts_examples_and_approves():
    council = make_council()
    # examples=None -> extracted from the prompt's fenced assert
    result = council.solve(CouncilTask(prompt=ADD_PROMPT, entrypoint="add"))
    assert result.extraction_ok
    assert "def add" in result.code
    assert result.approved
    assert result.attempts == 1
    assert result.usage.requests >= 2  # planner + coder + verifier calls counted


def test_example_failure_triggers_retry_then_fix():
    council = make_council(coder=scripted(WRONG_CODE, GOOD_CODE), verifier=APPROVE)
    result = council.solve(
        CouncilTask(prompt=ADD_PROMPT, entrypoint="add", examples=["add(1, 2) == 3"])
    )
    assert result.attempts == 2
    assert result.approved
    assert "a + b" in result.code


def test_verifier_revise_triggers_retry():
    council = make_council(coder=GOOD_CODE, verifier=scripted(REVISE, APPROVE))
    result = council.solve(
        CouncilTask(prompt=ADD_PROMPT, entrypoint="add", examples=["add(1, 2) == 3"])
    )
    assert result.attempts == 2
    assert result.approved


def test_extraction_failure_triggers_retry():
    council = make_council(coder=scripted("sorry, no code here", GOOD_CODE), verifier=APPROVE)
    result = council.solve(
        CouncilTask(prompt=ADD_PROMPT, entrypoint="add", examples=["add(1, 2) == 3"])
    )
    assert result.attempts == 2
    assert result.extraction_ok
    assert result.approved


def test_out_of_retries_returns_unapproved_candidate():
    # Coder always wrong; examples never pass -> verifier never reached -> unapproved.
    council = make_council(coder=scripted(WRONG_CODE), verifier=APPROVE, max_retries=1)
    result = council.solve(
        CouncilTask(prompt=ADD_PROMPT, entrypoint="add", examples=["add(1, 2) == 3"])
    )
    assert result.attempts == 2
    assert not result.approved
    assert result.extraction_ok  # the wrong code still parses
    assert "a - b" in result.code


def test_no_examples_goes_straight_to_verifier():
    council = make_council(coder=GOOD_CODE, verifier=APPROVE)
    result = council.solve(
        CouncilTask(prompt="Implement add(a, b).", entrypoint="add", examples=[])
    )
    assert result.approved
    assert result.attempts == 1
