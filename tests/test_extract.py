from small_council.extract import extract_code


def test_prefers_python_fence_defining_entrypoint():
    text = "Here you go:\n```python\ndef foo(x):\n    return x + 1\n```\nDone."
    code, ok = extract_code(text, entrypoint="foo")
    assert ok
    assert "def foo" in code
    assert "Here you go" not in code


def test_strips_think_block():
    text = "<think>let me reason about this</think>\n```python\ndef f():\n    return 1\n```"
    code, ok = extract_code(text, entrypoint="f")
    assert ok
    assert "reason" not in code


def test_picks_block_defining_entrypoint_over_longer_other():
    text = (
        "```python\nimport math  # a long but irrelevant helper block\nx = math.pi\n```\n"
        "```python\ndef target():\n    return 42\n```"
    )
    code, ok = extract_code(text, entrypoint="target")
    assert ok
    assert "def target" in code


def test_no_code_returns_not_ok():
    code, ok = extract_code("I'm sorry, I can't help with that.", entrypoint="foo")
    assert not ok
