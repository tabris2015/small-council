from small_council.runners import LocalCodeRunner

GOOD = "def add(a, b):\n    return a + b\n"
WRONG = "def add(a, b):\n    return a - b\n"
BROKEN = "def add(a, b)\n    return a + b\n"  # syntax error


def test_all_examples_pass():
    r = LocalCodeRunner().run_examples(GOOD, ["add(1, 2) == 3", "add(0, 0) == 0"])
    assert r.ok
    assert r.n == 2
    assert not r.failures


def test_failure_detected():
    r = LocalCodeRunner().run_examples(WRONG, ["add(1, 2) == 3"])
    assert not r.ok
    assert r.failures
    assert r.failures[0].expr == "add(1, 2) == 3"


def test_load_error_flagged():
    r = LocalCodeRunner().run_examples(BROKEN, ["add(1, 2) == 3"])
    assert not r.loaded
    assert not r.ok


def test_no_examples_is_ok():
    r = LocalCodeRunner().run_examples(GOOD, [])
    assert r.n == 0
    assert r.ok
