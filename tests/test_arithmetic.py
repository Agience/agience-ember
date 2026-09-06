"""Arithmetic domain + the content-context-operator triple. Pure, deterministic, no store.

Imports resolve through `ember.runtime.runner`, which re-exports the distributed bundle from
`prism.runner` — the single code path serve, worker and router also run."""
from ember.runtime.runner import arithmetic as ar
from ember.runtime.runner import category as cat


# ---- arithmetic domain (computed + self-verifying) --------------------------
def test_expression_and_precedence():
    a = ar.compute("2 + 3 * 4")
    assert a.grounded and a.text.endswith("= 14")


def test_nl_phrasing():
    assert ar.compute("what is 7 times 8").text.endswith("= 56")
    assert ar.compute("2 to the power of 10").text.endswith("= 1024")


def test_named_ops_verified():
    assert "not prime" in ar.compute("is 51 prime").text
    assert "prime" in ar.compute("is 97 prime").text and "not" not in ar.compute("is 97 prime").text
    assert ar.compute("factor 60").read["check"].endswith("== 60")
    assert ar.compute("5 factorial").text == "5! = 120"
    assert ar.compute("gcd of 48 and 36").text.endswith("= 12")


def test_every_answer_carries_a_check():
    for q in ["7 * 8", "is 13 prime", "sqrt of 81", "6 factorial"]:
        a = ar.compute(q)
        assert a is not None and a.read["verified"] and a.read["check"]


def test_non_arithmetic_returns_none():
    assert ar.compute("define artifact") is None
    assert ar.compute("how do I frobnicate a widget") is None


def test_safe_no_eval():
    # a code-injection attempt is not a valid arithmetic expression, so there is nothing to compute
    assert ar.compute("__import__('os').system('echo x')") is None


# ---- category / triple (any 2 predict the 3rd) ------------------------------
def test_composed_operators():
    assert cat.apply("subtract", 10, 3) == 7
    assert cat.apply("divide", 12, 4) == 3
    assert cat.apply("power", 2, 10) == 1024


def test_triple_apply_and_preimage():
    assert cat.complete_triple(content=[7, 8], operator="multiply")["context"] == 56
    assert cat.complete_triple(operator="multiply", context=56, known=[7, None], missing_index=1)["content"] == 8
    assert cat.complete_triple(operator="add", context=10, known=[None, 3], missing_index=0)["content"] == 7


def test_infer_operator_from_examples():
    assert cat.infer([([7, 8], 56), ([3, 4], 12), ([2, 5], 10)]) == "multiply"
    assert cat.infer([([10, 3], 7), ([5, 2], 3)]) == "subtract"
    assert cat.infer([([2, 10], 1024), ([3, 2], 9)]) == "power"
    assert cat.infer([([7, 8], 15), ([3, 4], 7)]) == "add"


def test_infer_refuses_when_no_fit():
    assert cat.infer([([7, 8], 99)]) is None


# ---- learnable path: infer the operator from example pairs in a query -------
def test_learn_operator_from_query():
    assert "multiply" in ar.learn_operator("infer the operator: (7,8)->56 and (3,4)->12").text
    assert "subtract" in ar.learn_operator("what operator maps 10,3 to 7 and 5,2 to 3").text
    assert "power" in ar.learn_operator("learn the rule: 2,10 -> 1024; 3,2 -> 9").text


def test_learn_refuses_no_fit():
    a = ar.learn_operator("infer operator from 7,8 -> 99")
    assert a is not None and not a.grounded


def test_learn_ignores_non_learn_queries():
    assert ar.learn_operator("what is 7 * 8") is None      # a compute query, not a learn query


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("ok", fn.__name__)
    print("ALL ARITHMETIC/CATEGORY TESTS PASS"); sys.exit(0)
