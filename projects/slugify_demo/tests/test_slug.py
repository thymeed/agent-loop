from textutil import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_accents_are_folded():
    assert slugify("Café Münster") == "cafe-munster"


def test_punctuation_and_repeats_collapse_to_one_dash():
    assert slugify("Rock & Roll -- Forever!") == "rock-roll-forever"


def test_edges_are_trimmed():
    assert slugify("  --Hello--  ") == "hello"


def test_empty_and_symbols_only():
    assert slugify("") == ""
    assert slugify("!!!") == ""
