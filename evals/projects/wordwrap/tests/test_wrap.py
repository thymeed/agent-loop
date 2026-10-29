from wrapper import wrap


def test_short_text_is_one_line():
    assert wrap("hello world", 20) == ["hello world"]


def test_breaks_at_word_boundaries():
    assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]


def test_exact_fit_stays_on_one_line():
    assert wrap("abcde fghij", 11) == ["abcde fghij"]


def test_long_word_gets_its_own_line_unbroken():
    assert wrap("a supercalifragilistic b", 8) == ["a", "supercalifragilistic", "b"]


def test_runs_of_whitespace_collapse():
    assert wrap("a   b \n c", 20) == ["a b c"]


def test_empty_and_blank():
    assert wrap("", 10) == []
    assert wrap("   ", 10) == []
