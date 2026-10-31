from csvrow import parse_row


def test_plain_fields():
    assert parse_row("a,b,c") == ["a", "b", "c"]


def test_empty_fields():
    assert parse_row("a,,c,") == ["a", "", "c", ""]


def test_quoted_field_with_comma():
    assert parse_row('a,"b,c",d') == ["a", "b,c", "d"]


def test_doubled_quote_is_one_literal_quote():
    assert parse_row('"say ""hi""",x') == ['say "hi"', "x"]


def test_quoted_empty_field():
    assert parse_row('"",a') == ["", "a"]


def test_spaces_are_kept():
    assert parse_row(" a , b ") == [" a ", " b "]


def test_comma_after_doubled_quote_that_ends_field():
    assert parse_row('"a""",b') == ['a"', "b"]


def test_doubled_quote_then_comma_inside_quotes():
    assert parse_row('"a"",b"') == ['a",b']


def test_field_that_is_only_an_escaped_quote():
    assert parse_row('""""') == ['"']


def test_doubled_quote_at_start_of_quoted_field():
    assert parse_row('"""a",b') == ['"a', "b"]


def test_quoted_empty_field_then_trailing_empty_field():
    assert parse_row('a,"",') == ["a", "", ""]
