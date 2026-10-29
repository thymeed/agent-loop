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
