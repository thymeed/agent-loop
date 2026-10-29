from ivl import merge


def test_overlapping_intervals_merge():
    assert merge([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]


def test_unsorted_input():
    assert merge([(8, 10), (1, 3), (2, 6)]) == [(1, 6), (8, 10)]


def test_touching_intervals_merge():
    assert merge([(1, 2), (2, 3)]) == [(1, 3)]


def test_nested_intervals():
    assert merge([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]


def test_empty_and_single():
    assert merge([]) == []
    assert merge([(5, 6)]) == [(5, 6)]


def test_input_is_not_modified():
    data = [(3, 4), (1, 2)]
    merge(data)
    assert data == [(3, 4), (1, 2)]
