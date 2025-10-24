from __future__ import annotations

import pytest

from app.utils.pdf import resolve_page_range


@pytest.mark.parametrize(
    "total_pages,start,end,max_pages,expected",
    [
        (10, 1, None, 5, range(0, 5)),
        (10, 3, 8, 10, range(2, 8)),
        (4, 2, 10, 3, range(1, 4)),
    ],
)
def test_resolve_page_range(total_pages, start, end, max_pages, expected):
    assert list(resolve_page_range(total_pages, start, end, max_pages)) == list(expected)
