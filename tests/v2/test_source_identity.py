from __future__ import annotations

import pytest

from vra.api.app import canonical_for_source
from vra.identity import source_identity
from vra.service.ingester import _canonical_id


@pytest.mark.parametrize('left,right', [
    ('https://youtube.com/watch?v=A', 'https://youtube.com/watch?v=B'),
    ('https://x/?a=1&b=2', 'https://x/?b=2&a=1'),
    ('https://x/?a=1&a=2', 'https://x/?a=2'),
    ('https://x/?v=a%2Fb', 'https://x/?v=a/b'),
    ('/Media/A.mp4', '/Media/a.mp4'),
])
def test_meaningful_source_distinctions_are_preserved(left, right):
    assert source_identity(left) != source_identity(right)


def test_feed_and_api_use_the_same_identity_independent_of_guid():
    source = 'https://youtube.com/watch?v=A'
    assert _canonical_id('https://unrelated-guid', source) == canonical_for_source(source)
    assert _canonical_id('yt:video:A', source) == canonical_for_source(' ' + source + ' ')
    assert source_identity(source).startswith('source:v2:')


def test_blank_source_rejected():
    with pytest.raises(ValueError):
        source_identity('   ')
