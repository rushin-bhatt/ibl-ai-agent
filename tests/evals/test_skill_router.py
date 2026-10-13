"""Unit tests for the packet -> skill file router. No API calls."""

import pytest

from tests.evals.skill_router import PACKETS, get_skill_files


@pytest.mark.parametrize("packet", sorted(PACKETS))
def test_every_packet_file_exists(packet):
    # A renamed skill file would silently drop context from every question's prompt.
    missing = [str(path) for path in PACKETS[packet] if not path.exists()]
    assert not missing


def test_files_are_deduplicated_in_declaration_order():
    files = get_skill_files("data_loading", "data_loading", "anatomy")
    assert files == PACKETS["data_loading"] + PACKETS["anatomy"]


def test_unknown_packet_raises():
    with pytest.raises(KeyError):
        get_skill_files("not-a-packet")
