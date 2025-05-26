import os
from typing import Any, Generator, TypeVar

import pytest
from get_unicode_emojis_list import (
    EMOJI_TESTFILE_FILENAME,
    get_all_emojis_from_latest_unicode_emojis_specification_with_download,
)
from pytest_benchmark.fixture import (
    BenchmarkFixture,
)

import d3lta.emojis_remover


@pytest.fixture(
    name="emojis_remover",
    params=[
        d3lta.emojis_remover.ExplicitUnicodeBlocksEmojisRemover,
        pytest.param(
            d3lta.emojis_remover.DemojiEmojisRemover,
            marks=pytest.mark.xfail(
                reason="`demoji`'s detection engine does not detect all emojis in the Unicode specification",
                strict=True,
            ),
        ),
    ],
)
def fixture_emojis_remover(
    request: pytest.FixtureRequest,
) -> d3lta.emojis_remover.EmojisRemover:
    return request.param()


T = TypeVar("T")
FixtureWithTeardown = Generator[T, Any, Any]


@pytest.fixture(name="latest_unicode_public_emojis", scope="session")
def fixture_latest_unicode_public_emojis() -> FixtureWithTeardown[list[str]]:
    """Latest list of emojis from the unicode consortium"""
    emojis = get_all_emojis_from_latest_unicode_emojis_specification_with_download()

    print(f"Retrieved {len(emojis)} unique emojis")

    yield emojis

    os.remove(EMOJI_TESTFILE_FILENAME)


ACCEPTABLE_ASCII_SYMBOLS = [
    "*",
    "#",
    "©",
    "®",
    "™",
    "‼",
    "⁇",
    "⁈",
    "⁉",
    "ℹ",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


def is_acceptable_ascii_symbol(text: str):
    return text in ACCEPTABLE_ASCII_SYMBOLS


def test_removes_all_emojis_in_latest_unicode_emojis_specification(
    latest_unicode_public_emojis: list[str],
    emojis_remover: d3lta.emojis_remover.EmojisRemover,
):
    for i, emoji in enumerate(latest_unicode_public_emojis):
        replacement = emojis_remover.remove_symbols(emoji)
        assert len(replacement) == 0 or is_acceptable_ascii_symbol(replacement), (
            f"Error at index {i}: {emoji} yields {replacement} ({replacement.encode('unicode-escape')})"
        )


@pytest.fixture(name="sample_text")
def fixture_sample_text() -> str:
    return """
The representatives of the French People, formed into a National Assembly, considering ignorance, forgetfulness or contempt of the rights of man to be the only causes of public misfortunes and the corruption of Governments, have resolved to set forth, in a solemn Declaration, the natural, unalienable and sacred rights of man, to the end that this Declaration, constantly present to all members of the body politic, may remind them unceasingly of their rights and their duties; to the end that the acts of the legislative power and those of the executive power, since they may be continually compared with the aim of every political institution, may thereby be the more respected; to the end that the demands of the citizens, founded henceforth on simple and incontestable principles, may always be directed toward the maintenance of the Constitution and the happiness of all.

In consequence whereof, the National Assembly recognises and declares, in the presence and under the auspices of the Supreme Being, the following Rights of Man and of the Citizen.
""".strip()


@pytest.fixture(name="sample_text_with_emojipasta")
def fixture_sample_text_with_emojipasta() -> str:
    return """
The representatives of the French 🥖🥐🍟  People, 🚷  formed 🈸  into a National 🏞️  Assembly, 🧑‍🏭  considering 🤔  ignorance, 🤷‍♀️🤷‍♂️  forgetfulness or contempt of the rights ↪️🧎‍➡️  of man 👳👨‍🔬👳👨‍🔬👳👨‍🔬  to be the only causes 🎗️  of public 🚋🚅📢  misfortunes and the corruption of Governments, have 🈶  resolved to set 📐  forth, in a solemn Declaration, the natural, unalienable and sacred ❤️‍🔥  rights 👉  of man, 👨‍👩‍👧‍👧👨‍❤️‍💋‍👨👩‍❤️‍👨🚶‍♂️‍➡️👨‍🦳👨‍👩‍👦‍👦🚣‍♂️👨‍🦽‍➡️👞🧛‍♂️  to the end 🔚  that this 🙂  Declaration, constantly present 🎁  to all members of the body 🖐️👀🤟🦷👁️🤚🖕👄👅🤲  politic, may remind them unceasingly of their rights 👩‍🦽‍➡️  and their duties; to the end 🔚  that the acts of the legislative power 🔋🔌  and those of the executive power, ✊  since they 👩‍👩‍👦‍👦  may be continually compared with the aim of every political institution, may thereby be the more ➕  respected; to the end 🔚  that the demands 🫴  of the citizens, founded henceforth on simple and incontestable principles, may always be directed 🎯  toward the maintenance of the Constitution and the happiness ☺️  of all.

In consequence whereof, the National 🏞️  Assembly 👩‍🏭👨‍🏭  recognises and declares, in the presence and under 🌁🌁🌁  the auspices of the Supreme Being, 🐝  the following Rights 👨‍🦼‍➡️  of Man 👨‍🔬  and of the Citizen.
""".strip()


def test_on_text_sample(
    emojis_remover: d3lta.emojis_remover.EmojisRemover,
    sample_text_with_emojipasta: str,
    sample_text: str,
    benchmark: BenchmarkFixture,
):
    processed = benchmark(
        emojis_remover.remove_symbols,
        sample_text_with_emojipasta,
    )
    assert processed == sample_text
