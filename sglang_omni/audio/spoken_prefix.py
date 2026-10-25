# SPDX-License-Identifier: Apache-2.0
"""How much of a known text a transcript of its speech says, as a strict prefix of that text."""

import re

ENGLISH_NUMBERS: dict[str, int] = dict(
    zip(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        range(21),
    )
)
CHINESE_DIGITS: dict[str, int] = dict(
    zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9))
)
CHINESE_MAGNITUDES: dict[str, int] = {"十": 10, "百": 100, "千": 1000}
TEXT_UNITS = re.compile(r"\d+|[零〇一二两三四五六七八九十百千]+|[a-zA-Z]+|[^\W_]")
TRAILING_PUNCTUATION = re.compile(r"[^\w\s]*")


def chinese_number(token: str) -> str:
    """The value of an ordinary Chinese number below ten thousand such as 三百零五; the token itself when it is not one."""
    number = digit = 0
    previous_magnitude = 10000
    after_digit = False
    is_number = True
    for char in token:
        if char in CHINESE_DIGITS:
            is_number &= not after_digit or digit == 0
            digit = CHINESE_DIGITS[char]
            after_digit = True
        else:
            magnitude = CHINESE_MAGNITUDES[char]
            is_number &= magnitude < previous_magnitude and (
                after_digit or (char == "十" and number == 0)
            )
            number += (digit or 1) * magnitude
            digit = 0
            previous_magnitude = magnitude
            after_digit = False
    if is_number:
        return str(number + digit)
    else:
        return token


def normalized_units(text: str) -> list[tuple[str, int]]:
    """The text's words, characters and numbers, case-folded and with numbers as Arabic digits, each with the character offset where it ends in `text`.

    Punctuation is not a unit. A run of Chinese digits such as 一二三 counts
    one by one, as does a count up to ten such as 一二三四五六七八九十.
    """
    units: list[tuple[str, int]] = []
    for match in TEXT_UNITS.finditer(text):
        token = match.group().casefold()
        if token.isdecimal():
            units.append((str(int(token)), match.end()))
        elif token in ENGLISH_NUMBERS:
            units.append((str(ENGLISH_NUMBERS[token]), match.end()))
        elif all(char in CHINESE_DIGITS for char in token):
            units.extend(
                (str(CHINESE_DIGITS[char]), match.start() + index + 1)
                for index, char in enumerate(token)
            )
        elif (
            token.endswith("十")
            and len(token) > 2
            and all(char in CHINESE_DIGITS for char in token[:-1])
            and [CHINESE_DIGITS[char] for char in token[:-1]]
            == list(range(CHINESE_DIGITS[token[0]], 10))
        ):
            units.extend(
                (str(CHINESE_DIGITS[char]), match.start() + index + 1)
                for index, char in enumerate(token[:-1])
            )
            units.append(("10", match.end()))
        elif all(
            char in CHINESE_DIGITS or char in CHINESE_MAGNITUDES for char in token
        ):
            units.append((chinese_number(token), match.end()))
        else:
            units.append((token, match.end()))
    return units


def unit_characters(text: str) -> list[tuple[str, int | None]]:
    """The characters of the text's units in order, each with the offset in `text` where its unit ends when it is the unit's last character, else None."""
    return [
        (char, end if index == len(unit) - 1 else None)
        for unit, end in normalized_units(text)
        for index, char in enumerate(unit)
    ]


def spoken_prefix_end(text: str, transcript: str) -> int:
    """The end of the longest prefix of `text` that `transcript` says, with the punctuation that follows it in `text`; 0 when the first units differ.

    Matching ignores case and punctuation and reads digits, English numbers
    up to twenty and ordinary Chinese numbers as the same number. Units may
    join or split, as long-term and longterm do, so the units are compared
    character by character and the prefix ends where a unit ends in both. It
    stops at the first omission, repetition or mismatch and never skips
    ahead.
    """
    end = 0
    for (char, unit_end), (heard, heard_end) in zip(
        unit_characters(text), unit_characters(transcript)
    ):
        if char != heard:
            break
        elif unit_end is not None and heard_end is not None:
            end = unit_end
        else:
            pass
    if end == 0:
        return 0
    else:
        return TRAILING_PUNCTUATION.match(text, end).end()
