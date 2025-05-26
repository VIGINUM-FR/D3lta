import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

import demoji


@dataclass
class EmojisRemover(ABC):
    skip_postprocessing: bool = False

    @final
    def remove_symbols(self, text: str) -> str:
        text_without_symbols = self._remove_symbols_implementation(text)
        if self.skip_postprocessing:
            return text_without_symbols

        return self._postprocess(text_without_symbols)

    def _postprocess(self, text: str) -> str:
        # text = self._remove_whitespace_before_newline(text)
        text_without_repeated_whitespace = self._remove_repeated_whitespace(text)
        stripped_text_without_repeated_whitespace = (
            text_without_repeated_whitespace.strip()
        )
        return stripped_text_without_repeated_whitespace

    @abstractmethod
    def _remove_symbols_implementation(self, text: str) -> str: ...

    _whitespace_or_newline_capturing_group_name = "whitespace_or_newline"
    _repeated_whitespace_pattern = re.compile(
        rf"[ ]+(?P<{_whitespace_or_newline_capturing_group_name}> |\n)"
    )

    def _remove_repeated_whitespace(self, text: str) -> str:
        return re.sub(
            self._repeated_whitespace_pattern,
            rf"\g<{self._whitespace_or_newline_capturing_group_name}>",
            text,
        )


class ExplicitUnicodeBlocksEmojisRemover(EmojisRemover):
    # Unicode ranges for most emojis
    SYMBOLS_REGEX = re.compile(
        "["
        "\U000020d0-\U000020ff"  # Combining Diacritical Marks for Symbols
        "\U00002190-\U000021ff"  # Arrows
        "\U00002300-\U000023ff"  # Miscellaneous Technical
        "\U00002400-\U0000243f"  # Control Pictures
        "\U00002440-\U0000245f"  # Optical Character Recognition
        # WARNING: should we simply be transforming those enclosed characters to their plain, non-enclosed counterpart?
        "\U00002460-\U0000249f"  # Enclosed Alphanumerics
        # WARNING: should we simply be transforming those enclosed characters to their plain, non-enclosed counterpart?
        "\U000024b0-\U000024ff"  # Enclosed Alphanumerics Extension
        "\U00002500-\U0000257f"  # Box Drawing
        "\U00002580-\U000025ff"  # Block Elements
        "\U00002600-\U000026ff"  # Miscellaneous Symbols
        "\U00002700-\U000027bf"  # Dingbats
        "\U000027c0-\U000027ef"  # Miscellaneous Mathematical Symbols-A
        "\U000027f0-\U000027ff"  # Supplemental Arrows-A
        "\U00002800-\U000028ff"  # Braille Patterns
        "\U00002900-\U0000297f"  # Supplemental Arrows-B
        "\U00002980-\U000029ff"  # Miscellaneous Mathematical Symbols-B
        "\U00002a00-\U00002aff"  # Supplemental Mathematical Operators
        "\U00002b00-\U00002bff"  # Miscellaneous Symbols and Arrows
        "\U00003000-\U0000303f"  # CJK Symbols and Punctuation
        # WARNING: should we simply be transforming those enclosed characters to their plain, non-enclosed counterpart?
        "\U00003200-\U000032ff"  # Enclosed CJK Letters and Months
        "\U0001f000-\U0001f02f"  # Mahjong Tiles
        "\U0001f030-\U0001f09f"  # Domino Tiles
        "\U0001f0a0-\U0001f0ff"  # Playing cards
        # WARNING: should we simply be transforming those enclosed characters to their plain, non-enclosed counterpart?
        "\U0001f100-\U0001f1ff"  # Enclosed Alphanumeric Supplement
        # WARNING: should we simply be transforming those enclosed characters to their plain, non-enclosed counterpart?
        "\U0001f200-\U0001f2ff"  # Enclosed Ideographic Supplement
        "\U0001f300-\U0001f5ff"  # Miscellaneous Symbols and Pictographs
        "\U0001f600-\U0001f64f"  # Emoticons
        "\U0001f650-\U0001f67f"  # Ornamental Dingbats
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f700-\U0001f77f"  # alchemical symbols
        "\U0001f780-\U0001f7ff"  # Geometric Shapes
        "\U0001f800-\U0001f8ff"  # Supplemental Arrows-C
        "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
        "\U0001fa00-\U0001fa6f"  # Chess Symbols
        "\U0001fa70-\U0001faff"  # Symbols and Pictographs Extended-A
        "\U0001fb00-\U0001fbff"  # Symbols for Legacy Computing
        "\U000e0000-\U000e007f"  # Tags (used for modifying emojis with region modifiers in particular)
        "\U0000200d"  # Zero Width Joiner (ZWJ)
        "\U0000fe0f"  # Variation Selector-16 (emoji style)
        "\U0000fe0e"  # Variation Selector-15 (text style)
        "]+"
    )

    def _remove_symbols_implementation(self, text: str) -> str:
        return self.SYMBOLS_REGEX.sub(r"", text)


class DemojiEmojisRemover(EmojisRemover):
    def _remove_symbols_implementation(self, text: str) -> str:
        return demoji.replace(text)
