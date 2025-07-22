# taken from https://gist.github.com/msenol86/44082269be46aa446ccda9d02202e523
import os
import re
import urllib.request

EMOJI_TESTFILE_FILENAME = "emoji-test.txt"
EMOJI_DATA_URL = "https://unicode.org/Public/emoji/latest/emoji-test.txt"


def download_latest_emoji_test_data() -> None:
    with urllib.request.urlopen(EMOJI_DATA_URL) as emoji_data_request_response:
        emoji_test_file = emoji_data_request_response.read()

    with open(EMOJI_TESTFILE_FILENAME, "wb") as tmp_file:
        tmp_file.write(emoji_test_file)


def get_all_emojis_from_latest_unicode_emojis_specification_with_download() -> list[
    str
]:
    if not os.path.exists(EMOJI_TESTFILE_FILENAME):
        print(EMOJI_TESTFILE_FILENAME + " file not found. Downloading it ...")
        download_latest_emoji_test_data()

    emoji_matching_in_unicode_specification_v16_0_pattern = re.compile(
        r"(?:minimally|fully)-qualified[ ]*# (?P<emoji>.*?) "
    )

    with open(EMOJI_TESTFILE_FILENAME, "r", encoding="utf8") as unicode_data:
        unicode_data_rows = unicode_data.read()

    def _deduplicate(items: list[str]):
        return list(set(items))

    emojis = _deduplicate(
        emoji_matching_in_unicode_specification_v16_0_pattern.findall(unicode_data_rows)
    )

    return emojis


if __name__ == "__main__":
    print(get_all_emojis_from_latest_unicode_emojis_specification_with_download())
