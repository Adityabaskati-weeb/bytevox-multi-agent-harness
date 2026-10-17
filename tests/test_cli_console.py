"""Console encoding regressions for the command-line trace viewer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bytevox_harness.cli import _replay


class StrictEncodedStdout:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self.data = bytearray()

    def write(self, value: str) -> int:
        self.data.extend(value.encode(self.encoding, errors="strict"))
        return len(value)


class ReplayConsoleEncodingTests(unittest.TestCase):
    def test_replay_handles_text_not_supported_by_console_encoding(self) -> None:
        records = [
            {
                "sequence": 1,
                "elapsed_ms": 1,
                "event": "untrusted_content_detected",
                "detail": "The non\u2011randomized report contains an injection attempt.",
            },
            {
                "sequence": 2,
                "elapsed_ms": 2,
                "event": "checker_result",
                "accepted": False,
                "llm_review": {"accepted": False, "issues": ["A non\u2011randomized claim is unsupported."]},
                "deterministic_validation": {"passed": True, "issues": []},
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.jsonl"
            trace.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

            for show_all in (False, True):
                with self.subTest(show_all=show_all):
                    stdout = StrictEncodedStdout("cp1252")
                    with patch("sys.stdout", stdout):
                        _replay(trace, show_all=show_all)

                    rendered = bytes(stdout.data).decode("cp1252")
                    self.assertIn("Replay: trace.jsonl", rendered)
                    self.assertIn("non?randomized", rendered)
                    if not show_all:
                        self.assertIn("checker_result llm_review=FAIL", rendered)
                        self.assertIn("combined=REJECT", rendered)


if __name__ == "__main__":
    unittest.main()
