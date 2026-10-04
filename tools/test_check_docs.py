"""Tests for tools/check_docs.py."""

import tempfile
import unittest
from pathlib import Path

import check_docs

ROOT = Path(__file__).resolve().parents[1]


class LocalLinks(unittest.TestCase):
    def test_plain_angles_titles_and_ignored_examples(self):
        text = """[A](a.md "title") [B](<a b.md>)
```markdown
[example](not-real.md)
```
`[inline](not-real.md)`
[C](https://example.org/a)
"""
        self.assertEqual(
            check_docs.links(text), ["a.md", "a b.md", "https://example.org/a"]
        )

    def test_missing_and_escape_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ok.md").write_text("ok")
            text = "[ok](ok.md#part) [bad](missing.md) [escape](../outside.md)"
            errors = check_docs.path_errors(root, root / "README.md", text)
            self.assertEqual(len(errors), 2)
            self.assertIn("missing target", errors[0])
            self.assertIn("escapes repository", errors[1])

    def test_encoded_spaces_and_unknown_scheme(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a b.md").write_text("x")
            text = "[ok](a%20b.md) [bad](file:///etc/passwd)"
            errors = check_docs.path_errors(root, root / "README.md", text)
            self.assertEqual(len(errors), 1)
            self.assertIn("nonportable", errors[0])

    def test_repository_guides(self):
        count, errors = check_docs.check(ROOT)
        self.assertGreaterEqual(count, 5)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
