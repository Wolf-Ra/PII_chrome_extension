"""
tests.py — Unit tests for the sanitization pipeline (no PDF required).

Run:  python tests.py
"""

import sys, json, os, tempfile, unittest
sys.path.insert(0, os.path.dirname(__file__))

from models import WordRegistry, SensitiveEntity, Occurrence, WordEntry
from detector import _regex_tag, extract_field_value
from ocr import OCRToken


# ──────────────────────────────────────────────────────────────────────────────
# WordRegistry tests
# ──────────────────────────────────────────────────────────────────────────────

class TestWordRegistry(unittest.TestCase):

    def _tmp_registry(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)          # let registry create it fresh
        return path

    def test_register_and_retrieve(self):
        reg = WordRegistry(self._tmp_registry())
        reg.register_token("john", page=0, bbox=(10, 20, 60, 14), sensitive=False)
        entry = reg.get("john")
        self.assertIsNotNone(entry)
        self.assertEqual(len(entry.occurrences), 1)
        self.assertEqual(entry.occurrences[0].page, 0)

    def test_case_insensitive_key(self):
        reg = WordRegistry(self._tmp_registry())
        reg.register_token("John", page=0, bbox=(10, 20, 60, 14))
        self.assertIsNotNone(reg.get("john"))
        self.assertIsNotNone(reg.get("JOHN"))

    def test_multiple_occurrences(self):
        reg = WordRegistry(self._tmp_registry())
        reg.register_token("acme", page=0, bbox=(10, 20, 50, 14))
        reg.register_token("acme", page=1, bbox=(30, 40, 50, 14))
        entry = reg.get("acme")
        self.assertEqual(len(entry.occurrences), 2)

    def test_learn_field_marks_sensitive(self):
        reg = WordRegistry(self._tmp_registry())
        # Register the value before teaching the field
        reg.register_token("123456", page=0, bbox=(100, 50, 60, 14))
        reg.learn_field("123456", tag="ACCOUNT_NUMBER")
        entry = reg.get("123456")
        self.assertTrue(entry.is_learned_field)
        self.assertTrue(all(o.sensitive for o in entry.occurrences))

    def test_learn_field_persisted(self):
        path = self._tmp_registry()
        reg = WordRegistry(path)
        reg.learn_field("account_num", tag="ACCOUNT_NUMBER")

        # Reload from disk
        reg2 = WordRegistry(path)
        entry = reg2.get("account_num")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.is_learned_field)
        self.assertEqual(entry.default_tag, "ACCOUNT_NUMBER")
        os.remove(path)

    def test_new_token_auto_sensitive_if_learned_field(self):
        path = self._tmp_registry()
        reg = WordRegistry(path)
        reg.learn_field("9876543210", tag="ACCOUNT_NUMBER")
        # Save and reload — simulating next document
        reg2 = WordRegistry(path)
        entry = reg2.register_token("9876543210", page=2, bbox=(0, 0, 80, 14))
        self.assertTrue(entry.occurrences[-1].sensitive)
        os.remove(path)

    def test_ephemeral_redact_word_not_persisted(self):
        path = self._tmp_registry()
        reg = WordRegistry(path)
        reg.register_token("confidential", page=0, bbox=(5, 5, 80, 14))
        reg.redact_word_session("confidential")
        entry = reg.get("confidential")
        # Sensitive in session
        self.assertTrue(all(o.sensitive for o in entry.occurrences))
        # But not persisted (is_learned_field is False)
        self.assertFalse(entry.is_learned_field)
        if os.path.exists(path):
            os.remove(path)

    def test_all_sensitive_bboxes(self):
        reg = WordRegistry(self._tmp_registry())
        reg.register_token("foo", page=0, bbox=(0, 0, 10, 10), sensitive=True)
        reg.register_token("bar", page=0, bbox=(20, 0, 10, 10), sensitive=False)
        bboxes = reg.all_sensitive_bboxes()
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0][1], (0, 0, 10, 10))

    def test_reset_session_clears_occurrences(self):
        reg = WordRegistry(self._tmp_registry())
        reg.register_token("hello", page=0, bbox=(0, 0, 40, 14))
        self.assertEqual(len(reg.get("hello").occurrences), 1)
        reg.reset_session()
        entry = reg.get("hello")
        # Entry still exists but occurrences cleared
        self.assertEqual(len(entry.occurrences), 0)


# ──────────────────────────────────────────────────────────────────────────────
# Regex detection tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRegexDetector(unittest.TestCase):

    def test_email(self):
        self.assertEqual(_regex_tag("user@example.com"), "EMAIL")

    def test_phone_us(self):
        self.assertEqual(_regex_tag("555-867-5309"), "PHONE")

    def test_ssn(self):
        self.assertEqual(_regex_tag("123-45-6789"), "SSN")

    def test_credit_card(self):
        self.assertEqual(_regex_tag("4111-1111-1111-1111"), "CREDIT_CARD")

    def test_aadhaar(self):
        self.assertEqual(_regex_tag("1234 5678 9012"), "AADHAAR")

    def test_pan(self):
        self.assertEqual(_regex_tag("ABCDE1234F"), "PAN")

    def test_non_pii(self):
        self.assertIsNone(_regex_tag("hello"))
        self.assertIsNone(_regex_tag("document"))


# ──────────────────────────────────────────────────────────────────────────────
# Field value extraction tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFieldValueExtraction(unittest.TestCase):

    def _make_tokens(self, words: list[str]) -> list[OCRToken]:
        return [
            OCRToken(text=w, page=0, bbox=(i*50, 0, 45, 14), confidence=0.95)
            for i, w in enumerate(words)
        ]

    def test_simple_field_colon(self):
        tokens = self._make_tokens(["ACCOUNT", "NUMBER", ":", "9876543210", "other"])
        val = extract_field_value(tokens, "ACCOUNT NUMBER")
        self.assertIsNotNone(val)
        self.assertEqual(val.text, "9876543210")

    def test_field_without_colon(self):
        tokens = self._make_tokens(["Account", "Number", "9876543210"])
        val = extract_field_value(tokens, "account number")
        self.assertIsNotNone(val)
        self.assertEqual(val.text, "9876543210")

    def test_field_not_found(self):
        tokens = self._make_tokens(["Name", "John", "Doe"])
        val = extract_field_value(tokens, "ACCOUNT NUMBER")
        self.assertIsNone(val)


# ──────────────────────────────────────────────────────────────────────────────
# SensitiveEntity tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSensitiveEntity(unittest.TestCase):

    def test_to_dict(self):
        ent = SensitiveEntity(
            id=0, text="john@example.com", tag="EMAIL",
            source="regex", page=0, bbox=(10, 20, 80, 14),
            confidence=0.95, redact=True,
        )
        d = ent.to_dict()
        self.assertEqual(d["tag"], "EMAIL")
        self.assertEqual(d["bbox"], (10, 20, 80, 14))


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(
        sys.modules[__name__]
    ))
    sys.exit(0 if result.wasSuccessful() else 1)