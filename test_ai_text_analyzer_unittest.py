import unittest
from ai_text_analyzer import validate_result


class TestAITextAnalyzer(unittest.TestCase):

    def test_valid_response(self):
        result = {
            "sentiment": "positive",
            "keywords": ["Python", "AI"],
            "summary": "The user enjoys AI."
        }

        validated = validate_result(result)

        self.assertEqual(validated, result)

    def test_missing_field(self):
        result = {
            "sentiment": "positive",
            "keywords": ["Python"]
        }

        with self.assertRaises(ValueError):
            validate_result(result)

    def test_invalid_sentiment(self):
        result = {
            "sentiment": "happy",
            "keywords": ["Python"],
            "summary": "The user likes Python."
        }

        with self.assertRaises(ValueError):
            validate_result(result)

    def test_invalid_keywords(self):
        result = {
            "sentiment": "positive",
            "keywords": "Python, AI",
            "summary": "The user enjoys AI."
        }

        with self.assertRaises(ValueError):
            validate_result(result)


if __name__ == "__main__":
    unittest.main()