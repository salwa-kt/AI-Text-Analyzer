
import json
from ollama import chat


def analyze_text(text):
    response = chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a text analysis assistant. "
                    "Analyze the user's text and return JSON "
                    "with exactly three fields: "
                    "sentiment (positive, negative, or neutral), "
                    "keywords (a list of important words), "
                    "and summary (one short sentence). "
                    "Do not include any extra text."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        format="json",
        options={"temperature": 0}
    )

    return json.loads(response.message.content)

def validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("AI result must be a dictionary.")

    required_fields = ["sentiment", "keywords", "summary"]

    for field in required_fields:
        if field not in result:
            raise ValueError(f"Missing field: {field}")

    if result["sentiment"] not in [
        "positive", "negative", "neutral"
    ]:
        raise ValueError("Invalid sentiment value.")

    if not isinstance(result["keywords"], list):
        raise ValueError("Keywords must be a list.")

    if not all(
        isinstance(word, str) for word in result["keywords"]
    ):
        raise ValueError("All keywords must be strings.")

    if not isinstance(result["summary"], str):
        raise ValueError("Summary must be a string.")

    return result


def main():
    print("=== AI Text Analyzer ===")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Enter text to analyze: ").strip()

        if text.lower() == "quit":
            print("Goodbye!")
            break

        if not text:
            print("Error: Text cannot be empty.\n")
            continue

        try:
            result = validate_result(analyze_text(text))

            print("\n=== Analysis Result ===")
            print("Sentiment:", result["sentiment"])
            print("Keywords:", ", ".join(result["keywords"]))
            print("Summary:", result["summary"])
            print()

        except (ValueError, KeyError, TypeError) as error:
            print("Invalid AI response:", error)

        except Exception as error:
            print("AI model error:", error)

if __name__ == "__main__":
    main()
