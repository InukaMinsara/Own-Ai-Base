from pathlib import Path
import json
import random
import re


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "processed" / "instructions.txt"
OUTPUT = ROOT / "data" / "processed" / "instructions_v7.jsonl"

SEED = 42


QUESTION_TEMPLATES = [
    "What is {topic}?",
    "Can you explain {topic}?",
    "Explain {topic} in simple terms.",
    "I am a beginner. What should I know about {topic}?",
    "Why is {topic} important?",
    "How does {topic} work?",
    "Give me a simple explanation of {topic}.",
    "Teach me the basics of {topic}.",
    "What does {topic} mean?",
    "Could you describe {topic} with an example?",
    "Help me understand {topic}.",
    "What are the main ideas behind {topic}?",
]


STYLE_TEMPLATES = [
    "{answer}",
    "Here is a concise explanation: {answer}",
    "For a beginner, think of it this way: {answer}",
    "A useful way to understand it is: {answer}",
    "In practical terms: {answer}",
]


def parse_source():
    text = SOURCE.read_text(
        encoding="utf-8"
    )

    records = []

    for chunk in text.split("<BOS>"):
        chunk = chunk.strip()

        if not chunk:
            continue

        chunk = chunk.split(
            "<EOS>",
            1,
        )[0].strip()

        match = re.search(
            r"User:\s*(.*?)\s*Assistant:\s*(.*)",
            chunk,
            re.S,
        )

        if not match:
            continue

        user = match.group(1).strip()
        answer = match.group(2).strip()

        if user and answer:
            records.append(
                (user, answer)
            )

    return records


def topic_from_question(question):
    q = question.strip().rstrip("?. ")

    prefixes = [
        "What is ",
        "Can you explain ",
        "Explain ",
        "How does ",
        "Why is ",
        "What does ",
        "Help me understand ",
    ]

    for prefix in prefixes:
        if q.lower().startswith(prefix.lower()):
            return q[len(prefix):].rstrip("?. ")

    return q


def main():
    random.seed(SEED)

    seeds = parse_source()

    examples = []
    seen = set()

    for question, answer in seeds:
        topic = topic_from_question(question)

        for q_template in QUESTION_TEMPLATES:
            for style in STYLE_TEMPLATES:
                q = q_template.format(
                    topic=topic
                )

                a = style.format(
                    answer=answer
                )

                key = (q, a)

                if key not in seen:
                    seen.add(key)
                    examples.append(
                        {
                            "instruction": q,
                            "response": a,
                            "source": "original_own_ai_seed",
                        }
                    )

    # Add varied arithmetic supervision.
    for a in range(1, 51):
        for b in range(1, 21):
            examples.append(
                {
                    "instruction": f"What is {a} + {b}?",
                    "response": f"{a} + {b} = {a + b}.",
                    "source": "generated_math",
                }
            )

            examples.append(
                {
                    "instruction": f"What is {a} * {b}?",
                    "response": f"{a} × {b} = {a * b}.",
                    "source": "generated_math",
                }
            )

    # Useful coding micro-instructions.
    code_examples = [
        (
            "How do I define a function in Python?",
            "Use the def keyword followed by the function name and parentheses. Add the function body on indented lines."
        ),
        (
            "How do I create a list in Python?",
            "Use square brackets with comma-separated values, for example: numbers = [1, 2, 3]."
        ),
        (
            "How do I loop through a list in Python?",
            "Use a for loop such as: for item in items: followed by an indented block that handles each item."
        ),
        (
            "How do I handle an exception in Python?",
            "Use a try block for code that may fail and an except block for the error you want to handle."
        ),
        (
            "What is a Git commit?",
            "A Git commit records a set of changes in the repository history with a message describing the change."
        ),
        (
            "What is JSON?",
            "JSON is a text format commonly used to represent structured data using objects, arrays, strings, numbers, booleans, and null."
        ),
    ]

    for question, answer in code_examples:
        examples.append(
            {
                "instruction": question,
                "response": answer,
                "source": "generated_coding",
            }
        )

    random.shuffle(examples)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in examples:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("=" * 50)
    print("OWN AI v7 INSTRUCTION DATASET")
    print("=" * 50)
    print("Seed pairs:", len(seeds))
    print("Generated examples:", len(examples))
    print("Output:", OUTPUT)
    print("=" * 50)


if __name__ == "__main__":
    main()
