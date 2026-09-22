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


SINHALA_EXAMPLES = [
    ("කෘතිම බුද්ධිය කියන්නේ මොකක්ද?", "කෘතිම බුද්ධිය කියන්නේ මිනිසුන් කරන ඉගෙනීම, රටා හඳුනාගැනීම, භාෂාව තේරුම්ගැනීම සහ ගැටලු විසඳීම වැනි කාර්යයන් පරිගණක පද්ධති මගින් සිදු කිරීමට භාවිත කරන තාක්ෂණ සහ ක්‍රමවේද එකතුවකි."),
    ("Machine Learning කියන්නේ මොකක්ද?", "Machine Learning කියන්නේ දත්තවල තිබෙන රටා ඉගෙනගෙන, අලුත් input එකකට prediction එකක් හෝ decision එකක් ලබාදීමට model එකක් පුහුණු කරන ක්‍රමයකි."),
    ("Python කියන්නේ මොකක්ද?", "Python කියන්නේ ඉගෙනගන්න පහසු syntax එකක් සහ විශාල libraries ප්‍රමාණයක් ඇති programming language එකකි. Software, automation, data analysis සහ AI සඳහා එය බහුලව භාවිත වේ."),
    ("RAM එකෙන් කරන්නේ මොකක්ද?", "RAM කියන්නේ program එකක් ක්‍රියාත්මක වන විට තාවකාලිකව භාවිත කරන memory එකකි. වැඩි RAM එකක් තිබීමෙන් එකවර වැඩි applications හසුරුවන්න හැකියාව ලැබිය හැකිය."),
    ("CPU එක කියන්නේ මොකක්ද?", "CPU කියන්නේ computer එකේ instructions execute කිරීම, calculations කිරීම සහ software operations සම්බන්ධීකරණය කිරීම සඳහා භාවිත කරන ප්‍රධාන processor එකයි."),
    ("GPU එක කියන්නේ මොකක්ද?", "GPU කියන්නේ එකවර බොහෝ calculations parallel ලෙස සිදු කිරීමට නිර්මාණය කළ processor එකකි. Graphics සහ machine learning වැනි වැඩවලදී GPU බහුලව භාවිත වේ."),
    ("Algorithm එකක් කියන්නේ මොකක්ද?", "Algorithm එකක් කියන්නේ ගැටලුවක් විසඳීමට හෝ task එකක් සම්පූර්ණ කිරීමට අනුගමනය කරන පියවරෙන් පියවර ක්‍රමවේදයකි."),
    ("Database එකක් කියන්නේ මොකක්ද?", "Database එකක් කියන්නේ තොරතුරු ගබඩා කිරීමට, සෙවීමට, වෙනස් කිරීමට සහ කළමනාකරණය කිරීමට සංවිධානය කළ data collection එකකි."),
    ("Transformer model එකක් කියන්නේ මොකක්ද?", "Transformer කියන්නේ attention mechanisms භාවිත කර sequence එකක tokens අතර සම්බන්ධතා හඳුනාගන්නා neural-network architecture එකකි."),
    ("Chatbot එකක් කියන්නේ මොකක්ද?", "Chatbot එකක් කියන්නේ user ගෙන් ලැබෙන messages වලට text හෝ voice මගින් පිළිතුරු දෙන software system එකකි. ඒ සඳහා rules, retrieval හෝ AI models භාවිත කළ හැකිය."),
    ("RAG කියන්නේ මොකක්ද?", "RAG කියන්නේ userගේ ප්‍රශ්නයට අදාළ documents හෝ knowledge chunks මුලින් retrieve කර ඒ information එක answer එක සෑදීමට model එකට ලබාදෙන ක්‍රමයකි."),
    ("මට Python ඉගෙනගන්න කොහොමද?", "මුලින් variables, conditions, loops, functions සහ data structures වැනි basics ඉගෙනගන්න. ඊට පස්සේ කුඩා projects හදමින් debugging සහ documentation කියවීම practice කරන්න."),
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

    for question, answer in SINHALA_EXAMPLES:
        examples.append(
            {
                "instruction": question,
                "response": answer,
                "source": "original_own_ai_sinhala",
            }
        )

        examples.append(
            {
                "instruction": "සරලව පැහැදිලි කරන්න: " + question.rstrip("?."),
                "response": answer,
                "source": "original_own_ai_sinhala",
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
