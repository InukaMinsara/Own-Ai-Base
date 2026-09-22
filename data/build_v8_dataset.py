from pathlib import Path
import hashlib
import json
import os
import random
import re


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "sft_v8.jsonl"
)
CORPUS = (
    ROOT
    / "data"
    / "processed"
    / "corpus_v8.txt"
)

SEED = 2026
TARGET_EXAMPLES = 20000

ENGLISH_SEEDS = [
    (
        "What is artificial intelligence?",
        "Artificial intelligence is the field of building computer systems that perform tasks that normally require capabilities such as pattern recognition, language processing, planning, or decision making."
    ),
    (
        "What is machine learning?",
        "Machine learning trains a model on examples so it can learn useful patterns and make predictions or decisions on new inputs."
    ),
    (
        "What is a neural network?",
        "A neural network is a model made from layers of connected mathematical operations. During training, its parameters are adjusted so its outputs become more useful for the task."
    ),
    (
        "What is a transformer?",
        "A transformer is a neural-network architecture built around attention. Attention lets the model relate information from different positions in a sequence."
    ),
    (
        "What is RAG?",
        "Retrieval-augmented generation retrieves relevant information from a knowledge source and supplies that context to a language model before it generates an answer."
    ),
    (
        "What is a tokenizer?",
        "A tokenizer converts text into a sequence of tokens that a language model can process. Tokens may represent words, pieces of words, characters, or bytes."
    ),
    (
        "What is a database?",
        "A database is an organized system for storing, retrieving, updating, and managing information."
    ),
    (
        "What is an API?",
        "An API is an interface that lets one software component request data or actions from another software component using defined rules."
    ),
    (
        "What is Git?",
        "Git is a version-control system that records changes to files so software projects can be reviewed, restored, branched, and collaborated on."
    ),
    (
        "What is Python?",
        "Python is a general-purpose programming language known for readable syntax and a large ecosystem. It is widely used for automation, software, data work, and AI."
    ),
    (
        "What is HTML?",
        "HTML is the markup language used to structure content on web pages. Elements describe things such as headings, paragraphs, links, images, forms, and sections."
    ),
    (
        "What is CSS?",
        "CSS is the stylesheet language used to control how web content is presented, including layout, spacing, typography, and responsive behavior."
    ),
    (
        "What is JavaScript?",
        "JavaScript is a programming language commonly used to add behavior and interactivity to web pages and applications."
    ),
    (
        "What is memory in a chatbot?",
        "Chatbot memory is stored conversation or user information that can be supplied again later so the system can maintain useful continuity."
    ),
    (
        "What is overfitting?",
        "Overfitting happens when a model learns the training examples too specifically and performs worse on data it has not seen."
    ),
    (
        "What is validation loss?",
        "Validation loss measures model error on held-out examples that are not used for parameter updates. It is useful for detecting overfitting."
    ),
    (
        "What is gradient descent?",
        "Gradient descent updates model parameters in a direction that reduces the training objective, using gradients to estimate how each parameter affects the loss."
    ),
    (
        "What is an epoch?",
        "An epoch is one complete pass through a training dataset."
    ),
    (
        "What is a GPU?",
        "A GPU is a processor designed for highly parallel numerical work. That makes it useful for graphics and many machine-learning workloads."
    ),
    (
        "What is a file upload pipeline?",
        "A file upload pipeline accepts a file, validates it, stores it safely, extracts useful content when possible, and makes that content available to the application."
    ),
]

SINHALA_SEEDS = [
    (
        "කෘතිම බුද්ධිය කියන්නේ මොකක්ද?",
        "කෘතිම බුද්ධිය කියන්නේ රටා හඳුනාගැනීම, භාෂාව සැකසීම, සැලසුම් කිරීම සහ ගැටලු විසඳීම වැනි කාර්යයන් පරිගණක පද්ධති මගින් සිදු කිරීමට භාවිත කරන ක්ෂේත්‍රයකි."
    ),
    (
        "Machine Learning කියන්නේ මොකක්ද?",
        "Machine Learning කියන්නේ උදාහරණ දත්ත භාවිතයෙන් model එකකට රටා ඉගෙනගෙන අලුත් input සඳහා prediction හෝ decision ලබාදීමට පුහුණු කරන ක්‍රමයකි."
    ),
    (
        "Python කියන්නේ මොකක්ද?",
        "Python කියන්නේ පහසු syntax එකක් සහ විශාල libraries එකතුවක් ඇති programming language එකකි. Automation, software, data සහ AI වැඩ සඳහා එය බහුලව භාවිත වේ."
    ),
    (
        "RAG කියන්නේ මොකක්ද?",
        "RAG කියන්නේ ප්‍රශ්නයට අදාළ documents හෝ knowledge chunks මුලින් retrieve කර ඒ context එක language model එකට ලබාදී grounded answer එකක් සෑදීමට භාවිත කරන ක්‍රමයකි."
    ),
    (
        "RAM එකෙන් කරන්නේ මොකක්ද?",
        "RAM කියන්නේ programs ක්‍රියාත්මක වන විට අවශ්‍ය data සහ instructions තාවකාලිකව තබාගන්නා memory එකකි."
    ),
    (
        "GPU එක කියන්නේ මොකක්ද?",
        "GPU කියන්නේ එකවර calculations විශාල ප්‍රමාණයක් parallel ලෙස සිදු කිරීමට නිර්මාණය කළ processor එකකි. Graphics සහ machine learning වැඩ සඳහා එය ප්‍රයෝජනවත් වේ."
    ),
    (
        "මට Python ඉගෙනගන්න කොහොමද?",
        "මුලින් variables, conditions, loops, functions සහ lists වැනි basics ඉගෙනගන්න. ඉන්පසු කුඩා projects හදමින් errors කියවීම සහ debugging practice කරන්න."
    ),
    (
        "Chatbot එකක් කියන්නේ මොකක්ද?",
        "Chatbot එකක් කියන්නේ user messages වලට software logic, retrieval හෝ AI model එකක් භාවිතයෙන් පිළිතුරු ලබාදෙන application එකකි."
    ),
    (
        "Database එකක් කියන්නේ මොකක්ද?",
        "Database එකක් කියන්නේ තොරතුරු ගබඩා කිරීමට, සෙවීමට, වෙනස් කිරීමට සහ කළමනාකරණය කිරීමට සංවිධානය කළ data system එකකි."
    ),
    (
        "Algorithm එකක් කියන්නේ මොකක්ද?",
        "Algorithm එකක් කියන්නේ ගැටලුවක් විසඳීමට හෝ task එකක් සම්පූර්ණ කිරීමට අනුගමනය කරන නිශ්චිත පියවර මාලාවකි."
    ),
]

CODING_SEEDS = [
    (
        "How do I define a function in Python?",
        "Use the def keyword, give the function a name, add parentheses for parameters, and place the function body on indented lines."
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
        "Put code that may fail inside try and handle the expected error inside except."
    ),
    (
        "How do I read a text file in Python?",
        "Use open with a text mode such as: with open('file.txt', 'r', encoding='utf-8') as f: text = f.read()."
    ),
    (
        "How do I create an HTML button?",
        'Use a button element such as <button type="button">Click me</button> and add JavaScript only when the button needs behavior.'
    ),
    (
        "How do I center an element with CSS?",
        "For a simple flex layout, set the parent to display: flex and use justify-content: center and align-items: center."
    ),
    (
        "How do I fetch JSON in JavaScript?",
        "Use fetch(url), await the response, then call response.json() to parse the JSON body."
    ),
    (
        "What is a Git commit?",
        "A Git commit records a set of changes in repository history together with a message describing those changes."
    ),
    (
        "What is JSON?",
        "JSON is a text format for structured data using objects, arrays, strings, numbers, booleans, and null."
    ),
]

STYLE_VARIANTS = [
    "Answer directly: {answer}",
    "Explain it simply: {answer}",
    "Give a beginner-friendly explanation: {answer}",
    "Be concise and practical: {answer}",
    "Explain the key idea first: {answer}",
    "Use simple language: {answer}",
]

QUESTION_VARIANTS = [
    "{q}",
    "Can you explain {q}",
    "Please explain {q}",
    "I am a beginner. {q}",
    "What should I know about this? {q}",
    "Help me understand this: {q}",
]

SINHALA_QUESTION_VARIANTS = [
    "{q}",
    "සරලව පැහැදිලි කරන්න: {q}",
    "මට තේරෙන විදිහට කියන්න: {q}",
    "ආරම්භකයෙකුට කියනවා වගේ පැහැදිලි කරන්න: {q}",
]

MATH_PATTERNS = [
    ("What is {a} + {b}?", "{a} + {b} = {v}."),
    ("Calculate {a} - {b}.", "{a} - {b} = {v}."),
    ("What is {a} * {b}?", "{a} × {b} = {v}."),
    ("Calculate {a} / {b}.", "{a} / {b} = {v}."),
    ("What is {a} squared?", "{a}² = {v}."),
]

WEB_KNOWLEDGE = [
    (
        "How should a local AI use web search?",
        "A local AI can use web search as an explicit tool for current information, retrieve relevant sources, and clearly separate fetched facts from the model's own prior knowledge."
    ),
    (
        "Why should a chatbot use retrieval?",
        "Retrieval gives a small model access to relevant external context without requiring every fact to be stored in model weights."
    ),
]


def clean(text):
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def uid(task, question, response):
    raw = f"{task}|{question}|{response}".encode(
        "utf-8"
    )
    return hashlib.sha1(raw).hexdigest()[:16]


def add(records, seen, task, question, response, group_id):
    question = clean(question)
    response = clean(response)
    if not question or not response:
        return
    key = (question, response)
    if key in seen:
        return
    seen.add(key)
    records.append(
        {
            "id": uid(task, question, response),
            "task": task,
            "group_id": str(group_id),
            "instruction": question,
            "response": response,
            "source": "own_ai_v8_seed",
        }
    )


def load_old():
    path = ROOT / "data" / "processed" / "instructions_v7.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = clean(item.get("instruction", ""))
            a = clean(item.get("response", ""))
            if q and a:
                rows.append((q, a, item.get("source", "v7")))
    return rows


def build_records():
    records = []
    seen = set()

    # Preserve previous examples but place them into explicit task groups.
    for idx, (q, a, source) in enumerate(load_old()):
        add(
            records,
            seen,
            "general",
            q,
            a,
            f"legacy-{idx}",
        )

    for idx, (q, a) in enumerate(ENGLISH_SEEDS):
        for qv in QUESTION_VARIANTS:
            for sv in STYLE_VARIANTS:
                add(
                    records,
                    seen,
                    "explanation",
                    qv.format(q=q),
                    sv.format(answer=a),
                    f"eng-{idx}",
                )

    for idx, (q, a) in enumerate(CODING_SEEDS):
        for qv in QUESTION_VARIANTS:
            for sv in STYLE_VARIANTS[:4]:
                add(
                    records,
                    seen,
                    "coding",
                    qv.format(q=q),
                    sv.format(answer=a),
                    f"code-{idx}",
                )

    for idx, (q, a) in enumerate(SINHALA_SEEDS):
        for qv in SINHALA_QUESTION_VARIANTS:
            for sv in STYLE_VARIANTS:
                add(
                    records,
                    seen,
                    "sinhala",
                    qv.format(q=q),
                    sv.format(answer=a),
                    f"si-{idx}",
                )

    for idx, (q, a) in enumerate(WEB_KNOWLEDGE):
        for qv in QUESTION_VARIANTS:
            add(
                records,
                seen,
                "tool_use",
                qv.format(q=q),
                a,
                f"web-{idx}",
            )

    # Deterministic arithmetic supervision gives the model diverse surface
    # forms while the runtime calculator remains the exact source of truth.
    for a in range(1, 201):
        for b in range(1, 51):
            rows = [
                (
                    f"What is {a} + {b}?",
                    f"{a} + {b} = {a + b}.",
                ),
                (
                    f"What is {a} * {b}?",
                    f"{a} × {b} = {a * b}.",
                ),
            ]

            if a >= b:
                rows.append(
                    (
                        f"Calculate {a} - {b}.",
                        f"{a} - {b} = {a - b}.",
                    )
                )

            if b != 0 and a % b == 0:
                rows.append(
                    (
                        f"Calculate {a} / {b}.",
                        f"{a} / {b} = {a // b}.",
                    )
                )

            for q, answer in rows:
                add(
                    records,
                    seen,
                    "math",
                    q,
                    answer,
                    f"math-{a}-{b}",
                )

    # Expand with lightweight conversational behaviors rather than only facts.
    conversation = [
        (
            "hi",
            "Hi! I am Own AI. Tell me what you are trying to learn or build."
        ),
        (
            "hello",
            "Hello! I can help with programming, math, explanations, local files, and other tasks supported by my tools."
        ),
        (
            "what can you do?",
            "I can answer questions from my trained knowledge, use local retrieval, calculate exact arithmetic, work with supported files, and use explicit web search."
        ),
        (
            "I don't understand.",
            "No problem. I can explain the same idea more simply and step by step."
        ),
        (
            "Can you answer in Sinhala?",
            "ඔව්. මට සිංහලෙන් පිළිතුරු දෙන්න පුළුවන්. අවශ්‍ය නම් සරල සිංහලෙන් පැහැදිලි කරන්නත් පුළුවන්."
        ),
    ]

    for idx, (q, a) in enumerate(conversation):
        for qv in (q, "Please respond to this: " + q):
            add(
                records,
                seen,
                "conversation",
                qv,
                a,
                f"conv-{idx}",
            )

    # Keep the training distribution balanced. Pure random truncation would
    # let the large synthetic math pool crowd out coding, Sinhala, and chat.
    quotas = {
        "general": 2500,
        "explanation": 3000,
        "coding": 3500,
        "sinhala": 3500,
        "tool_use": 2000,
        "math": 4500,
        "conversation": 1000,
    }

    rng = random.Random(SEED)
    by_task = {}

    for item in records:
        by_task.setdefault(
            item["task"],
            [],
        ).append(item)

    selected = []
    selected_ids = set()

    for task, quota in quotas.items():
        pool = list(by_task.get(task, []))
        rng.shuffle(pool)

        for item in pool[:quota]:
            if item["id"] not in selected_ids:
                selected.append(item)
                selected_ids.add(item["id"])

    # Fill any remaining capacity from unused examples.
    if len(selected) < TARGET_EXAMPLES:
        remainder = [
            item
            for item in records
            if item["id"] not in selected_ids
        ]
        rng.shuffle(remainder)
        selected.extend(
            remainder[: TARGET_EXAMPLES - len(selected)]
        )

    rng.shuffle(selected)
    return selected[:TARGET_EXAMPLES]


def write_corpus(records):
    knowledge_dirs = [
        ROOT / "data" / "raw",
        ROOT / "data" / "knowledge",
        ROOT / "data" / "uploads",
    ]

    blocks = []
    seen = set()

    for directory in knowledge_dirs:
        if not directory.exists():
            continue

        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {
                ".txt",
                ".md",
                ".csv",
                ".json",
                ".py",
                ".js",
                ".ts",
                ".html",
                ".css",
            }:
                continue

            try:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                continue

            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text and text not in seen:
                seen.add(text)
                blocks.append(
                    f"[SOURCE: {path.relative_to(ROOT)}]\n{text}"
                )

    # Add clean, task-oriented training text without duplicating all answer
    # style variants. The SFT file carries the full supervision.
    for item in records:
        block = (
            f"Question: {item['instruction']}\n"
            f"Answer: {item['response']}"
        )
        blocks.append(block)

    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_text(
        "\n\n".join(blocks),
        encoding="utf-8",
    )


def main():
    records = build_records()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    if os.getenv(
        "OWN_AI_V8_BUILD_LEGACY_CORPUS",
        "0",
    ).lower() in {"1", "true", "yes"}:
        write_corpus(records)

    counts = {}
    groups = set()
    for item in records:
        counts[item["task"]] = counts.get(
            item["task"],
            0,
        ) + 1
        groups.add(item["group_id"])

    print("=" * 64)
    print("OWN AI v8 DATASET BUILDER")
    print("=" * 64)
    print("Examples:", len(records))
    print("Groups:", len(groups))
    print("Tasks:", counts)
    print("SFT:", OUTPUT)
    print("Corpus:", CORPUS)
    print("=" * 64)


if __name__ == "__main__":
    main()
