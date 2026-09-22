from pathlib import Path
import shutil
import re

# ============================================================
# OWN AI DATASET EXPANDER V2
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
BACKUP_DIR = BASE_DIR / "data" / "backup_before_expand_v2"

RAW_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "artificial_intelligence.txt",
    "basic_math.txt",
    "computer_basics.txt",
    "computer_science_basics.txt",
    "core_mathematics.txt",
    "general_knowledge.txt",
    "physics_chemistry_biology.txt",
    "programming_intermediate.txt",
    "python_and_algorithms.txt",
    "python_basics.txt",
    "qa_examples.txt",
    "science_basics.txt",
    "simple_conversations.txt",
    "technology_basics.txt",
    "train.txt",
    "web_and_software.txt",
]

print("=" * 60)
print("       OWN AI DATASET EXPANDER V2")
print("=" * 60)

# ============================================================
# Topic lists
# ============================================================

AI_TOPICS = [
    "machine learning",
    "neural networks",
    "deep learning",
    "transformers",
    "attention mechanisms",
    "language models",
    "tokenization",
    "embeddings",
    "model training",
    "model inference",
    "optimization",
    "overfitting",
    "underfitting",
    "datasets",
    "model evaluation",
    "classification",
    "regression",
    "clustering",
    "computer vision",
    "natural language processing",
]

PROGRAMMING_TOPICS = [
    "variables",
    "data types",
    "functions",
    "conditions",
    "loops",
    "lists",
    "tuples",
    "dictionaries",
    "sets",
    "classes",
    "objects",
    "modules",
    "exceptions",
    "file handling",
    "APIs",
    "databases",
    "testing",
    "debugging",
    "algorithms",
    "data structures",
]

SCIENCE_TOPICS = [
    "motion",
    "force",
    "energy",
    "electricity",
    "magnetism",
    "waves",
    "light",
    "atoms",
    "molecules",
    "chemical reactions",
    "acids and bases",
    "cells",
    "DNA",
    "genes",
    "ecosystems",
    "evolution",
    "photosynthesis",
    "cellular respiration",
    "gravity",
    "thermodynamics",
]

MATH_TOPICS = [
    "integers",
    "fractions",
    "percentages",
    "ratios",
    "linear equations",
    "inequalities",
    "functions",
    "polynomials",
    "quadratic equations",
    "geometry",
    "probability",
    "statistics",
    "vectors",
    "matrices",
    "limits",
    "derivatives",
    "integrals",
    "sequences",
    "coordinate geometry",
    "mathematical logic",
]

TECH_TOPICS = [
    "operating systems",
    "computer hardware",
    "CPUs",
    "GPUs",
    "RAM",
    "storage",
    "solid state drives",
    "computer networks",
    "DNS",
    "HTTP",
    "HTTPS",
    "web browsers",
    "servers",
    "cloud computing",
    "virtualization",
    "containers",
    "databases",
    "software development",
    "web applications",
    "computer security fundamentals",
]

# ============================================================
# Content generators
# ============================================================

def ai_content(topic):
    return f"""
Artificial Intelligence Topic: {topic}

{topic.capitalize()} is an important concept in artificial intelligence.
A beginner should first understand the definition and then study how the
concept is used in practical systems.

A useful way to analyze {topic} is to identify the input, the processing
operation, and the expected output. Different AI systems can implement
similar ideas using different algorithms or architectures.

In machine learning, a model learns parameters from training data. During
training, the model produces predictions and a loss function measures the
difference between predictions and target values. An optimization algorithm
then updates model parameters.

When studying {topic}, it is important to consider data quality, computational
requirements, model assumptions, evaluation methods, and generalization.

A useful learning process is:

1. Learn the definition.
2. Study a simple example.
3. Understand the main components.
4. Implement a small experiment.
5. Measure the result.
6. Compare the result with expectations.
7. Study limitations.

A common misunderstanding is that an AI model automatically understands the
world like a human. A trained model instead learns statistical patterns from
its training process and produces outputs according to its learned parameters.

Example reasoning:

Suppose a learner encounters a new AI problem involving {topic}. The learner
should identify what information is available, what output is required, which
representation is appropriate, and how the result can be evaluated.

Learning {topic} together with related concepts usually gives a stronger
understanding than memorizing an isolated definition.
"""


def programming_content(topic):
    return f"""
Programming Topic: {topic}

{topic.capitalize()} is a programming concept that can be applied to many
software projects. Although programming languages use different syntax,
the underlying computational idea can often be explained independently.

When working with {topic}, first define the expected input and expected
output. Then divide the problem into smaller operations.

A practical development process is:

1. Understand the requirement.
2. Identify the input.
3. Identify the output.
4. Break the problem into smaller operations.
5. Select suitable data structures.
6. Implement a clear solution.
7. Test normal cases.
8. Test unusual cases.
9. Handle expected errors.
10. Review the implementation.

Debugging should be systematic. If software involving {topic} behaves
incorrectly, inspect the input, intermediate values, and final output.
Changing many parts of the program at once can make debugging more difficult.

Readable code normally uses meaningful names, predictable control flow,
small reusable functions, and appropriate error handling.

A beginner can learn {topic} effectively by writing a small program and then
modifying it to observe how different inputs affect the result.

Good programming is not only about making code execute. It is also about
making the code understandable, testable, maintainable, and reliable.
"""


def science_content(topic):
    return f"""
Science Topic: {topic}

{topic.capitalize()} can be studied using observations, measurements,
experiments, models, and evidence.

A scientific explanation should distinguish an observation from an
interpretation. Measurements provide evidence, while models provide ways
to represent relationships between observations.

A useful investigation can follow these steps:

1. Define the question.
2. Identify relevant variables.
3. Determine how measurements will be collected.
4. Form a testable hypothesis when appropriate.
5. Collect observations.
6. Analyze the evidence.
7. Compare the evidence with the hypothesis.
8. Describe limitations.

When studying {topic}, it is useful to ask what quantities are involved,
how those quantities interact, and what evidence could support or contradict
a proposed explanation.

Scientific measurements can contain uncertainty. Experimental results can
also be affected by limitations in equipment, sampling, assumptions, or
experimental design.

A scientific model is useful when it explains observations and makes useful
predictions. Models can be improved when new evidence reveals limitations.

Learning {topic} becomes easier when the learner connects the topic with
related concepts instead of memorizing isolated facts.
"""


def math_content(topic):
    return f"""
Mathematics Topic: {topic}

{topic.capitalize()} can be understood through definitions, symbolic
representations, examples, calculations, and logical reasoning.

A useful mathematical problem-solving process is:

1. Read the problem carefully.
2. Identify known quantities.
3. Identify unknown quantities.
4. Choose a suitable representation.
5. Select the mathematical relationship.
6. Perform the calculation.
7. Check the result.
8. Explain what the result means.

When solving a problem involving {topic}, writing the relationship before
performing calculations can reduce mistakes.

A strong mathematical solution should contain enough reasoning for another
person to understand how the answer was obtained.

For example, if a problem contains an unknown value, assign it a symbol.
Translate the information into an equation or another mathematical
representation, solve the representation, and then verify the result using
the original conditions.

A common mistake is performing a correct calculation with an incorrect
interpretation of the problem.

Understanding why a mathematical method works is generally more useful than
memorizing only the final formula.
"""


def technology_content(topic):
    return f"""
Technology Topic: {topic}

{topic.capitalize()} is part of modern computing and digital systems.
Understanding it requires knowing what it does, what it depends on, and how
it interacts with other components.

A technology system can often be understood through several layers:

Hardware provides physical resources.
Operating systems manage those resources.
Applications provide user-facing functionality.
Networks allow systems to communicate.
Databases and storage preserve information.

When troubleshooting {topic}, start with simple checks:

1. Confirm the system is running.
2. Check connections.
3. Check configuration.
4. Read error messages.
5. Inspect logs when available.
6. Test the smallest component.
7. Change one variable at a time.
8. Verify the result.

Security is an important consideration. Systems should use appropriate
authentication, authorization, validation, secure communication, and careful
credential management.

Performance should normally be measured before optimization. A slowdown
can originate from CPU usage, memory pressure, storage latency, network
conditions, software configuration, or another bottleneck.

Learning {topic} as part of a larger system makes it easier to understand
real-world technology problems.
"""


# ============================================================
# Question / Answer generator
# ============================================================

def qa_content(topic, category):
    return f"""
Question: What is {topic}?

Answer: {topic.capitalize()} is a concept related to {category}. It can be
understood by studying its definition, purpose, examples, and limitations.

Question: Why is {topic} important?

Answer: It provides a useful way to represent, process, understand, or solve
problems within its field.

Question: How can a beginner learn {topic}?

Answer: A beginner can start with the definition, study a simple example,
practice applying the concept, and then compare it with related ideas.

Question: What is a common mistake?

Answer: A common mistake is memorizing terminology without understanding how
the concept behaves in a real example.

Question: How can understanding be tested?

Answer: The learner can explain the concept in their own words, solve a new
example, identify incorrect applications, and describe the limitations.
"""


# ============================================================
# Backup existing files
# ============================================================

print("\nCreating backups...")

for filename in FILES:
    source = RAW_DIR / filename

    if source.exists():
        destination = BACKUP_DIR / filename

        if not destination.exists():
            shutil.copy2(source, destination)

print("Backup complete.")

# ============================================================
# File -> content mapping
# ============================================================

MAPPING = {
    "artificial_intelligence.txt": (
        AI_TOPICS,
        ai_content,
        "Artificial Intelligence"
    ),

    "basic_math.txt": (
        MATH_TOPICS,
        math_content,
        "Mathematics"
    ),

    "computer_basics.txt": (
        TECH_TOPICS,
        technology_content,
        "Technology"
    ),

    "computer_science_basics.txt": (
        PROGRAMMING_TOPICS,
        programming_content,
        "Computer Science"
    ),

    "core_mathematics.txt": (
        MATH_TOPICS,
        math_content,
        "Mathematics"
    ),

    "general_knowledge.txt": (
        SCIENCE_TOPICS + TECH_TOPICS,
        science_content,
        "General Knowledge"
    ),

    "physics_chemistry_biology.txt": (
        SCIENCE_TOPICS,
        science_content,
        "Science"
    ),

    "programming_intermediate.txt": (
        PROGRAMMING_TOPICS,
        programming_content,
        "Programming"
    ),

    "python_and_algorithms.txt": (
        PROGRAMMING_TOPICS,
        programming_content,
        "Programming"
    ),

    "python_basics.txt": (
        PROGRAMMING_TOPICS,
        programming_content,
        "Python Programming"
    ),

    "qa_examples.txt": (
        AI_TOPICS + MATH_TOPICS + PROGRAMMING_TOPICS,
        qa_content,
        "General Knowledge"
    ),

    "science_basics.txt": (
        SCIENCE_TOPICS,
        science_content,
        "Science"
    ),

    "simple_conversations.txt": (
        AI_TOPICS + PROGRAMMING_TOPICS,
        qa_content,
        "General Knowledge"
    ),

    "technology_basics.txt": (
        TECH_TOPICS,
        technology_content,
        "Technology"
    ),

    "train.txt": (
        AI_TOPICS + MATH_TOPICS + SCIENCE_TOPICS,
        ai_content,
        "General Knowledge"
    ),

    "web_and_software.txt": (
        TECH_TOPICS + PROGRAMMING_TOPICS,
        technology_content,
        "Technology"
    ),
}

# ============================================================
# Expand files
# ============================================================

print("\nGenerating additional training material...\n")

total_added = 0

for filename, info in MAPPING.items():

    topics = info[0]
    generator = info[1]
    category = info[2]

    path = RAW_DIR / filename

    if not path.exists():
        print("[SKIP]", filename)
        continue

    existing = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Prevent accidental second execution.
    if "OWN AI DATASET EXPANSION V2" in existing:
        print("[SKIP] Already expanded:", filename)
        continue

    parts = []

    parts.append(
        "\n\n"
        + "=" * 70
        + "\nOWN AI DATASET EXPANSION V2\n"
        + "=" * 70
        + "\n"
    )

    for topic in topics:

        parts.append(generator(topic))

        parts.append(
            qa_content(topic, category)
        )

        parts.append(
            f"""
Practice Task:

Study the concept of {topic}.

Explain the concept in your own words.
Give one simple example.
Describe one related concept.
Describe one common misunderstanding.
Explain how you would check whether your understanding is correct.

Reasoning Task:

Imagine that you are given a new problem involving {topic}. Before trying
to solve it, identify the information that is known, identify what must be
found, select an appropriate method, perform the required reasoning, and
check the final result.

Review Task:

Explain why {topic} is useful and describe a situation where a different
approach might be more appropriate.
"""
        )

    generated = "\n\n".join(parts)

    with path.open("a", encoding="utf-8") as file:
        file.write(generated)

    total_added += len(generated)

    print(
        f"[OK] {filename:<40} "
        f"+{len(generated):,} characters"
    )

# ============================================================
# Measure result
# ============================================================

print("\n" + "=" * 60)
print("DATASET MEASUREMENT")
print("=" * 60)

all_text = []

for filename in FILES:

    path = RAW_DIR / filename

    if path.exists():
        all_text.append(
            path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        )

combined = "\n\n".join(all_text)

characters = len(combined)
words = len(re.findall(r"\S+", combined))

# This is only an estimate.
estimated_tokens = int(words * 1.35)

print(f"Files:              {len(all_text)}")
print(f"Characters:         {characters:,}")
print(f"Words:              {words:,}")
print(f"Estimated tokens:   {estimated_tokens:,}")
print(f"Added characters:   {total_added:,}")

print("\n100K target is based on the REAL tokenizer,")
print("so this estimate is only a rough indicator.")

print("\nBackup:")
print(BACKUP_DIR)

print("=" * 60)
print("EXPANSION FINISHED")
print("=" * 60)