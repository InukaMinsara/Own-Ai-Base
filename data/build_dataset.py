from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "processed" / "instructions.txt"

random.seed(42)

examples = [
    {
        "instruction": "What is artificial intelligence?",
        "response": "Artificial intelligence is a field of computer science that focuses on creating computer systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing patterns, learning from data, and solving problems."
    },
    {
        "instruction": "What is machine learning?",
        "response": "Machine learning is a branch of artificial intelligence where computer systems learn patterns from data and use those patterns to make predictions or decisions without being explicitly programmed for every individual case."
    },
    {
        "instruction": "What is Python?",
        "response": "Python is a high-level programming language known for its readable syntax and wide range of applications. It is commonly used for software development, automation, data analysis, artificial intelligence, and scientific computing."
    },
    {
        "instruction": "What is a computer?",
        "response": "A computer is an electronic machine that processes data according to instructions. It can receive input, perform calculations or other operations, store information, and produce output."
    },
    {
        "instruction": "What is RAM?",
        "response": "RAM, or random access memory, is temporary computer memory used to store data and programs that are actively being used. More available RAM can allow a computer to handle more tasks at the same time."
    },
    {
        "instruction": "What is a CPU?",
        "response": "A CPU, or central processing unit, is the main processor in a computer. It executes instructions, performs calculations, and coordinates many operations required by software."
    },
    {
        "instruction": "What is a GPU?",
        "response": "A GPU, or graphics processing unit, is a processor designed to perform many calculations in parallel. GPUs are widely used for graphics, scientific computing, machine learning, and artificial intelligence."
    },
    {
        "instruction": "What is the difference between RAM and storage?",
        "response": "RAM is temporary memory used while programs are running, while storage such as an SSD or hard drive keeps files and programs even when the computer is turned off."
    },
    {
        "instruction": "What is an algorithm?",
        "response": "An algorithm is a step-by-step procedure for solving a problem or completing a task. Algorithms are fundamental to programming because they describe how input should be transformed into a desired result."
    },
    {
        "instruction": "What is a database?",
        "response": "A database is an organized collection of information that can be stored, searched, updated, and managed efficiently. Databases are commonly used by applications to store user accounts, products, messages, and other structured data."
    },
    {
        "instruction": "What is a programming language?",
        "response": "A programming language is a formal language used to write instructions that a computer can execute. Examples include Python, JavaScript, C++, Java, and Rust."
    },
    {
        "instruction": "What is a neural network?",
        "response": "A neural network is a machine learning model made of connected computational units arranged in layers. During training, it adjusts numerical parameters to learn patterns from data."
    },
    {
        "instruction": "What is deep learning?",
        "response": "Deep learning is a type of machine learning that uses neural networks with multiple layers. These models can learn complex patterns from large amounts of data."
    },
    {
        "instruction": "What is natural language processing?",
        "response": "Natural language processing, or NLP, is an area of artificial intelligence focused on enabling computers to process and understand human language. Applications include translation, text classification, search, and conversational systems."
    },
    {
        "instruction": "What is a transformer model?",
        "response": "A transformer is a neural network architecture that uses attention mechanisms to process relationships between tokens in a sequence. Transformers are widely used in modern language models."
    },
    {
        "instruction": "What is tokenization in AI?",
        "response": "Tokenization is the process of converting text into smaller units called tokens. A language model processes these tokens as numerical identifiers rather than directly processing raw text."
    },
    {
        "instruction": "How does a language model generate text?",
        "response": "A language model generates text by predicting the next token based on the tokens that came before it. It repeatedly selects a next token until generation stops or reaches a configured limit."
    },
    {
        "instruction": "What is supervised learning?",
        "response": "Supervised learning is a machine learning approach where a model learns from examples that contain both inputs and expected outputs. The model uses these examples to learn a mapping that can be applied to new inputs."
    },
    {
        "instruction": "What is overfitting?",
        "response": "Overfitting happens when a machine learning model learns the training data too closely and performs poorly on new data. A model can have very low training error while still having high validation error."
    },
    {
        "instruction": "What is an operating system?",
        "response": "An operating system is system software that manages computer hardware and provides services for applications. Examples include Windows, Linux, macOS, Android, and iOS."
    },
    {
        "instruction": "What is an API?",
        "response": "An API, or application programming interface, is a defined way for software components to communicate with each other. An API specifies how requests and responses should be structured."
    },
    {
        "instruction": "What is Git?",
        "response": "Git is a distributed version control system used to track changes in files and collaborate on software projects. It allows developers to create commits, branches, and different versions of a project."
    },
    {
        "instruction": "What is GitHub?",
        "response": "GitHub is a platform for hosting and collaborating on software projects, especially Git repositories. It provides features such as code hosting, pull requests, issues, and project collaboration."
    },
    {
        "instruction": "What is HTML?",
        "response": "HTML, or HyperText Markup Language, is used to structure content on web pages. It defines elements such as headings, paragraphs, links, images, lists, and forms."
    },
    {
        "instruction": "What is CSS?",
        "response": "CSS, or Cascading Style Sheets, is used to control the appearance and layout of web pages. It can define colors, spacing, typography, positioning, responsive layouts, and animations."
    },
    {
        "instruction": "What is JavaScript?",
        "response": "JavaScript is a programming language widely used to create interactive web applications. It can run in web browsers and also on servers and other environments."
    },
    {
        "instruction": "Explain the scientific method.",
        "response": "The scientific method is a systematic approach to investigating questions. It commonly involves making observations, forming a hypothesis, performing experiments or collecting evidence, analyzing results, and drawing conclusions."
    },
    {
        "instruction": "What is energy?",
        "response": "Energy is the capacity to cause change or perform work. It exists in different forms, including kinetic, potential, thermal, chemical, electrical, and radiant energy."
    },
    {
        "instruction": "What is gravity?",
        "response": "Gravity is an interaction that causes objects with mass to attract one another. Near Earth's surface, gravity causes unsupported objects to accelerate toward the ground."
    },
    {
        "instruction": "What is photosynthesis?",
        "response": "Photosynthesis is a process used by plants and some other organisms to convert light energy into chemical energy. Plants use carbon dioxide and water to produce glucose and release oxygen."
    },
    {
        "instruction": "What is the water cycle?",
        "response": "The water cycle describes the continuous movement of water through Earth's atmosphere, land, oceans, and living systems. Important processes include evaporation, condensation, precipitation, and collection."
    },
    {
        "instruction": "What is mathematics?",
        "response": "Mathematics is a field concerned with numbers, quantities, structures, patterns, relationships, and logical reasoning. It provides tools for describing and solving problems in science, engineering, computing, and everyday life."
    },
    {
        "instruction": "What is a variable in programming?",
        "response": "A variable is a named reference used by a program to store or access a value. The value associated with a variable can often change while the program is running."
    },
    {
        "instruction": "What is a function in programming?",
        "response": "A function is a reusable section of code designed to perform a particular task. Functions can accept inputs called arguments and may return a result."
    },
    {
        "instruction": "How can I learn programming?",
        "response": "A practical way to learn programming is to start with one beginner-friendly language, learn its basic syntax and concepts, and build small projects. Gradually increase project complexity while practicing debugging and reading documentation."
    },
    {
        "instruction": "How should I debug a Python program?",
        "response": "Start by reproducing the problem and reading the error message carefully. Check the line mentioned in the traceback, inspect variable values, simplify the failing section, and test the program again after each change."
    },
    {
        "instruction": "Why is testing important in software development?",
        "response": "Testing helps developers detect incorrect behavior before software reaches users. It can verify that individual components and complete workflows behave as expected."
    },
    {
        "instruction": "What is cloud computing?",
        "response": "Cloud computing provides computing resources such as storage, servers, databases, and software through network-based services. Users can access these resources without necessarily owning or managing the underlying physical infrastructure."
    },
    {
        "instruction": "What is cybersecurity?",
        "response": "Cybersecurity is the practice of protecting computers, networks, applications, and data from unauthorized access, disruption, misuse, or damage."
    },
    {
        "instruction": "What is encryption?",
        "response": "Encryption transforms readable data into a protected form using a cryptographic algorithm and key. Authorized systems can use the appropriate key to recover the original information."
    },
    {
        "instruction": "Hello",
        "response": "Hello! I am Own AI. How can I help you?"
    },
    {
        "instruction": "Hi",
        "response": "Hi! I am Own AI. What would you like to learn or build?"
    },
    {
        "instruction": "Who are you?",
        "response": "I am Own AI, a small language model trained to understand and generate text from the knowledge and examples provided during training."
    },
    {
        "instruction": "What can you do?",
        "response": "I can answer questions, explain technical concepts, help with programming ideas, and generate text based on patterns learned during training. My abilities depend on my model architecture and training data."
    }
]


def format_example(item):
    return (
        "<BOS>\n"
        "User: " + item["instruction"] + "\n"
        "Assistant: " + item["response"] + "\n"
        "<EOS>\n\n"
    )


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    random.shuffle(examples)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in examples:
            f.write(format_example(item))

    print("=" * 32)
    print("   OWN AI INSTRUCTION DATA")
    print("=" * 32)
    print("Examples:", len(examples))
    print("Output:", OUTPUT)
    print("=" * 32)


if __name__ == "__main__":
    main()