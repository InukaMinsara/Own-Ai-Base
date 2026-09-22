from pathlib import Path
import re
import shutil
from datetime import datetime

# ============================================================
# OWN AI — 100K+ DATASET EXPANDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
BACKUP_DIR = BASE_DIR / "data" / "backup_before_expand"

TARGET_TOKENS = 100_000

print("=" * 60)
print("          OWN AI DATASET EXPANDER")
print("=" * 60)

RAW_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Existing files -> category
# ------------------------------------------------------------

FILE_CATEGORIES = {
    "artificial_intelligence.txt": "Artificial Intelligence",
    "basic_math.txt": "Basic Mathematics",
    "computer_basics.txt": "Computer Basics",
    "computer_science_basics.txt": "Computer Science",
    "core_mathematics.txt": "Core Mathematics",
    "general_knowledge.txt": "General Knowledge",
    "physics_chemistry_biology.txt": "Physics Chemistry Biology",
    "programming_intermediate.txt": "Programming",
    "python_and_algorithms.txt": "Python Algorithms",
    "python_basics.txt": "Python Basics",
    "qa_examples.txt": "Question Answer",
    "science_basics.txt": "Science",
    "simple_conversations.txt": "Conversations",
    "technology_basics.txt": "Technology",
    "train.txt": "General Training",
    "web_and_software.txt": "Web Software",
}

# ------------------------------------------------------------
# Original educational knowledge
# ------------------------------------------------------------

CONTENT = {

"Artificial Intelligence": """
Artificial intelligence is the study of computer systems that perform tasks
that normally require human reasoning, perception, language understanding,
learning, or decision making. An AI system receives information, processes
that information, and produces an output according to its model or rules.

Machine learning is a major area of artificial intelligence. Instead of
writing every rule manually, a machine learning system can learn patterns
from examples. Training data contains examples from which model parameters
are adjusted.

Supervised learning uses examples that contain inputs and expected outputs.
Classification predicts a category, while regression predicts a numerical
value. Unsupervised learning works with data where expected labels are not
provided. Clustering is one example of unsupervised learning.

A neural network contains layers of mathematical operations. A basic neural
network can transform an input vector through weighted connections and
activation functions. During training, a loss function measures the difference
between a prediction and the desired result. An optimizer changes parameters
to reduce the loss.

Deep learning uses neural networks with many layers. Convolutional networks
are commonly associated with image processing. Recurrent architectures were
designed for sequential data. Transformer architectures use attention
mechanisms to process relationships between tokens.

A language model predicts tokens based on context. Tokenization converts
text into smaller units that a model can process. During training, a causal
language model learns to predict the next token from previous tokens.

Attention allows a model to assign different importance to different parts
of an input sequence. Self-attention compares representations within the
same sequence. Multi-head attention performs several attention operations
in parallel.

A transformer normally contains attention layers, feed-forward layers,
normalization, residual connections, token embeddings, and positional
information. Training requires numerical data, a loss function, an optimizer,
and many parameter updates.

Model quality depends on many factors including data quality, model size,
training procedure, tokenization, context length, and evaluation. Increasing
the amount of data alone does not guarantee a better model.

Overfitting happens when a model learns the training examples too closely
and performs poorly on unseen examples. Validation data helps estimate how
well a model generalizes.

AI systems should be evaluated with representative tests. A useful evaluation
set should contain questions and tasks that were not simply copied into the
training data. Accuracy is useful for some tasks, while other tasks require
different evaluation measures.

A small AI model can still be useful for education, experimentation,
classification, text generation, summarization, and specialized tasks.
Building a small model is also a useful way to understand how modern AI
systems work internally.
""",

"Basic Mathematics": """
Mathematics provides formal methods for describing quantities, relationships,
patterns, and structures. Arithmetic is the foundation of many mathematical
operations.

Addition combines quantities. Subtraction finds a difference. Multiplication
can represent repeated addition. Division distributes a quantity into equal
parts or determines how many times one number fits into another.

An integer can be positive, negative, or zero. A rational number can be
written as a ratio of integers where the denominator is not zero. Real numbers
include rational and irrational numbers.

A fraction has a numerator and denominator. Equivalent fractions represent
the same value. To add fractions with different denominators, a common
denominator can be used.

A percentage is a value expressed relative to one hundred. To convert a
percentage to a decimal, divide by one hundred. To calculate a percentage
of a quantity, multiply the quantity by the percentage written as a decimal.

An equation states that two expressions have equal values. Solving an
equation means finding values of the unknown that make the equality true.

For a linear equation such as 3x + 5 = 20, subtracting 5 from both sides
gives 3x = 15, and dividing by 3 gives x = 5.

An exponent represents repeated multiplication. For example, a squared
quantity is multiplied by itself. The exponent rules provide efficient ways
to simplify expressions.

A square root is a number that produces a given non-negative value when
multiplied by itself. The square root of 25 is 5 because 5 times 5 equals 25.

A ratio compares quantities. Proportions state that two ratios are equal.
Ratios and proportions are useful in scale drawings, rates, mixtures, and
many practical calculations.

Geometry studies shapes, sizes, angles, distances, and spatial relationships.
A triangle has three sides. The angles inside a triangle add to 180 degrees.

The area of a rectangle is length multiplied by width. The area of a triangle
is one half multiplied by its base and height. The circumference of a circle
is related to its radius through the constant pi.

Probability measures how likely an event is. A probability of zero means an
event cannot occur, while a probability of one means the event is certain
within the model being considered.

Statistics provides methods for collecting, organizing, analyzing, and
interpreting data. Mean, median, and mode are common measures used to
describe data.
""",

"Computer Basics": """
A computer is an electronic system that processes information according to
instructions. Most modern computers contain a processor, memory, storage,
input devices, output devices, and communication hardware.

The central processing unit executes instructions. CPU performance depends
on architecture, clock frequency, number of cores, cache, and workload.
A higher clock frequency does not automatically mean that one CPU is faster
for every task.

Random access memory provides temporary working space for programs and data.
RAM is generally faster than permanent storage but loses its contents when
the computer is powered off.

Storage devices keep data persistently. Hard disk drives use magnetic storage,
while solid state drives use flash memory. SSDs generally provide much lower
access latency than mechanical hard drives.

An operating system manages hardware and provides services for applications.
Windows, Linux, and macOS are examples of desktop operating systems.

A file contains data organized according to a format. A filename extension
often indicates the expected file type, although the extension itself does
not determine the actual contents.

Folders provide a hierarchical organization for files. A path identifies
where a file or folder is located in a filesystem.

Processes are running instances of programs. The operating system allocates
CPU time, memory, and other resources to processes.

Device drivers allow an operating system to communicate with hardware.
Graphics processing units are specialized processors designed to perform
large numbers of parallel operations efficiently.

Computer networks allow systems to communicate. A local area network
connects devices within a limited area, while the internet connects networks
around the world.

An IP address identifies a network interface within an addressing system.
Domain names provide human-readable names that can resolve to network
addresses.

A web browser requests resources from web servers and displays web pages.
HTTP and HTTPS are protocols commonly used for web communication.

Software updates can fix bugs, improve compatibility, and address security
problems. Keeping important software updated is one part of normal computer
maintenance.
""",

"Computer Science": """
Computer science studies computation, algorithms, data, software, hardware,
and information systems. It is broader than learning a particular programming
language.

An algorithm is a finite sequence of steps for solving a problem. A good
algorithm should have clearly defined inputs, operations, and expected
outputs.

Data structures organize information so that algorithms can process it
efficiently. Arrays provide indexed storage. Linked structures connect
elements through references. Stacks use last-in-first-out behavior, while
queues use first-in-first-out behavior.

A tree is a hierarchical data structure. A graph consists of vertices and
edges and can represent networks, relationships, or connections.

Searching algorithms find information in a data structure. Linear search
checks elements sequentially. Binary search repeatedly divides a sorted
search space into smaller sections.

Sorting algorithms arrange data according to an ordering rule. Different
sorting algorithms have different time and memory characteristics.

Algorithm complexity describes how resource requirements change as input
size grows. Big O notation is commonly used to describe an upper-bound style
of growth rate.

A compiler translates source code into another representation such as
machine code or intermediate code. An interpreter executes instructions
through a runtime system.

Operating systems provide process management, memory management, file
systems, device access, and security mechanisms.

Databases organize structured information and provide methods for storing,
querying, updating, and deleting records. Relational databases use tables,
rows, and columns.

A primary key identifies records within a table. A foreign key can represent
a relationship between tables.

Software engineering applies systematic methods to designing, implementing,
testing, maintaining, and documenting software.

Version control records changes to source code. Git is a distributed version
control system commonly used for collaborative development.

Testing checks whether software behaves as expected. Unit tests focus on
small components, while integration tests examine interactions between
components.
""",

"Core Mathematics": """
Algebra represents quantities using symbols and provides rules for manipulating
expressions and equations. Variables represent values that may be unknown
or may change.

A polynomial is an expression containing variables and non-negative integer
powers with coefficients. Linear and quadratic expressions are common
polynomial forms.

The quadratic formula can be used to solve a quadratic equation when the
equation is written in standard form. The discriminant indicates whether
the equation has two distinct real solutions, one repeated real solution,
or complex solutions.

Functions describe relationships between inputs and outputs. A function
assigns each allowed input a corresponding output according to its rule.

The domain is the set of allowed input values. The range is the set of
resulting output values.

A linear function has a constant rate of change. Its graph is a straight
line. The slope describes how much the output changes when the input changes
by one unit.

Systems of equations contain multiple equations involving common variables.
A solution must satisfy all equations in the system.

Exponential functions involve variables in exponents. They can describe
growth or decay processes.

Logarithms are inverse operations to exponentiation. A logarithm answers
the question of which exponent is required to produce a particular value.

Calculus studies change and accumulation. A derivative measures an instantaneous
rate of change, while an integral represents accumulation and can be used
to calculate areas under curves.

A limit describes the value that a mathematical expression approaches under
a specified condition. Limits provide a foundation for derivatives and
integrals.

Vectors have magnitude and direction. They can represent displacement,
velocity, force, and many other quantities.

Matrices organize numbers into rows and columns. Matrix operations are
used in linear algebra, graphics, scientific computing, and machine learning.

A mathematical proof is a logical argument showing why a statement follows
from accepted definitions, assumptions, and previously established results.
""",

"General Knowledge": """
Earth is the third planet from the Sun and has a natural satellite called
the Moon. Earth has a dynamic atmosphere, oceans, land regions, and diverse
ecosystems.

The water cycle describes the movement of water through evaporation,
condensation, precipitation, infiltration, runoff, and other processes.

The atmosphere contains several gases, with nitrogen and oxygen making up
most of its composition. Atmospheric conditions influence weather and climate.

The solar system contains the Sun and objects gravitationally associated
with it, including planets, dwarf planets, moons, asteroids, and comets.

Light travels through vacuum at approximately 299,792 kilometers per second.
Different wavelengths of visible light correspond to different perceived
colors.

Sound is a mechanical wave that requires a medium for propagation. Its speed
depends on the properties of the medium and environmental conditions.

Maps represent geographic information using symbols, coordinates, scales,
and projections. Latitude measures position north or south of the equator,
while longitude measures position east or west of a reference meridian.

Languages allow people to communicate ideas, information, instructions,
and emotions. Human languages develop and change over time.

History studies evidence about past events, societies, institutions, and
human activities. Historians use sources and compare evidence when constructing
historical interpretations.

Economics studies how people and organizations allocate scarce resources.
Supply and demand are basic concepts used to analyze markets.

A government is an institution or collection of institutions that exercises
authority within a political system. Different countries use different
constitutional structures and systems of government.

Culture includes shared practices, knowledge, arts, traditions, values, and
forms of expression within communities.

Scientific knowledge changes when new evidence supports improved explanations.
A scientific claim should be testable and open to examination.
""",

"Physics Chemistry Biology": """
Physics studies matter, energy, motion, forces, fields, waves, and interactions.
Classical mechanics describes many everyday motions using concepts such as
position, velocity, acceleration, mass, and force.

Newton's laws provide a framework for describing motion and forces. A net
force can cause acceleration according to the relationship between force,
mass, and acceleration.

Energy is a quantity associated with the ability of a system to produce
change. Kinetic energy is associated with motion, while potential energy
depends on position or configuration.

Electric charge is a fundamental physical property. Electric current is
the movement of charge through a conducting system.

Voltage represents electric potential difference. Resistance describes how
strongly a component opposes electric current. Ohm's law relates voltage,
current, and resistance for components that follow the relationship.

Chemistry studies matter, its composition, properties, and transformations.
Atoms contain nuclei made from protons and neutrons and are surrounded by
electrons.

Elements are defined by their atomic number, which corresponds to the number
of protons in the nucleus. The periodic table organizes chemical elements
according to their properties and atomic structure.

Chemical bonds hold atoms together in molecules and compounds. Ionic bonding
involves electrostatic attraction between oppositely charged ions, while
covalent bonding involves shared electrons.

A chemical reaction transforms reactants into products. Conservation of mass
requires that atoms are accounted for on both sides of a balanced reaction.

Acids and bases have characteristic chemical behaviors. The pH scale is
commonly used to describe the acidity or basicity of aqueous solutions.

Biology studies living systems. Cells are the basic structural and functional
units of life.

DNA stores genetic information in living organisms. Genes are regions of
DNA that contribute to biological characteristics through complex molecular
processes.

Proteins perform many functions including structural support, transport,
signaling, and catalysis. Enzymes are biological catalysts that can increase
the rate of chemical reactions.

Photosynthesis allows plants, algae, and some microorganisms to convert light
energy into stored chemical energy.

Cellular respiration involves biochemical pathways that release usable energy
from molecules. ATP is an important energy-carrying molecule in cells.

Ecology examines interactions among organisms and their environments.
Ecosystems contain living organisms and nonliving environmental factors.
""",

"Programming": """
Programming is the process of expressing instructions that a computer can
execute. A program normally contains data, operations, control flow, and
interactions with external systems.

Variables provide names for values. A variable may contain a number, string,
boolean, collection, object, or another data type depending on the language.

Conditional statements allow a program to choose between different paths.
Loops repeat operations while a condition remains true or for each item in
a collection.

Functions group reusable operations. Parameters allow a function to receive
input, and a return value can provide a result.

Good software separates responsibilities into understandable components.
Clear names, small functions, and useful comments can improve maintainability.

Exceptions provide a mechanism for handling unexpected conditions. A robust
program should validate external input and handle expected failure cases.

Application programming interfaces allow software components to communicate
through defined interfaces. A web API commonly exchanges structured data
between a client and a server.

JSON is a text-based data representation format commonly used by web APIs.
Objects contain key-value pairs, while arrays contain ordered collections.

A database-backed application often contains a frontend, backend, and
database. The frontend presents information, the backend implements business
logic, and the database stores persistent information.

Authentication determines who a user is. Authorization determines what an
authenticated user is allowed to do.

Environment variables can store configuration values outside source code.
Sensitive credentials should not be hard-coded into public repositories.

Debugging is the process of locating and correcting program behavior that
does not match expectations. Logs, tests, breakpoints, and careful inspection
are common debugging techniques.

Refactoring changes internal code structure without intentionally changing
the externally expected behavior. Refactoring can improve readability and
maintainability.

Software projects benefit from version control, automated testing,
documentation, dependency management, and reproducible development environments.
""",

"Python Algorithms": """
Python is a high-level programming language known for readable syntax and
a large standard library. Python programs can be used for automation,
web development, scientific computing, data processing, and artificial
intelligence research.

A Python list is an ordered mutable collection. Tuples are ordered but
immutable collections. Dictionaries store key-value associations, and sets
store unique elements.

List comprehensions provide a concise way to construct lists from iterable
objects. They should remain readable and should not replace clearer loops
when the logic becomes complicated.

An algorithm should be selected according to the problem and the constraints.
For a small dataset, a simple algorithm may be sufficient even if a more
complex algorithm has better asymptotic behavior.

Binary search requires an ordered search space. Each comparison can eliminate
roughly half of the remaining candidates.

A stack can be implemented using a Python list. Appending an item pushes it,
and removing the final item pops it.

A queue can be represented using collections.deque, which provides efficient
operations at both ends.

Recursion occurs when a function calls itself. Recursive algorithms require
a base case that stops further recursive calls.

Dynamic programming solves problems by storing results of overlapping
subproblems. It can turn repeated computation into a more efficient process.

Graphs can be represented using adjacency lists or adjacency matrices.
Breadth-first search explores neighboring vertices level by level, while
depth-first search explores one path before returning.

Python's time module and other standard tools can be used to measure program
behavior. Performance optimization should normally begin after identifying
an actual bottleneck.

Type hints document expected types and can help static analysis tools detect
some classes of mistakes.

Virtual environments isolate Python project dependencies. This helps prevent
one project from changing the packages required by another project.

Unit testing frameworks allow programmers to define expected behavior as
automated tests. Small tests make regressions easier to detect.
""",

"Python Basics": """
Python source code is usually stored in files ending with the .py extension.
The Python interpreter reads and executes the program.

The print function displays information. The input function can read text
from a user, although applications often receive input from files, APIs,
or graphical interfaces.

Strings represent text. Python supports indexing and slicing for sequences.
String methods can search, replace, split, join, and transform text.

Integers represent whole numbers, while floating-point values represent
numbers with fractional components. Boolean values represent true or false
conditions.

The if statement performs conditional execution. elif provides another
condition, and else handles the remaining case.

The for loop iterates over items in an iterable. The while loop continues
while its condition remains true.

A function is defined using the def keyword. Parameters describe inputs
accepted by a function.

Modules allow code to be organized across files. The import statement makes
functions, classes, or other objects available.

Python exceptions can be caught using try and except. A finally block can
run cleanup code regardless of whether an exception occurred.

Files can be opened using the open function. The with statement is commonly
used because it automatically handles closing the file.

Classes provide a way to define objects with attributes and methods.
Object-oriented programming can help model related data and behavior.

Python packages distribute reusable code. The Python Package Index contains
many community and third-party packages.

A virtual environment can be created for a project so that dependencies
are installed independently from the global Python installation.

Readable Python code generally follows consistent formatting and naming.
PEP 8 provides widely used style recommendations for Python code.
""",

"Question Answer": """
Question: What is a computer?
Answer: A computer is an electronic system that processes information
according to instructions.

Question: What is a program?
Answer: A program is a set of instructions that a computer can execute.

Question: What is machine learning?
Answer: Machine learning is a method in which computational models learn
patterns from data.

Question: What is a neural network?
Answer: A neural network is a computational model composed of connected
mathematical transformations that can learn parameters from examples.

Question: What is a token?
Answer: A token is a unit of text processed by a language model.

Question: What is an algorithm?
Answer: An algorithm is a defined sequence of steps for solving a problem.

Question: What is RAM?
Answer: RAM is temporary computer memory used to hold active programs and data.

Question: What is an operating system?
Answer: An operating system manages computer hardware and provides services
for applications.

Question: What is a database?
Answer: A database is a structured system for storing and retrieving data.

Question: What is an API?
Answer: An API is a defined interface that allows software components to
communicate.

Question: What is Python?
Answer: Python is a general-purpose programming language with readable syntax.

Question: What is a variable?
Answer: A variable is a named reference to a value used by a program.

Question: What is a function?
Answer: A function is a reusable unit of code that performs a defined operation.

Question: What is probability?
Answer: Probability is a mathematical way of representing the likelihood
of an event.

Question: What is gravity?
Answer: Gravity is an interaction associated with mass and energy that causes
objects to attract one another.

Question: What is DNA?
Answer: DNA is a molecule that stores genetic information in living organisms.

Question: What is photosynthesis?
Answer: Photosynthesis is a biological process that converts light energy
into chemical energy in organisms capable of performing it.

Question: What is a database table?
Answer: A database table organizes related records into rows and columns.

Question: What is debugging?
Answer: Debugging is the process of finding and correcting problems in software.

Question: What is a compiler?
Answer: A compiler translates source code into another representation that can
be executed or processed by a computer system.
""",

"Science": """
Science is a systematic way of developing explanations about the natural
world using observations, measurements, experiments, models, and evidence.

A hypothesis is a testable proposed explanation. An experiment can be designed
to investigate whether observations are consistent with a hypothesis.

Measurements include numerical values and units. SI units provide standardized
units for many scientific quantities.

Mass measures the amount of matter in an object. Weight is a force associated
with gravity acting on mass.

Temperature describes the thermal state of a system. Heat refers to energy
transferred because of a temperature difference.

Matter can exist in different states such as solid, liquid, and gas. Changes
of state include melting, freezing, evaporation, condensation, and sublimation.

Density is mass divided by volume. It can help identify materials when other
conditions are known.

Waves transfer energy and information through oscillations. Mechanical waves
require a medium, while electromagnetic waves can travel through vacuum.

The electromagnetic spectrum includes radio waves, microwaves, infrared,
visible light, ultraviolet, X-rays, and gamma rays.

Atoms can form molecules through chemical interactions. Chemical equations
represent transformations using formulas for reactants and products.

Biological systems are organized at multiple levels, including molecules,
cells, tissues, organs, organisms, populations, communities, and ecosystems.

Evolution describes changes in inherited characteristics of populations over
generations. Natural selection is one mechanism that can change populations.

The immune system protects organisms from many biological threats. It contains
multiple types of cells and molecular processes.

The scientific method is not a single rigid sequence used for every research
project. Scientists use different methods depending on the question and
available evidence.
""",

"Conversations": """
User: Hello
Assistant: Hello! How can I help you today?

User: What can you do?
Assistant: I can explain concepts, answer questions, help with programming,
work through mathematical problems, and assist with learning.

User: I do not understand this topic.
Assistant: No problem. We can break the topic into smaller ideas and explain
each part using simple examples.

User: Can you explain it simply?
Assistant: Yes. I can start with the basic idea and then gradually introduce
the more advanced parts.

User: What should I do when my program has an error?
Assistant: First read the error message carefully. Identify the file and line
where the error occurred, then inspect the surrounding code and reproduce
the problem with the smallest useful example.

User: How can I learn programming?
Assistant: Start with variables, conditions, loops, functions, data structures,
and small projects. Practice regularly and read error messages carefully.

User: How can I improve my AI project?
Assistant: Measure the current behavior first, then improve one major component
at a time such as data quality, tokenization, model architecture, training,
evaluation, or inference.

User: What is the difference between training and inference?
Assistant: Training changes model parameters using data and a loss function.
Inference uses an already trained model to produce outputs.

User: My model gives strange answers.
Assistant: Check the dataset, tokenizer, training loss, context length,
sampling settings, and whether the model has enough relevant training data.

User: Can a small model become useful?
Assistant: Yes. A small model can become useful for focused tasks, although
its capabilities and knowledge will be limited compared with much larger
models.

User: What is the best way to debug?
Assistant: Make the problem reproducible, inspect the error, isolate the
smallest failing component, test assumptions, and make one change at a time.

User: I want to learn something difficult.
Assistant: Start with the prerequisites, learn one concept at a time, and
practice by applying each concept to small problems.
""",

"Technology": """
Technology includes tools and systems created to solve practical problems.
Computers, networks, databases, mobile devices, and cloud services are
examples of modern information technology.

A server provides services or resources to clients. A client sends requests
and receives responses.

Cloud computing provides computing resources through network-accessible
services. Common categories include infrastructure, platforms, and software
services.

Virtualization allows multiple isolated computing environments to operate
on shared physical hardware. Containers provide another form of application
isolation.

The internet is a global network of interconnected networks. Routers forward
packets between networks.

DNS translates domain names into network information such as IP addresses.
It allows users to access services using memorable names.

HTTPS uses TLS to provide encrypted communication between a client and server.
Encryption protects data while it travels across a network, although it does
not make an entire system automatically secure.

Backups create additional copies of important data. A useful backup strategy
considers storage location, frequency, retention, and restoration testing.

Version control helps developers track changes and collaborate on software.
Branches can isolate changes before they are integrated into a main codebase.

Continuous integration automatically builds and tests software when changes
are submitted. Continuous delivery extends automation toward deployment.

Web applications commonly use HTML for structure, CSS for presentation, and
JavaScript for behavior. Modern frameworks can organize larger interfaces.

A REST-style API commonly represents resources through HTTP methods and
structured responses. Good API design considers validation, errors,
authentication, rate limits, and versioning.

Artificial intelligence is increasingly integrated into search, software
development, education, creative tools, data analysis, and automation.
""",

"General Training": """
The purpose of a training dataset is to provide examples from which a model
can learn useful statistical patterns. A dataset should be relevant to the
model's intended tasks.

High-quality training data should be clear, consistent, diverse, and
reasonably accurate. Duplicate examples can cause a model to see some
patterns disproportionately often.

Data preprocessing can remove unwanted formatting, normalize structures,
split documents, and create training sequences.

A tokenizer maps text to numerical token identifiers. A vocabulary contains
the token units recognized by the tokenizer.

A language model training example can contain a sequence of input tokens and
a target sequence shifted by one position. The model learns to predict the
next token.

Cross-entropy loss is commonly used for next-token prediction. Lower loss
generally means the model assigns higher probability to the observed target
tokens, although loss alone does not fully describe usefulness.

Training and validation data should be separated. If validation examples
are also used during training, validation results can become misleading.

Batch size determines how many training examples are processed before an
optimizer update. Larger batches may require more memory.

Learning rate controls the approximate size of parameter updates. An
excessively large learning rate can destabilize training, while an extremely
small rate can make training slow.

Gradient clipping limits unusually large gradients. Weight decay can help
regularize some neural network training setups.

A checkpoint stores model parameters and other information needed to resume
or evaluate training. Saving the best validation checkpoint can preserve a
model from an earlier point in training.

Evaluation should include examples representing the actual intended use of
the model. A model that memorizes training text may still perform poorly on
new questions.

Small experiments are useful when building a model from scratch. It is often
better to verify each component independently before increasing model size.
""",

"Web Software": """
The web is built around communication between clients and servers. A browser
acts as a client and requests resources from servers.

HTML describes the structure of a web document. Elements can represent
headings, paragraphs, links, images, forms, tables, and other content.

CSS controls presentation. Selectors identify elements, while properties
define visual and layout behavior.

JavaScript adds interactive behavior to web pages. It can respond to events,
modify the document, communicate with APIs, and manage application state.

A frontend application runs primarily on the user's device. A backend
application runs on a server and can handle authentication, business logic,
database access, and external services.

HTTP requests contain methods, headers, and sometimes a body. Common methods
include GET, POST, PUT, PATCH, and DELETE.

HTTP status codes communicate the general result of a request. Successful
responses commonly use codes in the 200 range, client errors in the 400
range, and server errors in the 500 range.

JSON is frequently used to exchange structured data between web applications.
Its simple structure makes it convenient for many APIs.

A database query retrieves or modifies stored information. Parameterized
queries help prevent many classes of injection problems.

Frontend applications often use state to represent changing information.
State management becomes important when multiple components depend on the
same data.

Caching can reduce repeated work by storing reusable results. Cache design
must consider freshness and invalidation.

Responsive design allows interfaces to adapt to different screen sizes.
Accessibility helps ensure that websites can be used by people with
different abilities and assistive technologies.

Web performance depends on factors such as network latency, asset size,
JavaScript execution, rendering, server response time, and caching.

Good web applications validate inputs, handle errors, protect credentials,
use secure communication, and expose only the functionality necessary for
each user or service.
"""
}

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def approximate_tokens(text):
    """
    Approximate token count before running the real tokenizer.
    This is intentionally conservative.
    """
    words = re.findall(r"\S+", text)
    return int(len(words) * 1.35)


def backup_file(path):
    if path.exists():
        destination = BACKUP_DIR / path.name
        if not destination.exists():
            shutil.copy2(path, destination)


def append_content(path, content):
    with path.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 60)
        f.write("\n")
        f.write("ORIGINAL OWN AI TRAINING MATERIAL\n")
        f.write("=" * 60)
        f.write("\n\n")
        f.write(content.strip())
        f.write("\n")


# ------------------------------------------------------------
# Backup
# ------------------------------------------------------------

print("\nCreating backups...")

for filename in FILE_CATEGORIES:
    path = RAW_DIR / filename
    backup_file(path)

print("Backup directory:")
print(BACKUP_DIR)

# ------------------------------------------------------------
# Expand files
# ------------------------------------------------------------

print("\nExpanding dataset files...")

total_added_chars = 0

for filename, category in FILE_CATEGORIES.items():

    path = RAW_DIR / filename

    if not path.exists():
        print(f"[SKIP] {filename} not found")
        continue

    content = CONTENT.get(category)

    if not content:
        print(f"[SKIP] No generated content for {filename}")
        continue

    # Prevent accidental duplicate expansion.
    existing = path.read_text(encoding="utf-8", errors="ignore")

    marker = "ORIGINAL OWN AI TRAINING MATERIAL"

    if marker in existing:
        print(f"[SKIP] Already expanded: {filename}")
        continue

    append_content(path, content)

    total_added_chars += len(content)

    print(f"[OK] {filename}")
    print(f"     Added characters: {len(content):,}")


# ------------------------------------------------------------
# Build combined dataset
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EXPANSION COMPLETE")
print("=" * 60)

print(f"Added characters: {total_added_chars:,}")

all_text = []

for filename in FILE_CATEGORIES:

    path = RAW_DIR / filename

    if path.exists():
        try:
            all_text.append(
                path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
            )
        except Exception as e:
            print(f"[WARNING] Could not read {filename}: {e}")

combined = "\n\n".join(all_text)

characters = len(combined)
words = len(re.findall(r"\S+", combined))
estimated_tokens = approximate_tokens(combined)

print(f"Files: {len(all_text)}")
print(f"Characters: {characters:,}")
print(f"Words: {words:,}")
print(f"Estimated tokens: {estimated_tokens:,}")

print("\nTarget:")
print(f"{TARGET_TOKENS:,}+ tokens")

if estimated_tokens >= TARGET_TOKENS:
    print("\nSTATUS: TARGET REACHED")
else:
    remaining = TARGET_TOKENS - estimated_tokens
    print("\nSTATUS: MORE DATA REQUIRED")
    print(f"Estimated tokens remaining: {remaining:,}")

print("=" * 60)
print("IMPORTANT:")
print("This script creates original educational training material.")
print("Review the generated dataset before serious model training.")
print("=" * 60)