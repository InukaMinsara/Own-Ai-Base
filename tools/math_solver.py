import ast
import math
import operator


BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant) and isinstance(
        node.value,
        (int, float),
    ):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in BIN_OPS:
        left = _eval(node.left)
        right = _eval(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")

        return BIN_OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPS:
        return UNARY_OPS[type(node.op)](
            _eval(node.operand)
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function.")

        fn = FUNCTIONS.get(node.func.id)
        if fn is None:
            raise ValueError("Unsupported function.")

        return fn(*[_eval(arg) for arg in node.args])

    raise ValueError("Unsupported mathematical expression.")


def solve_expression(expression):
    expression = expression.strip().replace("^", "**")

    if not expression or len(expression) > 300:
        raise ValueError("Invalid expression.")

    tree = ast.parse(
        expression,
        mode="eval",
    )

    value = _eval(tree)

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    return value


def looks_like_math(text):
    lowered = text.lower().strip()
    keywords = (
        "calculate",
        "solve",
        "what is",
        "how much",
        "math",
        "equation",
    )

    has_math_symbol = any(
        symbol in lowered
        for symbol in ("+", "-", "*", "/", "^", "=")
    )

    has_math_word = any(
        word in lowered
        for word in keywords
    )

    digits = sum(ch.isdigit() for ch in lowered)

    return digits > 0 and (has_math_symbol or has_math_word)


def extract_expression(text):
    cleaned = text.strip().rstrip("?.")

    prefixes = (
        "calculate ",
        "solve ",
        "what is ",
        "what's ",
        "how much is ",
    )

    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    if "=" in cleaned:
        cleaned = cleaned.split("=", 1)[0]

    return cleaned.strip()
