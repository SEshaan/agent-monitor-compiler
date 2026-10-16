"""AST -> TAC policy compiler."""

import ast
import operator

# Comparison operators: (operator symbol, Python callable).
CMP = {
    ast.Gt: (">", operator.gt),
    ast.Lt: ("<", operator.lt),
    ast.GtE: (">=", operator.ge),
    ast.LtE: ("<=", operator.le),
    ast.Eq: ("==", operator.eq),
    ast.NotEq: ("!=", operator.ne),
}

# Arithmetic operators: (operator symbol, Python callable).
ARITH = {
    ast.Add: ("+", operator.add),
    ast.Sub: ("-", operator.sub),
    ast.Mult: ("*", operator.mul),
    ast.Div: ("/", operator.truediv),
}

# Symbol -> function lookups, built once instead of on every VM instruction.
ARITH_FN = {sym: fn for sym, fn in ARITH.values()}
CMP_FN = {sym: fn for sym, fn in CMP.values()}

ACTIONS = {"HALT", "CONTINUE", "CHANGE_STRATEGY"}


class PolicyCompiler:
    def __init__(self):
        self.code = []
        self.temp_id = 0
        self.label_id = 0

    def temp(self):
        self.temp_id += 1
        return f"t{self.temp_id}"

    def label(self, prefix="L"):
        self.label_id += 1
        return f"{prefix}{self.label_id}"

    def emit(self, *instruction):
        self.code.append(instruction)

    def expression(self, node):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise ValueError("Unsupported constant")
            return ("CONST", node.value)

        if isinstance(node, ast.Name):
            return ("VAR", node.id)

        if isinstance(node, ast.BinOp):
            op = ARITH.get(type(node.op))
            if op is None:
                raise ValueError("Unsupported arithmetic operator")

            left = self.expression(node.left)
            right = self.expression(node.right)
            target = self.temp()
            self.emit("BIN", target, op[0], left, right)
            return ("TEMP", target)

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError("Chained comparisons unsupported")

            op = CMP.get(type(node.ops[0]))
            if op is None:
                raise ValueError("Unsupported comparison")

            left = self.expression(node.left)
            right = self.expression(node.comparators[0])
            target = self.temp()
            self.emit("CMP", target, op[0], left, right)
            return ("TEMP", target)

        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    def condition(self, node, true_label, false_label):
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for child in node.values[:-1]:
                    next_label = self.label("L_and")
                    self.condition(child, next_label, false_label)
                    self.emit("LABEL", next_label)

                self.condition(node.values[-1], true_label, false_label)
                return

            if isinstance(node.op, ast.Or):
                for child in node.values[:-1]:
                    next_label = self.label("L_or")
                    self.condition(child, true_label, next_label)
                    self.emit("LABEL", next_label)

                self.condition(node.values[-1], true_label, false_label)
                return

        value = self.expression(node)
        self.emit("BRANCH", value, true_label, false_label)

    def compile(self, policies):
        self.code = []
        self.temp_id = 0
        self.label_id = 0

        end = self.label("L_end")

        for expression, action in policies:
            if action not in ACTIONS:
                raise ValueError(f"Invalid action: {action}")

            tree = ast.parse(expression, mode="eval")
            next_rule = self.label("L_next")
            matched = self.label("L_match")

            self.condition(tree.body, matched, next_rule)
            self.emit("LABEL", matched)
            self.emit("ACTION", action)
            self.emit("JUMP", end)
            self.emit("LABEL", next_rule)

        self.emit("ACTION", "CONTINUE")
        self.emit("LABEL", end)
        return self.code
