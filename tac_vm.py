"""TAC virtual machine that executes compiled policy code."""

from .tac_compiler import ARITH_FN, CMP_FN


class TACInterpreter:
    def __init__(self, code):
        self.code = code
        self.labels = {
            instruction[1]: i
            for i, instruction in enumerate(code)
            if instruction[0] == "LABEL"
        }

    @staticmethod
    def value(operand, state, temps):
        kind, value = operand

        if kind == "CONST":
            return value
        if kind == "VAR":
            if value not in state:
                raise ValueError(f"Missing signal: {value}")
            return state[value]
        if kind == "TEMP":
            return temps[value]

        raise ValueError(f"Unknown operand: {operand}")

    def run(self, state):
        temps = {}
        pc = 0

        while pc < len(self.code):
            ins = self.code[pc]
            op = ins[0]

            if op == "LABEL":
                pc += 1

            elif op == "BIN":
                _, target, symbol, left, right = ins
                a = self.value(left, state, temps)
                b = self.value(right, state, temps)
                temps[target] = ARITH_FN[symbol](a, b)
                pc += 1

            elif op == "CMP":
                _, target, symbol, left, right = ins
                a = self.value(left, state, temps)
                b = self.value(right, state, temps)
                temps[target] = CMP_FN[symbol](a, b)
                pc += 1

            elif op == "BRANCH":
                _, operand, yes, no = ins
                result = self.value(operand, state, temps)
                pc = self.labels[yes if result else no]

            elif op == "JUMP":
                pc = self.labels[ins[1]]

            elif op == "ACTION":
                return ins[1]

            else:
                raise ValueError(f"Unknown instruction: {ins}")

        return "CONTINUE"
