import abc
import copy
from typing import List, override
from errors import VariableDefinedError
from tokenclass import TokType, Token
from dataclasses import dataclass


def get_label() -> str:
    attr = getattr(get_label, "index", 0) + 1
    setattr(get_label, "index", attr)
    return ".L" + str(attr)


class AstNode(abc.ABC):
    @abc.abstractmethod
    def compile(self, var_map: dict, stack_offset: int) -> tuple[str, dict, int]:
        return "\t; NOT IMPLEMENTED\n", var_map, stack_offset


class Constant(AstNode):
    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def compile(self, var_map, stack_offset):
        return f"\tmov rax, {self.value}\n", var_map, stack_offset


class UnaryOperator(AstNode):
    def __init__(self, op: TokType, exp) -> None:
        self.op = op
        self.exp = exp

    @override
    def compile(self, var_map, stack_offset):
        res, var_map, stack_offset = self.exp.compile(var_map, stack_offset)

        if self.op == TokType.MINUS:
            res += "\tneg rax\n"
        if self.op == TokType.BITW_COMPLIMENT:
            res += "\tnot rax\n"
        if self.op == TokType.LOGIC_NEGATION:
            res += "\ttest rax, rax\n"
            res += "\tmov rax, 0\n"
            res += "\tsetz al\n"

        return res, var_map, stack_offset


class BinaryOperator(AstNode):
    def __init__(self, exp, op: TokType, right) -> None:
        self.exp = exp
        self.op = op
        self.right = right

    @override
    def compile(self, var_map, stack_offset):
        res, var_map, stack_offset = self.exp.compile(var_map, stack_offset)
        res += "\tpush rax\n"

        code, var_map, stack_offset = self.right.compile(var_map, stack_offset)
        res += code
        res += "\tpop rbx\n"

        if self.op == TokType.ADDITION:
            res += "\tadd rax, rbx\n"
        if self.op == TokType.MINUS:
            res += "\tsub rbx, rax\n"
            res += "\tmov rax, rbx\n"
        if self.op == TokType.MULTIPLICATION:
            res += "\timul rbx\n"
        if self.op == TokType.DIVISION:
            res += "\txor rdx, rdx\n"
            res += "\txchg rbx, rax\n"
            res += "\tidiv rbx\n"
        if self.op == TokType.MODULO:
            res += "\txor rdx, rdx\n"
            res += "\txchg rbx, rax\n"
            res += "\tidiv rbx\n"
            res += "\tmov rax, rdx\n"
        if self.op == TokType.EQUAL:
            res += "\tcmp rbx, rax\n"
            res += "\tsete al\n"
        if self.op == TokType.NOT_EQUAL:
            res += "\tcmp rbx, rax\n"
            res += "\tsetne al\n"
        if self.op == TokType.GREATER_OR_EQ:
            res += "\tcmp rbx, rax\n"
            res += "\tsetge al\n"
        if self.op == TokType.GREATER:
            res += "\tcmp rbx, rax\n"
            res += "\tsetg al\n"
        if self.op == TokType.LESS_OR_EQ:
            res += "\tcmp rbx, rax\n"
            res += "\tsetle al\n"
        if self.op == TokType.LESS:
            res += "\tcmp rbx, rax\n"
            res += "\tsetl al\n"
        if self.op == TokType.AND:
            label = get_label()

            res += "\ttest rbx, rbx\n"
            res += "\tsetnz bl\n"
            res += f"\tjz {label}\n"
            res += "\ttest rax, rax\n"
            res += "\tsetnz al\n"
            res += label + ":\n"
            res += "\tand al, bl\n"
        if self.op == TokType.OR:
            label = get_label()

            res += "\ttest rbx, rbx\n"
            res += "\tsetnz bl\n"
            res += f"\tjnz {label}\n"
            res += "\ttest rax, rax\n"
            res += "\tsetnz al\n"
            res += label + ":\n"
            res += "\tor al, bl\n"

        return res, var_map, stack_offset


class Return(AstNode):
    def __init__(self, exp) -> None:
        self.exp = exp

    @override
    def compile(self, var_map, stack_offset):
        res, var_map, stack_offset = self.exp.compile(var_map, stack_offset)

        if stack_offset > 0:
            deallocated = stack_offset
            res += f"\tadd rsp, {deallocated}\n"
            stack_offset -= deallocated

        res += "\tjmp .Le\n"

        return res, var_map, stack_offset


class Declare(AstNode):
    def __init__(self, variable: Token, exp) -> None:
        self.exp = exp
        self.variable = variable

    @override
    def compile(self, var_map, stack_offset):
        if self.exp:
            res, var_map, stack_offset = self.exp.compile(var_map, stack_offset)
        else:
            res = "\tmov rax, 0\n"

        identifier = self.variable.value

        stack_offset += 8

        new_map = copy.deepcopy(var_map)
        new_map[identifier] = stack_offset

        res += "\tsub rsp, 8\n"
        res += f"\tmov [rbp - {stack_offset}], rax\n"

        return res, new_map, stack_offset


class Variable(AstNode):
    def __init__(self, variable: Token) -> None:
        self.variable = variable

    @override
    def compile(self, var_map, stack_offset):
        identifier = self.variable.value

        offset = var_map[identifier]

        if offset > 0:
            res = f"\tmov rax, [rbp - {offset}]\n"
        else:
            res = f"\tmov rax, [rbp + {-offset}]\n"

        return res, var_map, stack_offset


class Assign(AstNode):
    def __init__(self, variable: Token, exp) -> None:
        self.exp = exp
        self.variable = variable

    @override
    def compile(self, var_map, stack_offset):
        res, var_map, stack_offset = self.exp.compile(var_map, stack_offset)

        identifier = self.variable.value

        offset = var_map[identifier]

        res += f"\tmov [rbp - {offset}], rax\n"

        return res, var_map, stack_offset


class Conditional(AstNode):
    def __init__(self, exp, statement, statement_option) -> None:
        self.exp = exp
        self.statement = statement
        self.statement_option = statement_option

    @override
    def compile(self, var_map, stack_offset):
        res = "\t; IF CONDITION\n"

        code, var_map, stack_offset = self.exp.compile(var_map, stack_offset)
        res += code

        label_else = ""
        label_end = get_label()

        res += "\ttest al, al\n"

        if self.statement_option:
            label_else = get_label()
            res += f"\tjz {label_else}\n"
        else:
            res += f"\tjz {label_end}\n"

        res += "\t; STATEMENT\n"

        code, var_map, stack_offset = self.statement.compile(var_map, stack_offset)
        res += code

        res += f"\tjmp {label_end}\n"

        if self.statement_option:
            res += "\t; ELSE\n"
            res += label_else + ":\n"

            code, var_map, stack_offset = self.statement_option.compile(
                var_map, stack_offset
            )
            res += code

        res += "\t; ENDIF\n"
        res += label_end + ":\n"

        return res, var_map, stack_offset


class For(AstNode):
    def __init__(self, exp, condition, control, block):
        self.exp = exp
        self.condition = condition
        self.control = control
        self.block = block

    @override
    def compile(self, var_map, stack_offset):
        # EXPRESSION
        code, var_map, stack_offset = self.exp.compile(var_map, stack_offset)
        res = code

        loop_label = get_label()
        condition_label = get_label()

        res += "\tjmp " + condition_label + "\n"

        # LOOP
        res += loop_label + ":\n"
        code, var_map, stack_offset = self.block.compile(var_map, stack_offset)
        res += code

        # CONTROL
        code, var_map, stack_offset = self.control.compile(var_map, stack_offset)
        res += code

        # CONDITION
        res += condition_label + ":\n"

        code, var_map, stack_offset = self.condition.compile(var_map, stack_offset)
        res += code
        res += "\ttest al, al\n"
        res += "\tjnz " + loop_label + "\n"

        return res, var_map, stack_offset


class ForDecl(AstNode):
    def __init__(self, declaration, condition, control, block):
        self.declaration = declaration
        self.condition = condition
        self.control = control
        self.block = block

    @override
    def compile(self, var_map, stack_offset):
        # DECLARATION
        code, var_map, stack_offset = self.declaration.compile(var_map, stack_offset)
        res = code

        loop_label = get_label()
        condition_label = get_label()

        res += "\tjmp " + condition_label + "\n"

        # LOOP
        res += loop_label + ":\n"
        code, var_map, stack_offset = self.block.compile(var_map, stack_offset)
        res += code

        # CONTROL
        code, var_map, stack_offset = self.control.compile(var_map, stack_offset)
        res += code

        # CONDITION
        res += condition_label + ":\n"

        code, var_map, stack_offset = self.condition.compile(var_map, stack_offset)
        res += code
        res += "\ttest al, al\n"
        res += "\tjnz " + loop_label + "\n"

        deallocated = 8
        res += f"\tadd rsp, {deallocated}\n"
        stack_offset -= deallocated

        return res, var_map, stack_offset


class While(AstNode):
    def __init__(self, condition, block) -> None:
        self.block = block
        self.condition = condition

    @override
    def compile(self, var_map, stack_offset):
        loop_label = get_label()
        condition_label = get_label()

        res = "\tjmp " + condition_label + "\n"

        # LOOP
        res += loop_label + ":\n"
        code, var_map, stack_offset = self.block.compile(var_map, stack_offset)
        res += code

        # CONDITION
        res += condition_label + ":\n"

        code, var_map, stack_offset = self.condition.compile(var_map, stack_offset)
        res += code
        res += "\ttest al, al\n"
        res += "\tjnz " + loop_label + "\n"

        return res, var_map, stack_offset


class DoLoop(AstNode):
    def __init__(self, condition, block) -> None:
        self.block = block
        self.condition = condition

    @override
    def compile(self, var_map, stack_offset):
        loop_label = get_label()
        condition_label = get_label()

        # LOOP
        res = loop_label + ":\n"
        code, var_map, stack_offset = self.block.compile(var_map, stack_offset)
        res += code

        # CONDITION
        res += condition_label + ":\n"

        code, var_map, stack_offset = self.condition.compile(var_map, stack_offset)
        res += code
        res += "\ttest al, al\n"
        res += "\tjnz " + loop_label + "\n"

        return res, var_map, stack_offset


class Break(AstNode):
    @override
    def compile(self, var_map, stack_offset):
        return super().compile(var_map, stack_offset)


class Continue(AstNode):
    @override
    def compile(self, var_map, stack_offset):
        return super().compile(var_map, stack_offset)


class Compound(AstNode):
    def __init__(self, block: List, function: bool) -> None:
        self.block = block
        self.function = function

    @override
    def compile(self, var_map, stack_offset):
        res = ""
        scope = set()

        for item in self.block:
            if type(item) is Declare:
                if item.variable.value in scope:
                    raise VariableDefinedError

                scope.add(item.variable.value)
                code, var_map, stack_offset = item.compile(var_map, stack_offset)
                res += code
            else:
                code, _, _ = item.compile(var_map, stack_offset)
                res += code

        if len(scope) > 0 and not self.function:
            dealocated = len(scope) * 8

            res += f"\tadd rsp, {dealocated}\n"
            stack_offset -= dealocated

        return res, var_map, stack_offset


class Function:
    def __init__(self, name: str, body: Compound, args: List[str]) -> None:
        self.name = name
        self.body = body
        self.args = args

    def compile(self) -> str:
        var_map = dict()
        stack_offset = 0

        offset = -16
        for arg in self.args:
            var_map[arg] = offset
            offset -= 8

        res = f"{self.name}:\n"

        res += "\tpush rbp\n"
        res += "\tmov rbp, rsp\n"

        code, var_map, stack_offset = self.body.compile(var_map, stack_offset)

        res += code

        res += ".Le:\n"

        # if stack_offset > 0:
        #    res += f"\tadd rsp, {stack_offset}\n"

        res += "\tpop rbp\n"
        res += "\tret\n\n"

        return res


class FunctionCall(AstNode):
    def __init__(self, name: str, args) -> None:
        self.name = name
        self.args = args

    @override
    def compile(self, var_map, stack_offset):
        args_bytes = 8 * len(self.args)
        res = ""

        self.args.reverse()

        for arg in self.args:
            code, var_map, stack_offset = arg.compile(var_map, stack_offset)
            res += code
            res += "\tpush rax\n"

        res += f"\tcall {self.name}\n"
        res += f"\tadd rsp, {args_bytes}\n"

        return res, var_map, stack_offset


class Program:
    def __init__(self, functions: List[Function]) -> None:
        self.functions = functions

    def compile(self):
        res = ""
        for func in self.functions:
            res += func.compile()

        return res

