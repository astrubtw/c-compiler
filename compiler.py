from types import MappingProxyType
from typing import TextIO, List
from enum import Enum, auto
import re
import subprocess
import copy
import abc


class TokType(Enum):
    # ONE CHAR OPERATORS
    OPEN_BRACE = auto()
    CLOSE_BRACE = auto()
    OPEN_PAREN = auto()
    CLOSE_PAREN = auto()
    SEMICOLON = auto()
    MINUS = auto()
    BITW_COMPLIMENT = auto()
    LOGIC_NEGATION = auto()
    ADDITION = auto()
    MULTIPLICATION = auto()
    DIVISION = auto()
    LESS = auto()
    GREATER = auto()
    MODULO = auto()
    ASSIGNMENT = auto()
    
    # TWO CHAR OPERATORS
    AND = auto()
    OR = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_OR_EQ = auto()
    GREATER_OR_EQ = auto()

    # OTHER
    KEYWORD = auto()
    IDENTIFIER = auto()
    INT_LITERAL = auto()


def get_label() -> str:
    get_label.index = getattr(get_label, 'index', 0) + 1
    return ".L" + str(get_label.index)


class VariableDefinedError(Exception):
    """Variable already defined in scope."""


class Token():
    def __init__(self, type_: TokType, value: int | str = None) -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        if self.value:
            return f"{self.type.name}[{self.value}]"
        
        return f"{self.type.name}"


class AstNode(abc.ABC):
    @abc.abstractclassmethod
    def compile(self, var_map: dict, stack_offset: int) -> tuple[str, dict, int]:
        return "\t; NOT IMPLEMENTED\n", var_map, stack_offset


class Constant(AstNode):
    def __init__(self, value: int) -> None:
        self.value = value

    def compile(self, var_map, stack_offset) -> str:
        return f"\tmov rax, {self.value}\n", var_map, stack_offset


class UnaryOperator(AstNode):
    def __init__(self, op: TokType, exp) -> None:
        self.op = op
        self.exp = exp

    def compile(self, var_map, stack_offset) -> str:
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

    def compile(self, var_map, stack_offset):
        res, var_map, stack_offset = self.exp.compile(var_map, stack_offset)

        res += "\tjmp .Le\n"

        return res, var_map, stack_offset
    

class Declare(AstNode):
    def __init__(self, variable: Token, exp) -> None:
        self.exp = exp
        self.variable = variable

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

    def compile(self, var_map, stack_offset):
        identifier = self.variable.value

        offset = var_map[identifier]

        res = f"\tmov rax, [rbp - {offset}]\n"

        return res, var_map, stack_offset
    

class Assign(AstNode):
    def __init__(self, variable: Token, exp) -> None:
        self.exp = exp
        self.variable = variable

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


    def compile(self, var_map, stack_offset):
        res = "\t; IF CONDITION\n"

        code, var_map, stack_offset = self.exp.compile(var_map, stack_offset)
        res += code

        label_else = None
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

            code, var_map, stack_offset = self.statement_option.compile(var_map, stack_offset)
            res += code

        res += "\t; ENDIF\n"
        res += label_end + ":\n"

        return res, var_map, stack_offset


class Compound(AstNode):
    def __init__(self, block: List, function: bool) -> None:
        self.block = block
        self.function = function

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


class Function():
    def __init__(self, name: str, body: Compound) -> None:
        self.name = name
        self.body = body


    def compile(self) -> str:
        var_map = dict()
        stack_offset = 0

        res = f"{self.name}:\n"

        res += "\tpush rbp\n"
        res += "\tmov rbp, rsp\n"

        code, var_map, stack_offset = self.body.compile(var_map, stack_offset)

        res += code

        res += ".Le:\n"

        if stack_offset > 0:
            res += f"\tadd rsp, {stack_offset}\n"

        res += "\tpop rbp\n"
        res += "\tret\n"

        return res


class Program():
    def __init__(self, entry: Function) -> None:
        self.entry = entry


class SyntaxTree():
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.current = 0
        self.parse()


    def parse(self):
        self.root = self.program()


    def error(self, msg: str):
        raise Exception(msg)


    def advance(self):
        self.current += 1

    def rewind(self):
        self.current -= 1


    def get_current(self) -> Token:
        return self.tokens[self.current]
    
    
    def get_previous(self) -> Token:
        return self.tokens[self.current - 1]


    def match_token(self, *types: TokType) -> bool:
        for expected_type in types:
            if self.get_current().type == expected_type:
                self.current += 1
                return True
        
        return False
    

    def consume(self, token: TokType):
        if not self.match_token(token):
            self.error("Expected token: " + token.name)


    def check_token(self, type: TokType) -> bool:
        return self.get_current().type == type

    
    def check_keyword(self) -> str | None:
        if self.get_current().type == TokType.KEYWORD:
            return self.get_current().value
        
        return None

    
    def program(self) -> Program:
        return Program(self.function())
    

    def function(self) -> Function:
        self.match_token(TokType.KEYWORD)

        self.match_token(TokType.IDENTIFIER)
        identifier = self.get_previous()

        self.match_token(TokType.OPEN_PAREN)
        self.match_token(TokType.CLOSE_PAREN)

        self.match_token(TokType.OPEN_BRACE)

        block = list()

        while not self.check_token(TokType.CLOSE_BRACE):
            block.append(self.block_item())

        self.match_token(TokType.CLOSE_BRACE)

        body = Compound(block, True)

        return Function(identifier.value, body)
    

    def block_item(self):
        keyword = self.check_keyword()

        if keyword == "int":
            self.consume(TokType.KEYWORD)

            return self.declaration()

        return self.statement()


    def declaration(self):
        self.consume(TokType.IDENTIFIER)

        identifier = self.get_previous()
        exp = None

        if self.match_token(TokType.ASSIGNMENT):
            exp = self.expression()

        self.consume(TokType.SEMICOLON)

        return Declare(identifier, exp)
    

    def statement(self):
        if self.match_token(TokType.KEYWORD):
            keyword = self.get_previous()

            match keyword.value:
                case "return":
                    exp = self.expression()

                    self.consume(TokType.SEMICOLON)

                    return Return(exp)
                
                case "if":
                    self.consume(TokType.OPEN_PAREN)

                    exp = self.expression()

                    self.consume(TokType.CLOSE_PAREN)

                    if_block = self.statement()

                    if not self.match_token(TokType.KEYWORD):
                        return Conditional(exp, if_block, None)
                    
                    keyword = self.get_previous()

                    if keyword.value != "else":
                        self.rewind()
                        return Conditional(exp, if_block, None)

                    else_block = self.statement()

                    return Conditional(exp, if_block, else_block)

        if self.match_token(TokType.OPEN_BRACE):
            block = list()

            while not self.check_token(TokType.CLOSE_BRACE):
                block.append(self.block_item())

            self.match_token(TokType.CLOSE_BRACE)

            return Compound(block, False)

        exp = self.expression()
        self.consume(TokType.SEMICOLON)

        return exp

    
    def expression(self):
        if self.match_token(TokType.IDENTIFIER):
            identifier = self.get_previous()

            if self.match_token(TokType.ASSIGNMENT):
                return Assign(identifier, self.expression())
            
            self.rewind()

        return self.or_comparison()
            

    def or_comparison(self):
        and_comparison = self.and_comparison()

        while self.match_token(TokType.OR):
            token = self.get_previous()
            right = self.and_comparison()
            and_comparison = BinaryOperator(and_comparison, token.type, right)
        
        return and_comparison
            
    
    def and_comparison(self):
        equality = self.equality()

        while self.match_token(TokType.AND):
            token = self.get_previous()
            right = self.equality()
            equality = BinaryOperator(equality, token.type, right)
        
        return equality

            
    def equality(self):
        comparison = self.comparison()

        while self.match_token(TokType.EQUAL, TokType.NOT_EQUAL):
            token = self.get_previous()
            right = self.comparison()
            comparison = BinaryOperator(comparison, token.type, right)
        
        return comparison

    
    def comparison(self):
        exp = self.add_expression()

        while self.match_token(TokType.LESS, TokType.LESS_OR_EQ, TokType.GREATER, TokType.GREATER_OR_EQ):
            token = self.get_previous()
            right = self.add_expression()
            exp = BinaryOperator(exp, token.type, right)
        
        return exp


    def add_expression(self):
        term = self.term()

        while self.match_token(TokType.ADDITION, TokType.MINUS):
            token = self.get_previous()
            right = self.term()
            term = BinaryOperator(term, token.type, right)

        return term
            

    def term(self):
        factor = self.factor()

        while self.match_token(TokType.MULTIPLICATION, TokType.DIVISION, TokType.MODULO):
            token = self.get_previous()
            right = self.factor()
            factor = BinaryOperator(factor, token.type, right)

        return factor


    def factor(self):
        if self.match_token(TokType.OPEN_PAREN):
            exp = self.or_comparison()

            if not self.match_token(TokType.CLOSE_PAREN):
                self.error("No closing parenthesis")

            return exp

        if self.match_token(TokType.INT_LITERAL):
            value = self.get_previous().value
            return Constant(value)
        
        if self.match_token(TokType.IDENTIFIER):
            identifier = self.get_previous()
            return Variable(identifier)

        if self.match_token(TokType.BITW_COMPLIMENT, TokType.MINUS, TokType.LOGIC_NEGATION):
            op = self.get_previous()
            exp = self.factor()

            return UnaryOperator(op.type, exp)
        
        self.error("Failed on token: " + self.get_current().type.name + " Index: " + str(self.current))



class Compiler():
    def __init__(self, ast: SyntaxTree) -> None:
        self.ast = ast


    def compile(self):
        file = open("assembly.asm", "w")
        file.write("section .text\n\nglobal _start\n\n")

        root = self.ast.root

        text = root.entry.compile()
        file.write(text)

        file.write("_start:\n\tcall main\n\tmov rdi, rax\n\tmov rax, 60\n\tsyscall")

        file.close()

        subprocess.run(["nasm", "-g", "-felf64", "assembly.asm"])
        subprocess.run(["ld", "-o", "assembly", "assembly.o"])


def lex(file: TextIO) -> List[Token]:
    output = list()
    operators = ["{", "}", "(", ")", ";", "-", "~", "!", "+", "*", "/", "<", ">", "%", "="]
    operators_two = ["&&", "||", "==", "!=", "<=", ">="]
    keywords = ["return", "int", "if", "else"]

    for line in file:
        line = line.strip("\n")
        line = line.lstrip()

        while len(line) > 0:
            cursor = 0
            token_type = None
            value = None

            if line[cursor].isnumeric():
                token_type = TokType.INT_LITERAL

                while len(line) > cursor + 1:
                    if not line[cursor+1].isnumeric():
                        break
                    cursor += 1
                
                value = int(line[0:cursor+1]) 
            elif line[cursor].isalpha():
                word = re.search("[a-zA-Z]\\w*", line)

                if word is None:
                    line = line[cursor+1:]
                    continue
                
                cursor = word.span()[1] - 1

                value = word.group().lower()

                if value in keywords:
                    token_type = TokType.KEYWORD
                else:
                    token_type = TokType.IDENTIFIER
            elif line[cursor] == ' ':
                line = line[cursor+1:]
                continue
            elif line[cursor:cursor+2] in operators_two:
                index = operators_two.index(line[cursor:cursor+2])
                cursor += 1
                token_type = TokType(len(operators) + index + 1)
            elif line[cursor] in operators:
                index = operators.index(line[cursor])
                token_type = TokType(index+1)

            if token_type is None:
                print(line)

            assert token_type is not None, "Unknown token"

            output.append(Token(token_type, value))
            line = line[cursor+1:]

    return output


file = open("return_2.c", "r")

tokens = lex(file)
print(tokens)

file.close()

ast = SyntaxTree(tokens)

com = Compiler(ast)
com.compile()

