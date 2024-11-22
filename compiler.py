from typing import TextIO, List
from enum import Enum, auto
import re
import subprocess


class TokType(Enum):
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

    KEYWORD = auto()
    IDENTIFIER = auto()
    INT_LITERAL = auto()
    

class Token():
    def __init__(self, type_: TokType, value: int | str = None) -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        if self.value:
            return f"{self.type.name}[{self.value}]"
        
        return f"{self.type.name}"


class Statement():
    pass


class Constant():
    def __init__(self, value: int) -> None:
        self.value = value


class Expression():
    def __init__(self, value: Constant) -> None:
        self.value = value

    def compile(self) -> str:
        return f"\tmov rax, {self.value.value}\n"


class UnaryOperator():
    def __init__(self, op: TokType, exp: Expression) -> None:
        self.op = op
        self.exp = exp

    def compile(self) -> str:
        res = self.exp.compile()

        if self.op == TokType.MINUS:
            res += "\tneg rax\n"
        if self.op == TokType.BITW_COMPLIMENT:
            res += "\tnot rax\n"
        if self.op == TokType.LOGIC_NEGATION:
            res += "\ttest rax, rax\n\tmov rax, 0\n\tsetz al\n"
        
        return res


class BinaryOperator():
    def __init__(self, exp: Expression, op: TokType, right: Expression) -> None:
        self.exp = exp
        self.op = op
        self.right = right

    def compile(self) -> str:
        res = self.exp.compile()
        res += "\tmov rbx, rax\n"
        res += self.right.compile()

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

        return res

class Return(Statement):
    def __init__(self, exp: Expression) -> None:
        self.exp = exp

    def compile(self) -> str:
        res = self.exp.compile()

        return res


class Function():
    def __init__(self, name: str, body: List[Statement]) -> None:
        self.name = name
        self.body = body


    def compile(self) -> str:
        res = f"{self.name}:\n"

        for statement in self.body:
            res += statement.compile()

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

    def error(self, msg: str):
        raise Exception(msg)


    def advance(self):
        self.current += 1


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
    

    def check_token(self, type: TokType) -> bool:
        return self.get_current().type == type

    
    def program(self) -> Program:
        return Program(self.function())
    

    def function(self) -> Function:
        self.match_token(TokType.KEYWORD)

        self.match_token(TokType.IDENTIFIER)
        identifier = self.get_previous()

        self.match_token(TokType.OPEN_PAREN)
        self.match_token(TokType.CLOSE_PAREN)

        self.match_token(TokType.OPEN_BRACE)

        body = list()

        while not self.check_token(TokType.CLOSE_BRACE):
            body.append(self.statement())

        self.match_token(TokType.CLOSE_BRACE)

        return Function(identifier.value, body)
    

    def statement(self) -> Statement:
        self.match_token(TokType.KEYWORD)
        keyword = self.get_previous()

        match keyword.value:
            case "return":
                exp = self.expression()
                self.match_token(TokType.SEMICOLON)

                return Return(exp)


    def expression(self) -> Expression:
        exp = self.term()

        while self.match_token(TokType.ADDITION, TokType.MINUS):
            token = self.get_previous()
            right = self.term()
            exp = BinaryOperator(exp, token.type, right)

        return exp
            

    def term(self) -> Expression:
        factor = self.factor()

        while self.match_token(TokType.MULTIPLICATION, TokType.DIVISION):
            token = self.get_previous()
            right = self.factor()
            factor = BinaryOperator(factor, token.type, right)

        return factor


    def factor(self) -> Expression:
        if self.match_token(TokType.OPEN_PAREN):
            exp = self.expression()

            if not self.match_token(TokType.CLOSE_PAREN):
                self.error("No closing parenthesis")

            return exp

        if self.match_token(TokType.INT_LITERAL):
            value = self.get_previous().value
            return Expression(Constant(value))
        
        if self.match_token(TokType.BITW_COMPLIMENT, TokType.MINUS, TokType.LOGIC_NEGATION):
            op = self.get_previous()
            exp = self.factor()

            return UnaryOperator(op.type, exp)
        
        self.error()
    

    def parse(self):
        self.root = self.program()


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

        subprocess.run(["nasm", "-felf64", "assembly.asm"])
        subprocess.run(["ld", "-o", "assembly", "assembly.o"])


def lex(file: TextIO) -> List[Token]:
    output = list()
    operators = ["{", "}", "(", ")", ";", "-", "~", "!", "+", "*", "/"]
    keywords = ["return", "int"]

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
            elif line[cursor] in operators:
                index = operators.index(line[cursor])
                token_type = TokType(index+1)
            else:
                word = re.search("[a-zA-Z]\w*", line)

                if word is None:
                    line = line[cursor+1:]
                    continue
                
                cursor = word.span()[1] - 1

                value = word.group().lower()

                if value in keywords:
                    token_type = TokType.KEYWORD
                else:
                    token_type = TokType.IDENTIFIER


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

