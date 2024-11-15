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
    NEGATION = auto()
    BITW_COMPLIMENT = auto()
    LOGIC_NEGATION = auto()

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


class UnaryOperator():
    pass


class Expression():
    def __init__(self, value: UnaryOperator | Constant) -> None:
        self.value = value

    def compile(self) -> str:
        if isinstance(self.value, Constant):
            return f"\tmov rax, {self.value.value}\n"

        res = self.value.compile()
        return res



class UnaryOperator():
    def __init__(self, op: TokType, exp: Expression) -> None:
        self.op = op
        self.exp = exp


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
        #tokens.insert(0, Token(TokType.SEMICOLON))
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
        
        self.error("Unexpected token")
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
            
    
    def expression(self) -> Expression | UnaryOperator:
        if self.check_token(TokType.INT_LITERAL):
            self.advance()
            value = self.get_previous().value
            return Expression(Constant(value))
        
        self.match_token(TokType.BITW_COMPLIMENT, TokType.NEGATION, TokType.LOGIC_NEGATION)
        op = self.get_previous()
        exp = self.expression()

        return UnaryOperator(op, exp)
    

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
    operators = ["{", "}", "(", ")", ";", "-", "~", "!"]
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

ast = SyntaxTree(tokens)

com = Compiler(ast)
com.compile()

file.close()