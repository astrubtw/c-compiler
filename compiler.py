from sys import argv
from typing import TextIO, List
import re
import subprocess
from tokenclass import TokType, Token, convert_to_char

from astnodes import (
    Break,
    Constant,
    Continue,
    DoLoop,
    For,
    ForDecl,
    FunctionCall,
    UnaryOperator,
    BinaryOperator,
    Return,
    Declare,
    Variable,
    Assign,
    Conditional,
    Compound,
    Function,
    Program,
    While,
)


class SyntaxTree:
    class ParsingError(Exception):
        """Error while parsing"""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.current = 0
        self.parse()

    def parse(self):
        self.root = self.program()

    def error(self, msg: str):
        raise self.ParsingError(msg)

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
            self.error(
                "Expected token at line "
                + str(self.get_current().line)
                + ": "
                + convert_to_char(token)
            )

    def check_token(self, type: TokType) -> bool:
        return self.get_current().type == type

    def check_keyword(self):
        if self.get_current().type == TokType.KEYWORD:
            return self.get_current().value

        return None

    def check_keyword_val(self, value: str) -> bool:
        if self.get_current().type == TokType.KEYWORD:
            return self.get_current().value == value

        return False

    def program(self) -> Program:
        functions = list()

        while not self.match_token(TokType.EOF):
            functions.append(self.function())

        return Program(functions)

    def function_args(self):
        self.consume(TokType.OPEN_PAREN)

        args = list()

        if self.match_token(TokType.CLOSE_PAREN):
            return args

        self.check_keyword_val("int")
        self.consume(TokType.KEYWORD)

        self.consume(TokType.IDENTIFIER)
        args.append(self.get_previous().value)

        while self.match_token(TokType.COMMA):
            self.check_keyword_val("int")
            self.consume(TokType.KEYWORD)

            self.consume(TokType.IDENTIFIER)
            args.append(self.get_previous().value)

        self.consume(TokType.CLOSE_PAREN)
        return args

    def function(self) -> Function:
        self.consume(TokType.KEYWORD)

        self.consume(TokType.IDENTIFIER)
        identifier = self.get_previous()

        args = self.function_args()
        print(str(identifier.value) + ": " + str(args))

        self.consume(TokType.OPEN_BRACE)

        block = list()

        while not self.check_token(TokType.CLOSE_BRACE):
            block.append(self.block_item())

        self.consume(TokType.CLOSE_BRACE)

        body = Compound(block, True)

        return Function(str(identifier.value), body, args)

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

                case "for":
                    self.consume(TokType.OPEN_PAREN)

                    declaration = False
                    first_exp = None

                    if self.check_keyword_val("int"):
                        self.consume(TokType.KEYWORD)
                        first_exp = self.declaration()
                        declaration = True
                    else:
                        first_exp = self.expression_option()
                        self.consume(TokType.SEMICOLON)

                    second_exp = self.expression_option()
                    self.consume(TokType.SEMICOLON)

                    third_exp = self.expression_option()

                    self.consume(TokType.CLOSE_PAREN)

                    inner = self.statement()

                    if declaration:
                        return ForDecl(first_exp, second_exp, third_exp, inner)

                    return For(first_exp, second_exp, third_exp, inner)

                case "while":
                    self.consume(TokType.OPEN_PAREN)
                    exp = self.expression()
                    self.consume(TokType.CLOSE_PAREN)

                    inner = self.statement()

                    return While(exp, inner)

                case "do":
                    inner = self.statement()

                    if not self.check_keyword_val("while"):
                        self.error("Missing while keyword")

                    self.consume(TokType.KEYWORD)

                    self.consume(TokType.OPEN_PAREN)
                    exp = self.expression()
                    self.consume(TokType.CLOSE_PAREN)
                    self.consume(TokType.SEMICOLON)

                    return DoLoop(exp, inner)

                case "break":
                    self.consume(TokType.SEMICOLON)
                    return Break()

                case "continue":
                    self.consume(TokType.SEMICOLON)
                    return Continue()

        if self.match_token(TokType.OPEN_BRACE):
            block = list()

            while not self.check_token(TokType.CLOSE_BRACE):
                block.append(self.block_item())

            self.match_token(TokType.CLOSE_BRACE)

            return Compound(block, False)

        exp = self.expression_option()
        self.consume(TokType.SEMICOLON)

        return exp

    def expression_option(self):
        if (
            self.get_current().type == TokType.SEMICOLON
            or self.get_current().type == TokType.CLOSE_PAREN
        ):
            return Constant(1)

        return self.expression()

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

        while self.match_token(
            TokType.LESS, TokType.LESS_OR_EQ, TokType.GREATER, TokType.GREATER_OR_EQ
        ):
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

        while self.match_token(
            TokType.MULTIPLICATION, TokType.DIVISION, TokType.MODULO
        ):
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

            if self.match_token(TokType.OPEN_PAREN):
                return self.function_call(identifier)

            return Variable(identifier)

        if self.match_token(
            TokType.BITW_COMPLIMENT, TokType.MINUS, TokType.LOGIC_NEGATION
        ):
            op = self.get_previous()
            exp = self.factor()

            return UnaryOperator(op.type, exp)

        self.error(
            "Failed on token: "
            + self.get_current().type.name
            + " Index: "
            + str(self.current)
        )

    def function_call(self, identifier):
        args = list()

        if self.match_token(TokType.CLOSE_PAREN):
            return FunctionCall(str(identifier.value), args)

        args.append(self.expression())

        while self.match_token(TokType.COMMA):
            args.append(self.expression())

        self.consume(TokType.CLOSE_PAREN)
        # print(args)
        return FunctionCall(str(identifier.value), args)


def compile_tree(ast: SyntaxTree):
    code = ""
    failed = False

    try:
        code = ast.root.compile()
    except Exception as e:
        print(e.__class__)
        failed = True

    if failed:
        return

    file = open("assembly.asm", "w")
    file.write("section .text\n\nglobal _start\n\n")

    file.write(code)

    file.write("_start:\n\tcall main\n\tmov rdi, rax\n\tmov rax, 60\n\tsyscall")
    file.close()

    asmcomp = subprocess.run(["nasm", "-g", "-felf64", "assembly.asm"])

    if asmcomp.returncode != 0:
        return

    subprocess.run(["ld", "-o", "assembly", "assembly.o"])

    program = subprocess.run("./assembly")
    print("Return code: " + str(program.returncode))
    return program.returncode


def lex(file: TextIO) -> List[Token]:
    output = list()
    operators = [
        "{",
        "}",
        "(",
        ")",
        ";",
        "-",
        "~",
        "!",
        "+",
        "*",
        "/",
        "<",
        ">",
        "%",
        "=",
        ",",
    ]
    operators_two = ["&&", "||", "==", "!=", "<=", ">="]
    keywords = [
        "return",
        "int",
        "if",
        "else",
        "for",
        "while",
        "do",
        "break",
        "continue",
    ]

    for n, line in enumerate(file):
        line = line.strip("\n")
        line = line.lstrip()

        while len(line) > 0:
            cursor = 0
            token_type = None
            value = None

            if line[cursor].isnumeric():
                token_type = TokType.INT_LITERAL

                while len(line) > cursor + 1:
                    if not line[cursor + 1].isnumeric():
                        break
                    cursor += 1

                value = int(line[0 : cursor + 1])
            elif line[cursor].isalpha():
                word = re.search("[a-zA-Z]\\w*", line)

                if word is None:
                    line = line[cursor + 1 :]
                    continue

                cursor = word.span()[1] - 1

                value = word.group().lower()

                if value in keywords:
                    token_type = TokType.KEYWORD
                else:
                    token_type = TokType.IDENTIFIER
            elif line[cursor] == " ":
                line = line[cursor + 1 :]
                continue
            elif line[cursor : cursor + 2] in operators_two:
                index = operators_two.index(line[cursor : cursor + 2])
                cursor += 1
                token_type = TokType(len(operators) + index + 1)
            elif line[cursor] in operators:
                index = operators.index(line[cursor])
                token_type = TokType(index + 1)

            if token_type is None:
                print(line)

            assert token_type is not None, "Unknown token"

            output.append(Token(token_type, n, value))
            line = line[cursor + 1 :]

    output.append(Token(TokType.EOF, -1))

    return output


def compile_file(name: str):
    file = open(name, "r")

    tokens = lex(file)

    file.close()

    ast = SyntaxTree(tokens)

    compile_tree(ast)


if __name__ == "__main__":
    pattern = re.compile(r"^.*\.c$", re.IGNORECASE)

    if len(argv) <= 1:
        print("Please specify .c file for compilation")
        exit(1)

    if not pattern.match(argv[1]):
        print("This is not a .c file")
        exit(1)

    compile_file(argv[1])
