
from enum import Enum, auto


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
    COMMA = auto()
    
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
    EOF = auto()


class Token():
    def __init__(self, type_: TokType, value: int | str = None) -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        if self.value:
            return f"{self.type.name}[{self.value}]"
        
        return f"{self.type.name}"
