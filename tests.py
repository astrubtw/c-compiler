import unittest
import compiler


class CompilerTest(unittest.TestCase):
    def test_variables(self):
        with open("test1.c", "r") as file:
            tokens = compiler.lex(file)

            ast = compiler.SyntaxTree(tokens)

            returncode = compiler.compile_tree(ast)

            self.assertEqual(returncode, 33)

    def test_recursive_func(self):
        with open("test2.c", "r") as file:
            tokens = compiler.lex(file)

            ast = compiler.SyntaxTree(tokens)

            returncode = compiler.compile_tree(ast)

            self.assertEqual(returncode, 55)
