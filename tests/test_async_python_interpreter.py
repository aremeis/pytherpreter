# coding=utf-8
# Copyright 2025 Are Meisfjord. All rights reserved.
# Portions Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import io
import types
import unittest
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from pytherpreter.python_interpreter import (
    BASE_BUILTIN_FUNCTIONS,
    ClientError,
    InterpreterError,
    check_module_authorized,
    get_safe_module,
)
from pytherpreter.async_python_interpreter import (
    AsyncPythonInterpreter,
    evaluate_condition,
    evaluate_delete,
    async_evaluate,
)


# Fake function we will use as tool
def add_two(x):
    return x + 2


class TestAsyncPythonInterpreter():
    def assertDictEqual(self, dict1, dict2):
        assert dict1 == dict2, f"Dictionaries not equal:\n{dict1}\n!=\n{dict2}"
        
    def assertDictEqualNoPrint(self, dict1, dict2):
        filtered1 = {k: v for k, v in dict1.items() if k != "_print_outputs"}
        filtered2 = {k: v for k, v in dict2.items() if k != "_print_outputs"}
        assert filtered1 == filtered2, f"Dictionaries not equal:\n{filtered1}\n!=\n{filtered2}"

    def assertListEqual(self, list1, list2):
        assert list1 == list2, f"Lists not equal:\n{list1}\n!=\n{list2}"

    # @pytest.mark.asyncio
    # async def test_fail(self):
    #     assert False

    @pytest.mark.asyncio
    async def test_evaluate_assign(self):
        code = "x = 3"
        state = {}
        result = await async_evaluate(code, {}, variables=state)
        assert result == 3
        self.assertDictEqualNoPrint(state, {"x": 3, "_operations_count": 2})

        code = "x = y"
        state = {"y": 5}
        result = await async_evaluate(code, {}, variables=state)
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 5, "y": 5, "_operations_count": 2})

        code = "a=1;b=None"
        result = await async_evaluate(code, {}, variables={})
        assert result is None

        state = {}
        code = "x = y = 1"
        result = await async_evaluate(code, {}, variables=state)
        assert result == 1
        self.assertDictEqualNoPrint(state, {"x": 1, "y": 1, "_operations_count": 2})

    @pytest.mark.asyncio
    async def test_assignment_cannot_overwrite_tool(self):
        code = "print = '3'"
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code, {"print": print}, variables={})
        assert "Cannot assign to name 'print': doing this would erase the existing function!" in str(e)

    @pytest.mark.asyncio
    async def test_subscript_call(self):
        code = """def foo(x,y):return x*y\n\ndef boo(y):\n\treturn y**3\nfun = [foo, boo]\nresult_foo = fun[0](4,2)\nresult_boo = fun[1](4)"""
        state = {}
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables=state)
        assert result == 64
        assert state["result_foo"] == 8
        assert state["result_boo"] == 64

    @pytest.mark.asyncio
    async def test_evaluate_call(self):
        code = "y = add_two(x)"
        state = {"x": 3}
        result = await async_evaluate(code, {"add_two": add_two}, variables=state)
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 5, "_operations_count": 4})

        # Should not work without the tool
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code, {}, variables=state)
        assert "The name `add_two` is not defined" in str(e.value)

    @pytest.mark.asyncio
    async def test_evaluate_constant(self):
        code = "x = 3"
        state = {}
        result = await async_evaluate(code, {}, variables=state)
        assert result == 3
        self.assertDictEqualNoPrint(state, {"x": 3, "_operations_count": 2})

    @pytest.mark.asyncio
    async def test_evaluate_dict(self):
        code = "test_dict = {'x': x, 'y': add_two(x)}"
        state = {"x": 3}
        result = await async_evaluate(code, {"add_two": add_two}, variables=state)
        self.assertDictEqual(result, {"x": 3, "y": 5})
        self.assertDictEqualNoPrint(state, {"x": 3, "test_dict": {"x": 3, "y": 5}, "_operations_count": 8})

    @pytest.mark.asyncio
    async def test_evaluate_expression(self):
        code = "x = 3\ny = 5"
        state = {}
        result = await async_evaluate(code, {}, variables=state)
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 5, "_operations_count": 4})

    @pytest.mark.asyncio
    async def test_evaluate_f_string(self):
        code = "text = f'This is x: {x}.'"
        state = {"x": 3}
        result = await async_evaluate(code, {}, variables=state)
        assert result == "This is x: 3."
        self.assertDictEqualNoPrint(state, {"x": 3, "text": "This is x: 3.", "_operations_count": 6})

    @pytest.mark.asyncio
    async def test_evaluate_if(self):
        code = "if x <= 3:\n    y = 2\nelse:\n    y = 5"
        state = {"x": 3}
        result = await async_evaluate(code, {}, variables=state)
        assert result == 2
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 2, "_operations_count": 6})

        state = {"x": 8}
        result = await async_evaluate(code, {}, variables=state)
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 8, "y": 5, "_operations_count": 6})

    @pytest.mark.asyncio
    async def test_evaluate_list(self):
        code = "test_list = [x, add_two(x)]"
        state = {"x": 3}
        result = await async_evaluate(code, {"add_two": add_two}, variables=state)
        self.assertListEqual(result, [3, 5])
        self.assertDictEqualNoPrint(state, {"x": 3, "test_list": [3, 5], "_operations_count": 6})

    @pytest.mark.asyncio
    async def test_evaluate_name(self):
        code = "y = x"
        state = {"x": 3}
        result = await async_evaluate(code, {}, variables=state)
        assert result == 3
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 3, "_operations_count": 2})

    @pytest.mark.asyncio
    async def test_evaluate_subscript(self):
        code = "test_list = [x, add_two(x)]\ntest_list[1]"
        state = {"x": 3}
        result = await async_evaluate(code, {"add_two": add_two}, variables=state)
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 3, "test_list": [3, 5], "_operations_count": 10})

        code = "test_dict = {'x': x, 'y': add_two(x)}\ntest_dict['y']"
        state = {"x": 3}
        result = await async_evaluate(code, {"add_two": add_two}, variables=state)
        assert result == 5
        self.assertDictEqualNoPrint(state, {"x": 3, "test_dict": {"x": 3, "y": 5}, "_operations_count": 12})

        code = "vendor = {'revenue': 31000, 'rent': 50312}; vendor['ratio'] = round(vendor['revenue'] / vendor['rent'], 2)"
        state = {}
        await async_evaluate(code, {"min": min, "print": print, "round": round}, variables=state)
        assert state["vendor"] == {"revenue": 31000, "rent": 50312, "ratio": 0.62}

    @pytest.mark.asyncio
    async def test_subscript_string_with_string_index_raises_appropriate_error(self):
        code = """
search_results = "[{'title': 'Paris, Ville de Paris, France Weather Forecast | AccuWeather', 'href': 'https://www.accuweather.com/en/fr/paris/623/weather-forecast/623', 'body': 'Get the latest weather forecast for Paris, Ville de Paris, France , including hourly, daily, and 10-day outlooks. AccuWeather provides you with reliable and accurate information on temperature ...'}]"
for result in search_results:
    if 'current' in result['title'].lower() or 'temperature' in result['title'].lower():
        current_weather_url = result['href']
        print(current_weather_url)
        break"""
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
            assert "You're trying to subscript a string with a string index" in e

    @pytest.mark.asyncio
    async def test_evaluate_for(self):
        code = "x = 0\nfor i in range(3):\n    x = i"
        state = {}
        result = await async_evaluate(code, {"range": range}, variables=state)
        assert result == 2
        self.assertDictEqualNoPrint(state, {"x": 2, "i": 2, "_operations_count": 12})

    @pytest.mark.asyncio
    async def test_evaluate_binop(self):
        code = "y + x"
        state = {"x": 3, "y": 6}
        result = await async_evaluate(code, {}, variables=state)
        assert result == 9
        self.assertDictEqualNoPrint(state, {"x": 3, "y": 6, "_operations_count": 4})

    @pytest.mark.asyncio
    async def test_recursive_function(self):
        code = """
def recur_fibo(n):
    if n <= 1:
        return n
    else:
        return(recur_fibo(n-1) + recur_fibo(n-2))
recur_fibo(6)"""
        result = await async_evaluate(code, {}, variables={})
        assert result == 8

    @pytest.mark.asyncio
    async def test_evaluate_string_methods(self):
        code = "'hello'.replace('h', 'o').split('e')"
        result = await async_evaluate(code, {}, variables={})
        assert result == ["o", "llo"]

    @pytest.mark.asyncio
    async def test_evaluate_slicing(self):
        code = "'hello'[1:3][::-1]"
        result = await async_evaluate(code, {}, variables={})
        assert result == "le"

    @pytest.mark.asyncio
    async def test_access_attributes(self):
        code = "integer = 1\nobj_class = integer.__class__\nobj_class"
        result = await async_evaluate(code, {}, variables={})
        assert result is int

    @pytest.mark.asyncio
    async def test_list_comprehension(self):
        code = "sentence = 'THESEAGULL43'\nmeaningful_sentence = '-'.join([char.lower() for char in sentence if char.isalpha()])"
        result = await async_evaluate(code, {}, variables={})
        assert result == "t-h-e-s-e-a-g-u-l-l"

    @pytest.mark.asyncio
    async def test_string_indexing(self):
        code = """text_block = [
    "THESE",
    "AGULL"
]
sentence = ""
for block in text_block:
    for col in range(len(text_block[0])):
        sentence += block[col]
        """
        result = await async_evaluate(code, {"len": len, "range": range}, variables={})
        assert result == "THESEAGULL"

    @pytest.mark.asyncio
    async def test_tuples(self):
        code = "x = (1, 2, 3)\nx[1]"
        result = await async_evaluate(code, {}, variables={})
        assert result == 2

        code = """
digits, i = [1, 2, 3], 1
digits[i], digits[i + 1] = digits[i + 1], digits[i]"""
        await async_evaluate(code, {"range": range, "print": print, "int": int}, {})

        code = """
def calculate_isbn_10_check_digit(number):
    total = sum((10 - i) * int(digit) for i, digit in enumerate(number))
    remainder = total % 11
    check_digit = 11 - remainder
    if check_digit == 10:
        return 'X'
    elif check_digit == 11:
        return '0'
    else:
        return str(check_digit)

# Given 9-digit numbers
numbers = [
    "478225952",
    "643485613",
    "739394228",
    "291726859",
    "875262394",
    "542617795",
    "031810713",
    "957007669",
    "871467426"
]

# Calculate check digits for each number
check_digits = [calculate_isbn_10_check_digit(number) for number in numbers]
print(check_digits)
"""
        state = {}
        await async_evaluate(
            code,
            {
                "range": range,
                "print": print,
                "sum": sum,
                "enumerate": enumerate,
                "int": int,
                "str": str,
            },
            state,
        )

    @pytest.mark.asyncio
    async def test_listcomp(self):
        code = "x = [i for i in range(3)]"
        result = await async_evaluate(code, {"range": range}, variables={})
        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_generatorcomp(self):
        # Test regular generator expression
        code = "x = next(2 / i for i in reversed(range(3)))"
        result = await async_evaluate(code, {"range": range, "next": next, "reversed": reversed}, variables={})
        assert result == 1

        code = "x = next(a for a, b in [(1, 2), (3, 4)] if b > 2)"
        result = await async_evaluate(code, {"next": next}, variables={})
        assert result == 3

        data = [{"a": 1, "b": [1, 2, 3]}, {"a": 3, "b": [4, 5, 6]}]
        code = "sum(x for element in data for x in element['b'])"
        result = await async_evaluate(code, {"sum": sum}, variables={"data": data})
        assert result == 21

    @pytest.mark.asyncio
    async def test_async_for(self):
        # Test async generator expression
        async def gen():
            for i in reversed(range(3)):
                yield i
        code = dedent(
            """\
            async for i in gen():
                x = 2 / i
                break
            """)
        result = await async_evaluate(code, {"gen": gen}, variables={})
        assert result == 1

    @pytest.mark.asyncio
    async def test_break_continue(self):
        code = "for i in range(10):\n    if i == 5:\n        break\ni"
        result = await async_evaluate(code, {"range": range}, variables={})
        assert result == 5

        code = "for i in range(10):\n    if i == 5:\n        continue\ni"
        result = await async_evaluate(code, {"range": range}, variables={})
        assert result == 9

    @pytest.mark.asyncio
    async def test_call_int(self):
        code = "import math\nstr(math.ceil(149))"
        result = await async_evaluate(code, {"str": lambda x: str(x)}, variables={})
        assert result == "149"

    @pytest.mark.asyncio
    async def test_lambda(self):
        code = "f = lambda x: x + 2\nf(3)"
        result = await async_evaluate(code, {}, variables={})
        assert result == 5

    @pytest.mark.asyncio
    async def test_dictcomp(self):
        code = "x = {i: i**2 for i in range(3)}"
        result = await async_evaluate(code, {"range": range}, variables={})
        assert result == {0: 0, 1: 1, 2: 4}

        code = "{num: name for num, name in {101: 'a', 102: 'b'}.items() if name not in ['a']}"
        result = await async_evaluate(code, {"print": print}, variables={}, authorized_imports=["pandas"])
        assert result == {102: "b"}

        code = """
shifts = {'A': ('6:45', '8:00'), 'B': ('10:00', '11:45')}
shift_minutes = {worker: ('a', 'b') for worker, (start, end) in shifts.items()}
"""
        result = await async_evaluate(code, {}, variables={})
        assert result == {"A": ("a", "b"), "B": ("a", "b")}

    @pytest.mark.asyncio
    async def test_tuple_assignment(self):
        code = "a, b = 0, 1\nb"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == 1

    @pytest.mark.asyncio
    async def test_while(self):
        code = "i = 0\nwhile i < 3:\n    i += 1\ni"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == 3

        # test infinite loop
        code = "i = 0\nwhile i < 3:\n    i -= 1\ni"
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert "iterations in While loop exceeded" in str(e)

        # test lazy evaluation
        code = """
house_positions = [0, 7, 10, 15, 18, 22, 22]
i, n, loc = 0, 7, 30
while i < n and house_positions[i] <= loc:
    i += 1
"""
        state = {}
        await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables=state)

    @pytest.mark.asyncio
    async def test_generator(self):
        code = "a = [1, 2, 3, 4, 5]; b = (i**2 for i in a); list(b)"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == [1, 4, 9, 16, 25]

    @pytest.mark.asyncio
    async def test_boolops(self):
        code = """if (not (a > b and a > c)) or d > e:
    best_city = "Brooklyn"
else:
    best_city = "Manhattan"
    best_city
    """
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        assert result == "Brooklyn"

        code = """if d > e and a < b:
    best_city = "Brooklyn"
elif d < e and a < b:
    best_city = "Sacramento"
else:
    best_city = "Manhattan"
    best_city
    """
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        assert result == "Sacramento"

    @pytest.mark.asyncio
    async def test_if_conditions(self):
        code = """char='a'
if char.isalpha():
    print('2')"""
        state = {}
        stdout = io.StringIO()
        await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables=state, stdout=stdout)
        assert stdout.getvalue() == "2\n"

    @pytest.mark.asyncio
    async def test_imports(self):
        code = "import math\nmath.sqrt(4)"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == 2.0

        code = "from random import choice, seed\nseed(12)\nchoice(['win', 'lose', 'draw'])"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == "lose"

        code = "import time, re\ntime.sleep(0.1)"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result is None

        code = "from queue import Queue\nq = Queue()\nq.put(1)\nq.get()"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == 1

        code = "import itertools\nlist(itertools.islice(range(10), 3))"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == [0, 1, 2]

        code = "import re\nre.search('a', 'abc').group()"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == "a"

        code = "import stat\nstat.S_ISREG(0o100644)"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result

        code = "import statistics\nstatistics.mean([1, 2, 3, 4, 4])"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == 2.8

        code = "import unicodedata\nunicodedata.name('A')"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == "LATIN CAPITAL LETTER A"

        # Test submodules are handled properly, thus not raising error
        code = "import numpy.random as rd\nrng = rd.default_rng(12345)\nrng.random()"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={}, authorized_imports=["numpy"])

        code = "from numpy.random import default_rng as d_rng\nrng = d_rng(12345)\nrng.random()"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={}, authorized_imports=["numpy"])

    @pytest.mark.asyncio
    async def test_additional_imports(self):
        code = "import numpy as np"
        await async_evaluate(code, authorized_imports=["numpy"], variables={})

        code = "import numpy.random as rd"
        await async_evaluate(code, authorized_imports=["numpy.random"], variables={})
        await async_evaluate(code, authorized_imports=["numpy"], variables={})
        await async_evaluate(code, authorized_imports=["*"], variables={})
        with pytest.raises(InterpreterError):
            await async_evaluate(code, authorized_imports=["random"], variables={})

    @pytest.mark.asyncio
    async def test_multiple_comparators(self):
        code = "0 <= -1 < 4 and 0 <= -5 < 4"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert not result

        code = "0 <= 1 < 4 and 0 <= -5 < 4"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert not result

        code = "0 <= 4 < 4 and 0 <= 3 < 4"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert not result

        code = "0 <= 3 < 4 and 0 <= 3 < 4"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result

    @pytest.mark.asyncio
    async def test_print_output(self):
        code = "print('Hello world!')\nprint('Ok no one cares')"
        state = {}
        stdout = io.StringIO()
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables=state, stdout=stdout)
        assert result is None
        assert stdout.getvalue() == "Hello world!\nOk no one cares\n"

        # Test print in function (state copy)
        code = """
print("1")
def function():
    print("2")
function()"""
        state = {}
        stdout = io.StringIO()
        await async_evaluate(code, {}, variables=state, stdout=stdout)
        assert stdout.getvalue() == "1\n2\n"

        # Test print in list comprehension (state copy)
        code = """
print("1")
def function():
    print("2")
[function() for i in range(10)]"""
        stdout = io.StringIO()
        await async_evaluate(code, {"range": range}, variables={}, stdout=stdout)
        assert stdout.getvalue() == "1\n2\n2\n2\n2\n2\n2\n2\n2\n2\n2\n"

    @pytest.mark.asyncio
    async def test_tuple_target_in_iterator(self):
        code = "for a, b in [('Ralf Weikert', 'Austria'), ('Samuel Seungwon Lee', 'South Korea')]:res = a.split()[0]"
        result = await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert result == "Samuel"

    @pytest.mark.asyncio
    async def test_classes(self):
        code = """
class Animal:
    species = "Generic Animal"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sound(self):
        return "The animal makes a sound."

    def __str__(self):
        return f"{self.name}, {self.age} years old"

class Dog(Animal):
    species = "Canine"

    def __init__(self, name, age, breed):
        super().__init__(name, age)
        self.breed = breed

    def sound(self):
        return "The dog barks."

    def __str__(self):
        return f"{self.name}, {self.age} years old, {self.breed}"

class Cat(Animal):
    def sound(self):
        return "The cat meows."

    def __str__(self):
        return f"{self.name}, {self.age} years old, {self.species}"


# Testing multiple instances
dog1 = Dog("Fido", 3, "Labrador")
dog2 = Dog("Buddy", 5, "Golden Retriever")

# Testing method with built-in function
animals = [dog1, dog2, Cat("Whiskers", 2)]
num_animals = len(animals)

# Testing exceptions in methods
class ExceptionTest:
    def method_that_raises(self):
        raise ValueError("An error occurred")

try:
    exc_test = ExceptionTest()
    exc_test.method_that_raises()
except ValueError as e:
    exception_message = str(e)


# Collecting results
dog1_sound = dog1.sound()
dog1_str = str(dog1)
dog2_sound = dog2.sound()
dog2_str = str(dog2)
cat = Cat("Whiskers", 2)
cat_sound = cat.sound()
cat_str = str(cat)
    """
        state = {}
        await async_evaluate(
            code,
            {"print": print, "len": len, "super": super, "str": str, "sum": sum},
            variables=state,
        )

        # Assert results
        assert state["dog1_sound"] == "The dog barks."
        assert state["dog1_str"] == "Fido, 3 years old, Labrador"
        assert state["dog2_sound"] == "The dog barks."
        assert state["dog2_str"] == "Buddy, 5 years old, Golden Retriever"
        assert state["cat_sound"] == "The cat meows."
        assert state["cat_str"] == "Whiskers, 2 years old, Generic Animal"
        assert state["num_animals"] == 3
        assert state["exception_message"] == "An error occurred"

    @pytest.mark.asyncio
    async def test_variable_args(self):
        code = """
def var_args_method(self, *args, **kwargs):
    return sum(args) + sum(kwargs.values())

var_args_method(1, 2, 3, x=4, y=5)
"""
        state = {}
        result = await async_evaluate(code, {"sum": sum}, variables=state)
        assert result == 15

    @pytest.mark.asyncio
    async def test_exceptions(self):
        code = """
def method_that_raises(self):
    raise ValueError("An error occurred")

try:
    method_that_raises()
except ValueError as e:
    exception_message = str(e)
    """
        state = {}
        await async_evaluate(
            code,
            {"print": print, "len": len, "super": super, "str": str, "sum": sum},
            variables=state,
        )
        assert state["exception_message"] == "An error occurred"

    @pytest.mark.asyncio
    async def test_print(self):
        code = "print(min([1, 2, 3]))"
        state = {}
        stdout = io.StringIO()
        await async_evaluate(code, {"min": min}, variables=state, stdout=stdout)
        assert stdout.getvalue() == "1\n"

    @pytest.mark.asyncio
    async def test_types_as_objects(self):
        code = "type_a = float(2); type_b = str; type_c = int"
        state = {}
        result = await async_evaluate(code, {"float": float, "str": str, "int": int}, variables=state)
        assert result is int

    @pytest.mark.asyncio
    async def test_tuple_id(self):
        code = """
food_items = {"apple": 2, "banana": 3, "orange": 1, "pear": 1}
unique_food_items = [item for item, count in food_items.items() if count == 1]
"""
        state = {}
        result = await async_evaluate(code, {}, variables=state)
        assert result == ["orange", "pear"]

    @pytest.mark.asyncio
    async def test_nonsimple_augassign(self):
        code = """
counts_dict = {'a': 0}
counts_dict['a'] += 1
counts_list = [1, 2, 3]
counts_list += [4, 5, 6]

class Counter:
    self.count = 0

a = Counter()
a.count += 1
"""
        state = {}
        await async_evaluate(code, {}, variables=state)
        assert state["counts_dict"] == {"a": 1}
        assert state["counts_list"] == [1, 2, 3, 4, 5, 6]
        assert state["a"].count == 1

    @pytest.mark.asyncio
    async def test_adding_int_to_list_raises_error(self):
        code = """
counts = [1, 2, 3]
counts += 1"""
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert "Cannot add non-list value 1 to a list." in str(e)

    @pytest.mark.asyncio
    async def test_error_highlights_correct_line_of_code(self):
        code = """a = 1
b = 2

counts = [1, 2, 3]
counts += 1
b += 1"""
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert "Code execution failed at line 'counts += 1" in str(e)

    @pytest.mark.asyncio
    async def test_error_type_returned_in_function_call(self):
        code = """def error_function():
    raise ValueError("error")

error_function()"""
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code)
        assert "error" in str(e)
        assert "ValueError" in str(e)

    @pytest.mark.asyncio
    async def test_assert(self):
        code = """
assert 1 == 1
assert 1 == 2
"""
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code, BASE_BUILTIN_FUNCTIONS, variables={})
        assert "1 == 2" in str(e) and "1 == 1" not in str(e)

    @pytest.mark.asyncio
    async def test_with_context_manager(self):
        code = """
class SimpleLock:
    def __init__(self):
        self.locked = False

    def __enter__(self):
        self.locked = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.locked = False

lock = SimpleLock()

with lock as l:
    assert l.locked == True

assert lock.locked == False
    """
        state = {}
        tools = {}
        await async_evaluate(code, tools, variables=state)

    @pytest.mark.asyncio
    async def test_default_arg_in_function(self):
        code = """
def f(a, b=333, n=1000):
    return b + n
n = f(1, n=667)
"""
        res = await async_evaluate(code, {}, {})
        assert res == 1000

    @pytest.mark.asyncio
    async def test_set(self):
        code = """
S1 = {'a', 'b', 'c'}
S2 = {'b', 'c', 'd'}
S3 = S1.difference(S2)
S4 = S1.intersection(S2)
"""
        state = {}
        await async_evaluate(code, {}, variables=state)
        assert state["S3"] == {"a"}
        assert state["S4"] == {"b", "c"}

    @pytest.mark.asyncio
    async def test_break(self):
        code = """
i = 0

while True:
    i+= 1
    if i==3:
        break

i"""
        result = await async_evaluate(code, {"print": print, "round": round}, variables={})
        assert result == 3

    @pytest.mark.asyncio
    async def test_return(self):
        # test early returns
        code = """
def add_one(n, shift):
    if True:
        return n + shift
    return n

add_one(1, 1)
"""
        state = {}
        result = await async_evaluate(
            code, {"print": print, "range": range, "ord": ord, "chr": chr}, variables=state
        )
        assert result == 2

        # test returning None
        code = """
def returns_none(a):
    return

returns_none(1)
"""
        state = {}
        result = await async_evaluate(
            code, {"print": print, "range": range, "ord": ord, "chr": chr}, variables=state
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_nested_for_loop(self):
        code = """
all_res = []
for i in range(10):
    subres = []
    for j in range(i):
        subres.append(j)
    all_res.append(subres)

out = [i for sublist in all_res for i in sublist]
out[:10]
"""
        state = {}
        result = await async_evaluate(code, {"print": print, "range": range}, variables=state)
        assert result == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_pandas(self):
        code = """
import pandas as pd

df = pd.DataFrame.from_dict({'SetCount': ['5', '4', '5'], 'Quantity': [1, 0, -1]})

df['SetCount'] = pd.to_numeric(df['SetCount'], errors='coerce')

parts_with_5_set_count = df[df['SetCount'] == 5.0]
parts_with_5_set_count[['Quantity', 'SetCount']].values[1]
"""
        state = {}
        result = await async_evaluate(code, {}, variables=state, authorized_imports=["pandas"])
        assert np.array_equal(result, [-1, 5])

        code = """
import pandas as pd

df = pd.DataFrame.from_dict({"AtomicNumber": [111, 104, 105], "ok": [0, 1, 2]})

# Filter the DataFrame to get only the rows with outdated atomic numbers
filtered_df = df.loc[df['AtomicNumber'].isin([104])]
"""
        result = await async_evaluate(code, {"print": print}, variables={}, authorized_imports=["pandas"])
        assert np.array_equal(result.values[0], [104, 1])

        # Test groupby
        code = """import pandas as pd
data = pd.DataFrame.from_dict([
    {"Pclass": 1, "Survived": 1},
    {"Pclass": 2, "Survived": 0},
    {"Pclass": 2, "Survived": 1}
])
survival_rate_by_class = data.groupby('Pclass')['Survived'].mean()
"""
        result = await async_evaluate(code, {}, variables={}, authorized_imports=["pandas"])
        assert result.values[1] == 0.5

        # Test loc and iloc
        code = """import pandas as pd
data = pd.DataFrame.from_dict([
    {"Pclass": 1, "Survived": 1},
    {"Pclass": 2, "Survived": 0},
    {"Pclass": 2, "Survived": 1}
])
survival_rate_biased = data.loc[data['Survived']==1]['Survived'].mean()
survival_rate_biased = data.loc[data['Survived']==1]['Survived'].mean()
survival_rate_sorted = data.sort_values(by='Survived', ascending=False).iloc[0]
"""
        result = await async_evaluate(code, {}, variables={}, authorized_imports=["pandas"])

    @pytest.mark.asyncio
    async def test_starred(self):
        code = """
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

coords_geneva = (46.1978, 6.1342)
coords_barcelona = (41.3869, 2.1660)

distance_geneva_barcelona = haversine(*coords_geneva, *coords_barcelona)
"""
        result = await async_evaluate(code, {"print": print, "map": map}, variables={}, authorized_imports=["math"])
        assert round(result, 1) == 622395.4

    @pytest.mark.asyncio
    async def test_for(self):
        code = """
shifts = {
    "Worker A": ("6:45 pm", "8:00 pm"),
    "Worker B": ("10:00 am", "11:45 am")
}

shift_intervals = {}
for worker, (start, end) in shifts.items():
    shift_intervals[worker] = end
shift_intervals
"""
        result = await async_evaluate(code, {"print": print, "map": map}, variables={})
        assert result == {"Worker A": "8:00 pm", "Worker B": "11:45 am"}

    @pytest.mark.asyncio
    async def test_syntax_error_points_error(self):
        code = "a = ;"
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code)
        assert "SyntaxError" in str(e)
        assert "     ^" in str(e)

    @pytest.mark.asyncio
    async def test_dangerous_subpackage_access_blocked(self):
        # Direct imports with dangerous patterns should fail
        code = "import random._os"
        with pytest.raises(InterpreterError):
            await async_evaluate(code)

        # Import of whitelisted modules should succeed but dangerous submodules should not exist
        code = "import random;random._os.system('echo bad command passed')"
        with pytest.raises(InterpreterError) as e:
            await async_evaluate(code)
        assert "AttributeError: module 'random' has no attribute '_os'" in str(e)

        code = "import doctest;doctest.inspect.os.system('echo bad command passed')"
        with pytest.raises(InterpreterError):
            await async_evaluate(code, authorized_imports=["doctest"])

    @pytest.mark.asyncio
    async def test_close_matches_subscript(self):
        code = 'capitals = {"Czech Republic": "Prague", "Monaco": "Monaco", "Bhutan": "Thimphu"};capitals["Butan"]'
        with pytest.raises(Exception) as e:
            await async_evaluate(code)
        assert "KeyError: 'Butan'" in str(e).replace("\\", "")

    @pytest.mark.asyncio
    async def test_dangerous_builtins_calls_are_blocked(self):
        unsafe_code = "import os"
        dangerous_code = f"""
exec = callable.__self__.exec
compile = callable.__self__.compile
exec(compile('{unsafe_code}', 'no filename', 'exec'))
"""

        with pytest.raises(InterpreterError):
            await async_evaluate(unsafe_code, builtin_functions=BASE_BUILTIN_FUNCTIONS)

        with pytest.raises(InterpreterError):
            await async_evaluate(dangerous_code, builtin_functions=BASE_BUILTIN_FUNCTIONS)

    @pytest.mark.asyncio
    async def test_dangerous_builtins_are_callable_if_explicitly_added(self):
        dangerous_code = """
compile = callable.__self__.compile
eval = callable.__self__.eval
exec = callable.__self__.exec

eval("1 + 1")
exec(compile("1 + 1", "no filename", "exec"))

teval("1 + 1")
texec(tcompile("1 + 1", "no filename", "exec"))
        """

        await async_evaluate(
            dangerous_code, builtin_functions={"tcompile": compile, "teval": eval, "texec": exec} | BASE_BUILTIN_FUNCTIONS
        )

    @pytest.mark.asyncio
    async def test_can_import_os_if_explicitly_authorized(self):
        dangerous_code = "import os; os.listdir('./')"
        await async_evaluate(dangerous_code, authorized_imports=["os"])

    @pytest.mark.asyncio
    async def test_can_import_os_if_all_imports_authorized(self):
        dangerous_code = "import os; os.listdir('./')"
        await async_evaluate(dangerous_code, authorized_imports=["*"])

    @pytest.mark.asyncio
    async def test_comment(self):
        code = dedent(
            """\
            # Empty line comment
            1 + 1 # End of line comment
            """)
        result = await async_evaluate(code, {}, {})
        assert result == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, expected_result",
    [
        (
            dedent("""\
                x = 1
                x += 2
            """),
            3,
        ),
        (
            dedent("""\
                x = "a"
                x += "b"
            """),
            "ab",
        ),
        (
            dedent("""\
                class Custom:
                    def __init__(self, value):
                        self.value = value
                    def __iadd__(self, other):
                        self.value += other * 10
                        return self

                x = Custom(1)
                x += 2
                x.value
            """),
            21,
        ),
    ],
)
async def test_evaluate_augassign(code, expected_result):
    state = {}
    result = await async_evaluate(code, {}, variables=state)
    assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operator, expected_result",
    [
        ("+=", 7),
        ("-=", 3),
        ("*=", 10),
        ("/=", 2.5),
        ("//=", 2),
        ("%=", 1),
        ("**=", 25),
        ("&=", 0),
        ("|=", 7),
        ("^=", 7),
        (">>=", 1),
        ("<<=", 20),
    ],
)
async def test_evaluate_augassign_number(operator, expected_result):
    code = dedent("""\
        x = 5
        x {operator} 2
    """).format(operator=operator)
    state = {}
    result = await async_evaluate(code, {}, variables=state)
    assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operator, expected_result",
    [
        ("+=", 7),
        ("-=", 3),
        ("*=", 10),
        ("/=", 2.5),
        ("//=", 2),
        ("%=", 1),
        ("**=", 25),
        ("&=", 0),
        ("|=", 7),
        ("^=", 7),
        (">>=", 1),
        ("<<=", 20),
    ],
)
async def test_evaluate_augassign_custom(operator, expected_result):
    operator_names = {
        "+=": "iadd",
        "-=": "isub",
        "*=": "imul",
        "/=": "itruediv",
        "//=": "ifloordiv",
        "%=": "imod",
        "**=": "ipow",
        "&=": "iand",
        "|=": "ior",
        "^=": "ixor",
        ">>=": "irshift",
        "<<=": "ilshift",
    }
    code = dedent("""\
        class Custom:
            def __init__(self, value):
                self.value = value
            def __{operator_name}__(self, other):
                self.value {operator} other
                return self

        x = Custom(5)
        x {operator} 2
        x.value
    """).format(operator=operator, operator_name=operator_names[operator])
    state = {}
    result = await async_evaluate(code, {}, variables=state)
    assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, expected_error_message",
    [
        (
            dedent("""\
                x = 5
                del x
                x
            """),
            "The name `x` is not defined",
        ),
        (
            dedent("""\
                x = [1, 2, 3]
                del x[2]
                x[2]
            """),
            "list index out of range",
        ),
        (
            dedent("""\
                x = {"key": "value"}
                del x["key"]
                x["key"]
            """),
            "'key'",
        ),
        (
            dedent("""\
                del x
            """),
            "Cannot delete name 'x': name is not defined",
        ),
    ],
)
async def test_evaluate_python_code_with_evaluate_delete(code, expected_error_message):
    state = {}
    with pytest.raises(InterpreterError) as exception_info:
        await async_evaluate(code, {}, variables=state)
    assert expected_error_message in str(exception_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, state, expectation",
    [
        ("del x", {"x": 1}, {}),
        ("del x[1]", {"x": [1, 2, 3]}, {"x": [1, 3]}),
        ("del x['key']", {"x": {"key": "value"}}, {"x": {}}),
        ("del x", {}, InterpreterError("Cannot delete name 'x': name is not defined")),
    ],
)
async def test_evaluate_delete(code, state, expectation):
    delete_node = ast.parse(code).body[0]
    if isinstance(expectation, Exception):
        with pytest.raises(type(expectation)) as exception_info:
            await evaluate_delete(delete_node, state, {}, {}, [])
        assert str(expectation) in str(exception_info.value)
    else:
        await evaluate_delete(delete_node, state, {}, {}, [])
        _ = state.pop("_operations_count", None)
        assert state == expectation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition, state, expected_result",
    [
        ("a == b", {"a": 1, "b": 1}, True),
        ("a == b", {"a": 1, "b": 2}, False),
        ("a != b", {"a": 1, "b": 1}, False),
        ("a != b", {"a": 1, "b": 2}, True),
        ("a < b", {"a": 1, "b": 1}, False),
        ("a < b", {"a": 1, "b": 2}, True),
        ("a < b", {"a": 2, "b": 1}, False),
        ("a <= b", {"a": 1, "b": 1}, True),
        ("a <= b", {"a": 1, "b": 2}, True),
        ("a <= b", {"a": 2, "b": 1}, False),
        ("a > b", {"a": 1, "b": 1}, False),
        ("a > b", {"a": 1, "b": 2}, False),
        ("a > b", {"a": 2, "b": 1}, True),
        ("a >= b", {"a": 1, "b": 1}, True),
        ("a >= b", {"a": 1, "b": 2}, False),
        ("a >= b", {"a": 2, "b": 1}, True),
        ("a is b", {"a": 1, "b": 1}, True),
        ("a is b", {"a": 1, "b": 2}, False),
        ("a is not b", {"a": 1, "b": 1}, False),
        ("a is not b", {"a": 1, "b": 2}, True),
        ("a in b", {"a": 1, "b": [1, 2, 3]}, True),
        ("a in b", {"a": 4, "b": [1, 2, 3]}, False),
        ("a not in b", {"a": 1, "b": [1, 2, 3]}, False),
        ("a not in b", {"a": 4, "b": [1, 2, 3]}, True),
        # Chained conditions:
        ("a == b == c", {"a": 1, "b": 1, "c": 1}, True),
        ("a == b == c", {"a": 1, "b": 2, "c": 1}, False),
        ("a == b < c", {"a": 2, "b": 2, "c": 2}, False),
        ("a == b < c", {"a": 0, "b": 0, "c": 1}, True),
    ],
)
async def test_evaluate_condition(condition, state, expected_result):
    condition_ast = ast.parse(condition, mode="eval").body
    result = await evaluate_condition(condition_ast, state, {}, {}, [])
    assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition, state, expected_result",
    [
        ("a == b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([False, True, False])),
        ("a != b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([True, False, True])),
        ("a < b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([True, False, False])),
        ("a <= b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([True, True, False])),
        ("a > b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([False, False, True])),
        ("a >= b", {"a": pd.Series([1, 2, 3]), "b": pd.Series([2, 2, 2])}, pd.Series([False, True, True])),
        (
            "a == b",
            {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [1, 2], "y": [3, 5]})},
            pd.DataFrame({"x": [True, True], "y": [True, False]}),
        ),
        (
            "a != b",
            {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [1, 2], "y": [3, 5]})},
            pd.DataFrame({"x": [False, False], "y": [False, True]}),
        ),
        (
            "a < b",
            {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]})},
            pd.DataFrame({"x": [True, False], "y": [False, False]}),
        ),
        (
            "a <= b",
            {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]})},
            pd.DataFrame({"x": [True, True], "y": [False, False]}),
        ),
        (
            "a > b",
            {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]})},
            pd.DataFrame({"x": [False, False], "y": [True, True]}),
        ),
        (
            "a >= b",
            {"a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]})},
            pd.DataFrame({"x": [False, True], "y": [True, True]}),
        ),
    ],
)
async def test_evaluate_condition_with_pandas(condition, state, expected_result):
    condition_ast = ast.parse(condition, mode="eval").body
    result = await evaluate_condition(condition_ast, state, {}, {}, [])
    if isinstance(result, pd.Series):
        pd.testing.assert_series_equal(result, expected_result)
    else:
        pd.testing.assert_frame_equal(result, expected_result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition, state, expected_exception",
    [
        # Chained conditions:
        (
            "a == b == c",
            {
                "a": pd.Series([1, 2, 3]),
                "b": pd.Series([2, 2, 2]),
                "c": pd.Series([3, 3, 3]),
            },
            ValueError(
                "The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."
            ),
        ),
        (
            "a == b == c",
            {
                "a": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                "b": pd.DataFrame({"x": [2, 2], "y": [2, 2]}),
                "c": pd.DataFrame({"x": [3, 3], "y": [3, 3]}),
            },
            ValueError(
                "The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."
            ),
        ),
    ],
)
async def test_evaluate_condition_with_pandas_exceptions(condition, state, expected_exception):
    condition_ast = ast.parse(condition, mode="eval").body
    with pytest.raises(type(expected_exception)) as exception_info:
        _ = await evaluate_condition(condition_ast, state, {}, {}, [])
    assert str(expected_exception) in str(exception_info.value)


@pytest.mark.asyncio
async def test_get_safe_module_handle_lazy_imports():
    class FakeModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.non_lazy_attribute = "ok"

        def __getattr__(self, name):
            if name == "lazy_attribute":
                raise ImportError("lazy import failure")
            return super().__getattr__(name)

        def __dir__(self):
            return super().__dir__() + ["lazy_attribute"]

    fake_module = FakeModule("fake_module")
    safe_module = get_safe_module(fake_module, dangerous_patterns=[], authorized_imports=set())
    assert not hasattr(safe_module, "lazy_attribute")
    assert getattr(safe_module, "non_lazy_attribute") == "ok"


@pytest.mark.asyncio
async def test_non_standard_comparisons():
    code = dedent("""\
        class NonStdEqualsResult:
            def __init__(self, left:object, right:object):
                self._left = left
                self._right = right
            def __str__(self) -> str:
                return f'{self._left} == {self._right}'

        class NonStdComparisonClass:
            def __init__(self, value: str ):
                self._value = value
            def __str__(self):
                return self._value
            def __eq__(self, other):
                return NonStdEqualsResult(self, other)
        a = NonStdComparisonClass("a")
        b = NonStdComparisonClass("b")
        result = a == b
        """)
    result = await async_evaluate(code, variables={})
    assert not isinstance(result, bool)
    assert str(result) == "a == b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "module,authorized_imports,expected",
    [
        ("os", ["*"], True),
        ("AnyModule", ["*"], True),
        ("os", ["os"], True),
        ("AnyModule", ["AnyModule"], True),
        ("Module.os", ["Module"], False),
        ("Module.os", ["Module", "os"], True),
        ("os.path", ["os"], True),
        ("os", ["os.path"], False),
    ],
)
async def test_check_module_authorized(module: str, authorized_imports: list[str], expected: bool):
    dangerous_patterns = (
        "_os",
        "os",
        "subprocess",
        "_subprocess",
        "pty",
        "system",
        "popen",
        "spawn",
        "shutil",
        "sys",
        "pathlib",
        "io",
        "socket",
        "compile",
        "eval",
        "exec",
        "multiprocessing",
    )
    assert check_module_authorized(module, authorized_imports, dangerous_patterns) == expected


@pytest.mark.asyncio
async def test_temp():
    code = dedent("""\
        x = 1
        x += 2
        """)
    result = await async_evaluate(code, {}, {})
    assert result == 3

@pytest.mark.asyncio
async def test__name__():
    code = dedent("""\
        def foo(): return 0
        foo.__name__
        """)
    result = await async_evaluate(code, {}, {})
    assert result == "foo"

@pytest.mark.asyncio
async def test_function_returning_function():
    code = dedent("""\
        def f(): 
            return lambda x: x + 1
        f()(1)
        """)
    result = await async_evaluate(code, {}, {})
    assert result == 2

@pytest.mark.asyncio
async def test_custom_subscriptable():
    code = dedent("""\
        class MyList:
            def __init__(self, items):
                self.items = items
            def __getitem__(self, index):
                return self.items[index]
        my_list = MyList(["a", "b", "c"])
        my_list[1]
        """)
    result = await async_evaluate(code, {}, {})
    assert result == "b"


@pytest.mark.asyncio
async def test_not_callable_error():
    code = dedent("'foo'()")
    with pytest.raises(InterpreterError) as e:
        await async_evaluate(code)
    assert "'str' object is not callable" in str(e.value)


@pytest.mark.asyncio
async def test_similar_names():
    code = "fo"
    with pytest.raises(InterpreterError) as e:  
        await async_evaluate(code, {}, {}, { "foo": "bar" })
    assert "The name `fo` is not defined." in str(e.value)


@pytest.mark.asyncio
async def test_client_error():
    def foo():
        raise ClientError("foo")
    code = "foo()"
    with pytest.raises(ClientError) as e:
        await async_evaluate(code, {}, {"foo": foo})
    assert "foo" in str(e.value)

@pytest.mark.asyncio
async def test_contemplation():
    code = dedent("""\
        foo = [1, 2, 3]
        [a+1 for a in foo]
        """)
    result = await async_evaluate(code, {}, {})
    assert result == [2, 3, 4]

@pytest.mark.asyncio
async def test_await():
    code = dedent("""\
        import asyncio
        async def foo(): 
            await asyncio.sleep(0.01)
            return 0
        await foo()
        """)
    result = await async_evaluate(code, {}, {})
    assert result == 0


@pytest.mark.asyncio
async def test_python_interpreter():
    def add_one(lst):
        return [x + 1 for x in lst]
    variables = {"a": [1, 2, 3]}
    stdout = io.StringIO()
    interpreter = AsyncPythonInterpreter(
        additional_functions={"add_one": add_one},
        initial_variables=variables,
        stdout=stdout,
    )
    
    value = await interpreter("b = a + [4]; print(b); b")
    assert value == [1, 2, 3, 4]
    assert stdout.getvalue() == "[1, 2, 3, 4]\n"

    value = await interpreter("b = add_one(b); print(b); b")
    assert value == [2, 3, 4, 5]
    assert stdout.getvalue() == "[1, 2, 3, 4]\n[2, 3, 4, 5]\n"

    assert variables is interpreter.variables
    assert variables["a"] == [1, 2, 3]
    assert variables["b"] == [2, 3, 4, 5]

