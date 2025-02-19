<!---
# Copyright 2025 Are Meisfjord. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# 🐍 pytherpreter 🐍

A Python interpreter with built-in safeguards for executing untrusted code, like LLM-generated scripts.

This repository contains the Python interpreter tool extracted from HuggingFace’s [_smolagents_](https://github.com/huggingface/smolagents) project.
Big hug to the HuggingFace team for their initial implementation! 🤗

Some improvements over the smolagents tool:
- Supports async code execution using the `async_evaluate` function and `AsyncPythonInterpreter` class.
- Improved function call resolution.
- Supports custom subscriptable objects.
- No external dependencies.
- More flexible `print` handling.

## Installation
```shell
pip install pytherpreter
```

Latest development version:

```shell
pip install git+ssh://git@github.com/aremeis/pytherpreter.git
```

or

```shell
pip install git+https://github.com/aremeis/pytherpreter.git
```

## Usage

### Using `evaluate`

This function evaluates Python code and returns the result.
```python
from pytherpreter import evaluate

result = evaluate("""
from math import sqrt
sqrt(4)
""")
print(result)

# Output:
# 2.0
```

The `evaluate` function returns the result of the last expression in the code.

### Using `PythonInterpreter`

This class is a wrapper around the `evaluate` function that keeps the state of the interpreter between calls.
Variables and functions defined by the code will be be available in subsequent calls.

```python
from pytherpreter import PythonInterpreter

interpreter = PythonInterpreter()
result = interpreter("x = 3")
print(result)

# Output:
# 3

result = interpreter("x += 1")
print(result)

# Output:
# 4
```

### Printing

You may provide a `stdout` argument to capture the output of print statements in the code.

```python
from pytherpreter import evaluate
import io
stdout = io.StringIO()
result = evaluate("print('Hello, World!')", stdout=stdout)
print(stdout.getvalue())

# Output:
# Hello, World!
```

### Variables

You may provide a `variables` argument to preset variables and capture changes to them.

```python
from pytherpreter import evaluate
variables = {"x": 3}
result = evaluate("x += 1", variables=variables)
print(variables["x"])

# Output:
# 4
```

## Safeguards

### Built-in functions

You may provide a `builtin_functions` argument containing a dictionary of built-in functions the code is allowed to call.
If you don't provide this argument, the code will only be able to call the built-in functions in `BASE_BUILTIN_FUNCTIONS`.

The code will not be able to modify the provided built-in functions.

### Modules

By default, the code will only be able to import the modules in `BASE_BUILTIN_MODULES`.
You may provide an `authorized_imports` argument to allow the code to import additional modules.


## Documentation

For more details, the reference documentation is available [here](documentation.md).

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
