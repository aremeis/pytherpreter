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

This repository extracts the Python interpreter tool from HuggingFace’s [smolagents](https://github.com/huggingface/smolagents) project.

One improvement over smolagents is that pytherpreter also supports async code execution using the `async_evaluate` function and `AsyncPythonInterpreter` class.

## Installation
```shell
pip install pytherpreter
```

## Usage

### Using `evaluate`

This function evaluates Python code and returns the result.
```python
from pytherpreter import evaluate

result, final_answer = evaluate("""
from math import sqrt
sqrt(4)
""")
print(result)
print(final_answer)

# Output:
# 2.0
# False
```

By default, the `evaluate` function will return the result of the last expression in the code.
However, you can also return a value from the code by using the `final_answer` function:

```python
result, final_answer = evaluate_python_code("""
from math import sqrt
final_answer(sqrt(4))
""")
print(result)
print(final_answer)

# Output:
# 2.0
# True
```

### Using `PythonInterpreter`

This class is a wrapper around the `evaluate` function that keeps the state of the interpreter between calls.
Variables and functions defined by the code will be be available in subsequent calls.

```python
from pytherpreter import PythonInterpreter

interpreter = PythonInterpreter()
result, logs, is_final_answer = interpreter("x = 3")
print(result)
print(logs)
print(is_final_answer)

# Output:
# 3
# 
# False

result, logs, is_final_answer = interpreter("""
x += 1
print('x =', x)
final_answer(x);
""")
print(result)
print(logs)
print(is_final_answer)

# Output:
# 4
# x = 4
# True
```

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
