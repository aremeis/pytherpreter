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
# pytherpreter
A Python interpreter with built-in safeguards for executing untrusted code, like LLM-generated scripts.

This repository extracts the Python interpreter tool from HuggingFace’s [smolagents](https://github.com/huggingface/smolagents) project.

## Installation
```shell
pip install pytherpreter
```

## Usage

### `evaluate_python_code`

`evaluate_python_code` is a function that evaluates Python code and returns the result.
```python
from pytherpreter import evaluate_python_code

result, final_answer = evaluate_python_code("""
from math import sqrt
sqrt(4)
""")
print(result)
print(final_answer)

# Output:
# 2.0
# False
```

By default, the `evaluate_python_code` function will return the result of the last expression in the code.
However, you can also return a value from the code by using the `final_answer` function:

```python
result, _ = evaluate_python_code("""
from math import sqrt
final_answer(sqrt(4))
""")
print(result)
print(result)
print(final_answer)

# Output:
# 2.0
# True
```

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
