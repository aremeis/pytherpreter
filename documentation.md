# Pytherpreter Reference Documentation

## Core Functions

### evaluate(code, builtin_functions=BASE_BUILTIN_FUNCTIONS, custom_functions=None, variables=None, authorized_imports=BASE_BUILTIN_MODULES, stdout=sys.stdout)

Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set of functions.

#### Parameters:

- `code` (str): The code to evaluate.

- `builtin_functions` (Dict[str, Callable], optional): 
  - The built-in functions that may be called during the evaluation.
  - These functions cannot be overwritten in the code: any assignment to their name will raise an error.
  - Defaults to BASE_BUILTIN_FUNCTIONS.

- `custom_functions` (Dict[str, Callable], optional):
  - The custom functions that may be called during the evaluation.
  - These functions can be overwritten in the code: any assignment to their name will overwrite them.
  - Defaults to None.

- `variables` (Dict[str, Any], optional):
  - A dictionary mapping variable names to values.
  - The `variables` should contain the initial inputs but will be updated by this function to contain all variables as they are evaluated.
  - Defaults to None.

- `authorized_imports` (List[str], optional):
  - The list of modules that can be imported by the code.
  - Defaults to BASE_BUILTIN_MODULES.

- `stdout` (TextIO, optional):
  - The stream to be used for print outputs.
  - If None, the print function will be a no-op.
  - Defaults to sys.stdout.

#### Returns:

The result of the last evaluated expression in the code.

### async_evaluate(code, builtin_functions=BASE_BUILTIN_FUNCTIONS, custom_functions=None, variables=None, authorized_imports=ASYNC_BASE_BUILTIN_MODULES, stdout=sys.stdout)

Asynchronously evaluate a python expression using the content of the variables stored in a state and only evaluating a given set of functions.

#### Parameters:

- `code` (str): The code to evaluate.

- `builtin_functions` (Dict[str, Callable], optional): 
  - The built-in functions that may be called during the evaluation.
  - These functions cannot be overwritten in the code: any assignment to their name will raise an error.
  - Defaults to BASE_BUILTIN_FUNCTIONS.

- `custom_functions` (Dict[str, Callable], optional):
  - The custom functions that may be called during the evaluation.
  - These functions can be overwritten in the code: any assignment to their name will overwrite them.
  - Defaults to None.

- `variables` (Dict[str, Any], optional):
  - A dictionary mapping variable names to values.
  - The `variables` should contain the initial inputs but will be updated by this function to contain all variables as they are evaluated.
  - Defaults to None.

- `authorized_imports` (List[str], optional):
  - The list of modules that can be imported by the code.
  - Defaults to ASYNC_BASE_BUILTIN_MODULES (includes 'asyncio' in addition to BASE_BUILTIN_MODULES).

- `stdout` (TextIO, optional):
  - The stream to be used for print outputs.
  - If None, the print function will be a no-op.
  - Defaults to sys.stdout.

#### Returns:

The result of the last evaluated expression in the code.

## Classes

### PythonInterpreter

A class that allows you to evaluate python code with safeguards. This class is a wrapper around the `evaluate` function. It will keep track of the state between calls.

#### Constructor Parameters:

- `additional_authorized_imports` (Iterable[str], optional):
  - Additional authorized imports beyond the base imports.
  - Defaults to empty list.

- `additional_functions` (Dict[str, Callable], optional):
  - Additional functions to make available to the interpreter.
  - Defaults to empty dict.

- `initial_variables` (Dict[str, Any], optional):
  - Initial variables to populate the interpreter's state.
  - Defaults to empty dict.

- `stdout` (TextIO, optional):
  - The stream to be used for print outputs.
  - If None, the print function will be a no-op.
  - Defaults to sys.stdout.

#### Methods:

##### __call__(code: str, additional_variables: Dict = {}) -> Any

Evaluate the code and return the result.

Parameters:
- `code` (str): The code to evaluate
- `additional_variables` (Dict): Additional variables to add to the state. Defaults to empty dict.

Returns:
The result of the last evaluated expression in the code.

### AsyncPythonInterpreter

A class that allows you to asynchronously evaluate python code with safeguards. This class is a wrapper around the `async_evaluate` function. It will keep track of the state between calls.

#### Constructor Parameters:

- `additional_authorized_imports` (Iterable[str], optional):
  - Additional authorized imports beyond the base imports.
  - Defaults to empty list.

- `additional_functions` (Dict[str, Callable], optional):
  - Additional functions to make available to the interpreter.
  - Defaults to empty dict.

- `initial_variables` (Dict[str, Any], optional):
  - Initial variables to populate the interpreter's state.
  - Defaults to empty dict.

- `stdout` (TextIO, optional):
  - The stream to be used for print outputs.
  - If None, the print function will be a no-op.
  - Defaults to sys.stdout.

#### Methods:

##### async __call__(code: str, additional_variables: Dict = {}) -> Any

Asynchronously evaluate the code and return the result.

Parameters:
- `code` (str): The code to evaluate
- `additional_variables` (Dict): Additional variables to add to the state. Defaults to empty dict.

Returns:
The result of the last evaluated expression in the code.
