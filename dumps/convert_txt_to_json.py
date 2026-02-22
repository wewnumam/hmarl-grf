
import json
import numpy as np
import os
import sys

# Define actions as strings so eval works
actions = [
    'idle', 'left', 'top_left', 'top', 'top_right', 'right', 'bottom_right',
    'bottom', 'bottom_left', 'long_pass', 'high_pass', 'short_pass', 'shot',
    'sprint', 'release_direction', 'release_sprint', 'sliding', 'dribble',
    'release_dribble', 'release_direction', 'release_sprint',
]

class ActionMock:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f'"{self.name}"'

def array_mock(x, **kwargs):
    if isinstance(x, list):
        return x
    if hasattr(x, 'tolist'):
        return x.tolist()
    return x

eval_globals = {
    'array': array_mock,
    'uint8': 'uint8',
    'int32': 'int32',
    'float32': 'float32',
    'np': np,
}

for action in actions:
    eval_globals[action] = action

def convert_to_json(input_path, output_path):
    print(f"Reading {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist")
        return

    with open(input_path, 'r') as f:
        # Read in chunks if possible, but eval() needs the whole thing
        content = f.read()
    
    print("Evaluating content... (this may take a while for large files)")
    # We use a dict for globals to handle unquoted constants
    try:
        # We need to allow some built-ins for basic operations if any
        safe_builtins = __builtins__.copy() if isinstance(__builtins__, dict) else __builtins__.__dict__.copy()
        data = eval(content, {"__builtins__": safe_builtins}, eval_globals)
    except NameError as e:
        print(f"Detected missing name: {e}")
        missing_name = str(e).split("'")[1]
        eval_globals[missing_name] = missing_name
        # Try again once
        data = eval(content, {"__builtins__": safe_builtins}, eval_globals)
    except Exception as e:
        print(f"Error during eval: {e}")
        return

    print(f"Writing to {output_path}...")
    # Custom encoder to handle any remaining numpy types or mocks
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32, np.int8)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return super().default(obj)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, cls=MyEncoder)
    print("Successfully converted!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_txt_to_json.py <input.txt> <output.json>")
        sys.exit(1)
    convert_to_json(sys.argv[1], sys.argv[2])
