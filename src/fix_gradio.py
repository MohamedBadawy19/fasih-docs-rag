import gradio_client.utils as u
import inspect
import pathlib
import re

src = pathlib.Path(inspect.getfile(u))
code = src.read_text(encoding='utf-8')
original = code

# Patch 1: guard get_type
old1 = "def get_type(schema: dict):"
new1 = ("def get_type(schema: dict):\n"
        "    if not isinstance(schema, dict):\n"
        "        return \"any\"  # patched")

if old1 in code and new1 not in code:
    code = code.replace(old1, new1)
    print("Patched: get_type")
elif new1 in code:
    print("Already patched: get_type")

# Patch 2: guard _json_schema_to_python_type
# Find the function definition line and insert guard right after it
pattern = r'(def _json_schema_to_python_type\([^)]+\) -> str:\n)'
match = re.search(pattern, code)
guard = "    if not isinstance(schema, dict):\n        return \"any\"  # patched\n"

if match:
    if guard not in code:
        insert_pos = match.end()
        code = code[:insert_pos] + guard + code[insert_pos:]
        print("Patched: _json_schema_to_python_type")
    else:
        print("Already patched: _json_schema_to_python_type")
else:
    print("ERROR: Could not find _json_schema_to_python_type definition")

# Write and clear cache
if code != original:
    src.write_text(code, encoding='utf-8')
    print(f"Written: {src}")

cache = src.parent / '__pycache__'
for f in cache.glob('utils*.pyc'):
    f.unlink()
    print(f"Deleted cache: {f.name}")

print("Done.")