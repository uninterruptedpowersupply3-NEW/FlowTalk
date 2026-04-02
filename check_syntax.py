import ast
import sys

files = [
    r"C:\Users\chatr\Documents\Tech\VLLM\New folder\test_dataset_generalization.py",
    r"C:\Users\chatr\Documents\Tech\VLLM\New folder\gui_app.py"
]

has_error = False

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as file:
            content = file.read()
        ast.parse(content)
        print(f"[OK] {f}: Syntax OK")
    except Exception as e:
        print(f"[FAIL] {f}: Syntax Error: {e}")
        has_error = True

if has_error:
    sys.exit(1)
else:
    sys.exit(0)
