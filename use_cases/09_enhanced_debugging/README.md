# Use Case 09: Enhanced AI Debugging

The `@enhanced_debug` decorator catches exceptions, prints the standard Python traceback, and then invokes the Apple Neural Engine to perform an automated root-cause analysis. By default, summaries go to `stderr` and prompts go to `stdout`.

## How It Works

```python
from silicon_refinery import enhanced_debug

@enhanced_debug(prompt_to="crash_report_for_llm.txt")
def process_data(data_payload):
    """A buggy function that will inevitably crash."""
    parsed_value = data_payload["value"] + 10
    return parsed_value

process_data({"value": "100"})  # TypeError: can only concatenate str (not "int") to str
```

When the function crashes, SiliconRefinery will:

1. Print the standard Python traceback to stderr
2. Send the traceback to the on-device Foundation Model for analysis
3. Print a structured diagnosis with certainty level, possible causes, and suggested fix
4. Route the detailed prompt via `prompt_to` (defaults to `stdout`; supports `stderr`, `log`, and file path `.txt` output)

## Sample Output

The file [`sample_crash_report.txt`](sample_crash_report.txt) shows what the generated prompt file looks like — ready to paste into Claude, Codex, or any coding assistant for a deeper fix.

## Run It

```bash
python use_cases/09_enhanced_debugging/example.py
```
