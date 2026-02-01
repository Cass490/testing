# How to Run Our Code

## What is PYTHONPATH?

**PYTHONPATH** is an environment variable that tells Python **where to look for modules** when you run `import something`.

- Our scripts do `import auxiliary_files.labs_utils` — that module lives in `tutorial_notebook/auxiliary_files/`.
- If you run the tests from the repo root without setting PYTHONPATH, Python won’t find `auxiliary_files` and you’ll get `ModuleNotFoundError`.
- Setting **PYTHONPATH=tutorial_notebook** adds the `tutorial_notebook` folder to the search path, so `auxiliary_files.labs_utils` resolves to `tutorial_notebook/auxiliary_files/labs_utils.py`.

**In one line:** PYTHONPATH is the list of folders Python searches when resolving `import` statements.

---

## Running from the repo root (e.g. `2026-NVIDIA/`)

Make sure you’re in the folder that contains both `tutorial_notebook/` and `tests/` (the challenge repo root).

### Run the full Ex 6 workflow (QE-MTS vs Classical MTS)
```bash
PYTHONPATH=tutorial_notebook python tests/run_ex6.py
```

### Run the test suite (unit tests + kernel verification)
```bash
PYTHONPATH=tutorial_notebook python tests/tests.py
```

### Run unit tests only
```bash
PYTHONPATH=tutorial_notebook python tests/test_unit.py
```

### Run kernel verification only
```bash
PYTHONPATH=tutorial_notebook python tests/test_kernels.py
```

---

## Alternative: run from inside `tutorial_notebook/`

If you `cd tutorial_notebook` first, then `auxiliary_files` is in the current directory and you don’t need to set PYTHONPATH for imports from the notebook. For the **tests**, you still need the repo root and path setup so that `tests/` and `run_ex6` are found; the commands above from the repo root are the intended way to run everything.
