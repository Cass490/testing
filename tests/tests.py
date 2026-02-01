# =============================================================================
# tests.py — Rigorous test suite (Phase 2 requirement)
# Covers: energy functions and quantum kernels
# =============================================================================
# Run from 2026-NVIDIA: PYTHONPATH=tutorial_notebook python tests/tests.py
# Or: cd 2026-NVIDIA && PYTHONPATH=tutorial_notebook python -m tests.tests

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tutorial_notebook"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Running unit tests...")
    print()
    import test_unit
    test_unit.run_unit_tests()
    print()
    print("Running kernel verification...")
    print()
    import test_kernels
    test_kernels.run_kernel_tests()
    print()
    print("All test scripts completed.")
