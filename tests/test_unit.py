# =============================================================================
# UNIT TESTS — Energy functions and quantum circuit validity
# =============================================================================
# Run from 2026-NVIDIA: PYTHONPATH=tutorial_notebook python tests/test_unit.py
# Or: cd 2026-NVIDIA && PYTHONPATH=tutorial_notebook python tests/test_unit.py

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tutorial_notebook"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import run_ex6
import auxiliary_files.labs_utils as utils
import cudaq

# -----------------------------------------------------------------------------
# Test data (known optimal energies and sequences that achieve them)
# -----------------------------------------------------------------------------
KNOWN_OPTIMAL = {3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 6, 9: 4, 10: 4}
KNOWN_SEQUENCES = {
    3: [1, 1, -1],
    4: [1, -1, -1, -1],
    5: [1, 1, 1, -1, 1],
    6: [1, 1, 1, -1, -1, 1],
    7: [1, 1, 1, -1, -1, 1, -1],
    8: [1, 1, 1, -1, 1, -1, -1, 1],
    9: [1, 1, 1, 1, 1, -1, -1, 1, -1],
    10: [1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
}

# =============================================================================
# UNIT TESTS
# =============================================================================
def run_unit_tests():
    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60)

    compute_energy = run_ex6.compute_energy
    get_interactions = run_ex6.get_interactions
    trotterized_circuit = run_ex6.trotterized_circuit

    # Test 1: Energy correctness
    errors = []
    for N in [3, 4, 5, 7]:
        if N not in KNOWN_SEQUENCES or N not in KNOWN_OPTIMAL:
            continue
        seq = KNOWN_SEQUENCES[N]
        got = compute_energy(seq)
        want = KNOWN_OPTIMAL[N]
        if got != want:
            errors.append(f"N={N}: expected {want}, got {got}")
    if errors:
        print("FAIL energy:", "; ".join(errors))
    else:
        print("PASS: Energy correctness (N=3,4,5,7)")

    # Test 2: Energy non-negative
    bad = []
    for _ in range(20):
        N = np.random.randint(3, 11)
        seq = np.random.choice([1, -1], size=N)
        if compute_energy(seq) < 0:
            bad.append(seq)
    if bad:
        print("FAIL: energy < 0 for some sequence")
    else:
        print("PASS: Energy non-negative")

    # Test 3: Quantum circuit returns valid bitstrings
    N_test = 5
    G2, G4 = get_interactions(N_test)
    T, n_steps = 1.0, 2
    dt = T / n_steps
    thetas = [utils.compute_theta((i + 1) * dt, dt, T, N_test, G2, G4) for i in range(n_steps)]
    result = cudaq.sample(
        trotterized_circuit, N_test, G2, G4, n_steps, dt, T, thetas, shots_count=100
    )
    valid = True
    for bs in result:
        if len(bs) != N_test or not all(b in ("0", "1") for b in bs):
            valid = False
            break
    if not valid:
        print("FAIL: Quantum circuit returned invalid bitstring")
    else:
        print("PASS: Quantum circuit returns valid bitstrings")

    print("=" * 60)


if __name__ == "__main__":
    run_unit_tests()
