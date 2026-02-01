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

import random

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

    # Test 4: LABS symmetries (negation, reflection)
    np.random.seed(42)
    for _ in range(100):
        N = np.random.randint(3, 12)
        s = np.random.choice([-1, 1], size=N)
        if compute_energy(s) != compute_energy(-s):
            print("FAIL: LABS symmetry violated — E(s) != E(-s)")
            break
        if compute_energy(s) != compute_energy(s[::-1]):
            print("FAIL: LABS symmetry violated — E(s) != E(reverse(s))")
            break
    else:
        print("PASS: LABS symmetries (negation, reflection) on 100 random sequences")

    # Test 5: Tabu search improves or maintains energy
    np.random.seed(123)
    for N in [5, 7]:
        init = np.random.choice([-1, 1], size=N)
        init_e = compute_energy(init)
        best, best_e = run_ex6.tabu_search(init, max_iter=30, tabu_tenure=3)
        if best_e > init_e:
            print(f"FAIL: Tabu search worsened energy for N={N}: {init_e} -> {best_e}")
            break
    else:
        print("PASS: Tabu search improves or maintains energy")

    # Test 6: MTS finds known optimal (or close) for small N
    random.seed(456)
    np.random.seed(456)
    found_optimal = True
    for N, opt_e in [(5, 2), (7, 3)]:
        best_seq, best_e, _, _ = run_ex6.memetic_tabu_search(
            N=N, pop_size=15, generations=30, p_mut=0.15, tabu_iterations=25, verbose=False
        )
        if best_e > opt_e:
            found_optimal = False
            print(f"FAIL: MTS best E={best_e} > known optimal {opt_e} for N={N}")
            break
    if found_optimal:
        print("PASS: MTS finds known optimal for N=5,7")

    print("=" * 60)


if __name__ == "__main__":
    run_unit_tests()
