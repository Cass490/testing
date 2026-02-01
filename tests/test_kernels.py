# =============================================================================
# KERNEL VERIFICATION — get_interactions, thetas, trotterized_circuit, pipeline
# =============================================================================
# Run from 2026-NVIDIA: PYTHONPATH=tutorial_notebook python tests/test_kernels.py
# Or: cd 2026-NVIDIA && PYTHONPATH=tutorial_notebook python tests/test_kernels.py

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tutorial_notebook"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cudaq
import run_ex6
import auxiliary_files.labs_utils as utils


def bitstring_to_array(bs):
    """Convert CUDA-Q bitstring to LABS sequence (±1). '0' -> +1, '1' -> -1."""
    return np.array([1 if b == "0" else -1 for b in bs])


def test_get_interactions():
    errors = []
    for N in [3, 5, 7, 10]:
        G2, G4 = run_ex6.get_interactions(N)
        if not isinstance(G2, list):
            errors.append(f"N={N}: G2 is not a list")
            continue
        for pair in G2:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                errors.append(f"N={N}: G2 element {pair} is not a 2-tuple")
                break
            if not all(0 <= i < N for i in pair):
                errors.append(f"N={N}: G2 indices {pair} out of range [0,{N-1}]")
                break
        if not isinstance(G4, list):
            errors.append(f"N={N}: G4 is not a list")
            continue
        for quad in G4:
            if not (isinstance(quad, (list, tuple)) and len(quad) == 4):
                errors.append(f"N={N}: G4 element {quad} is not a 4-tuple")
                break
            if not all(0 <= i < N for i in quad):
                errors.append(f"N={N}: G4 indices {quad} out of range [0,{N-1}]")
                break
        G2_tuples = [tuple(sorted(p)) for p in G2]
        if len(G2_tuples) != len(set(G2_tuples)):
            errors.append(f"N={N}: G2 has duplicate pairs")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return False
    print("PASS: get_interactions(N) — structure and indices valid for N=3,5,7,10")
    return True


def test_thetas():
    N = 5
    G2, G4 = run_ex6.get_interactions(N)
    T = 1.0
    n_steps = 5
    dt = T / n_steps
    thetas = [utils.compute_theta((i + 1) * dt, dt, T, N, G2, G4) for i in range(n_steps)]
    if len(thetas) != n_steps:
        print(f"FAIL: thetas length {len(thetas)} != n_steps {n_steps}")
        return False
    for i, t in enumerate(thetas):
        if not isinstance(t, (int, float)) or np.isnan(t) or np.isinf(t):
            print(f"FAIL: theta[{i}] = {t} not a finite number")
            return False
    print("PASS: thetas — length correct, all finite")
    return True


def test_trotterized_circuit():
    N_test = 5
    G2, G4 = run_ex6.get_interactions(N_test)
    T, n_steps = 1.0, 3
    dt = T / n_steps
    thetas = [utils.compute_theta((i + 1) * dt, dt, T, N_test, G2, G4) for i in range(n_steps)]
    result = cudaq.sample(
        run_ex6.trotterized_circuit,
        N_test, G2, G4, n_steps, dt, T, thetas,
        shots_count=100,
    )
    total_shots = 0
    for bs in result:
        if len(bs) != N_test:
            print(f"FAIL: bitstring length {len(bs)} != N {N_test}: {bs}")
            return False
        if not all(b in ("0", "1") for b in bs):
            print(f"FAIL: invalid chars in bitstring: {bs}")
            return False
        total_shots += result.count(bs)
    if total_shots != 100:
        print(f"FAIL: total counts {total_shots} != shots_count 100")
        return False
    print("PASS: trotterized_circuit — runs and returns valid N-bit bitstrings, counts sum = shots_count")
    return True


def test_kernel_to_energy():
    N_test = 5
    G2, G4 = run_ex6.get_interactions(N_test)
    T, n_steps = 1.0, 3
    dt = T / n_steps
    thetas = [utils.compute_theta((i + 1) * dt, dt, T, N_test, G2, G4) for i in range(n_steps)]
    result = cudaq.sample(
        run_ex6.trotterized_circuit,
        N_test, G2, G4, n_steps, dt, T, thetas,
        shots_count=50,
    )
    for bs in result:
        arr = bitstring_to_array(bs)
        if len(arr) != N_test:
            print("FAIL: bitstring_to_array length != N")
            return False
        if not all(x in (-1, 1) for x in arr):
            print(f"FAIL: bitstring_to_array values not in {{-1,1}}: {arr}")
            return False
        E = run_ex6.compute_energy(arr)
        if E < 0:
            print(f"FAIL: compute_energy(bitstring_to_array(bs)) = {E} < 0 for bs={bs}")
            return False
    print("PASS: kernel -> bitstring_to_array -> compute_energy >= 0 for all samples")
    return True


# =============================================================================
# RUN ALL KERNEL VERIFICATION TESTS
# =============================================================================
def run_kernel_tests():
    print("=" * 60)
    print("KERNEL VERIFICATION")
    print("=" * 60)

    results = []
    results.append(("get_interactions", test_get_interactions()))
    results.append(("thetas", test_thetas()))
    results.append(("trotterized_circuit", test_trotterized_circuit()))
    results.append(("kernel_to_energy", test_kernel_to_energy()))

    print()
    passed = sum(1 for _, ok in results if ok)
    print(f"Kernel verification: {passed}/{len(results)} passed")
    print("=" * 60)


if __name__ == "__main__":
    run_kernel_tests()
