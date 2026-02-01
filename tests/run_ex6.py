import cudaq
import numpy as np
import auxiliary_files.labs_utils as utils
# ============================================================================
# EXERCISE 2: Memetic Tabu Search (MTS) Implementation
# ============================================================================

import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# LABS Energy Computation
# -----------------------------------------------------------------------------
def compute_energy(sequence):
    """
    Compute the LABS energy E(s) = sum_{k=1}^{N-1} C_k^2
    where C_k = sum_{i=1}^{N-k} s_i * s_{i+k}
    
    Args:
        sequence: numpy array or list of +1/-1 values
    Returns:
        Energy value (integer)
    """
    s = np.array(sequence)
    N = len(s)
    energy = 0
    for k in range(1, N):
        C_k = np.dot(s[:N-k], s[k:])
        energy += C_k ** 2
    return energy

def compute_autocorrelation(sequence):
    """Compute all autocorrelation values C_k for visualization."""
    s = np.array(sequence)
    N = len(s)
    C = []
    for k in range(N):
        C_k = np.dot(s[:N-k], s[k:]) if k < N else 0
        C.append(C_k)
    return C

# -----------------------------------------------------------------------------
# Combine Operation (Paper's Algorithm: Single-point crossover)
# -----------------------------------------------------------------------------
def combine(p1, p2):
    """
    Args:
        p1, p2: Parent sequences (numpy arrays)
    Returns:
        child: Combined sequence
    """
    N = len(p1)
    k = random.randint(1, N - 1)  # Cut point (1-based becomes 0-based slice)
    child = np.concatenate([p1[:k], p2[k:]])
    return child

# -----------------------------------------------------------------------------
# Mutate Operation (Paper's Algorithm: Flip each bit with probability p_mut)
# -----------------------------------------------------------------------------
def mutate(s, p_mut):
    """
    For i = 1 to N:
        if rand(0,1) < p_mut: flip bit s[i]
    
    Args:
        s: Sequence to mutate
        p_mut: Mutation probability per bit
    Returns:
        mutated: Mutated sequence
    """
    child = s.copy()
    for i in range(len(child)):
        if random.random() < p_mut:
            child[i] *= -1  # Flip the bit
    return child

# -----------------------------------------------------------------------------
# Tabu Search (Local Search)
# -----------------------------------------------------------------------------
def tabu_search(initial_seq, max_iter=100, tabu_tenure=5):
    """
    Greedy local search with tabu list.
    
    Args:
        initial_seq: Starting sequence
        max_iter: Maximum iterations
        tabu_tenure: How long a move stays tabu
    Returns:
        best: Best sequence found
        best_energy: Energy of best sequence
    """
    current = np.array(initial_seq).copy()
    best = current.copy()
    current_energy = compute_energy(current)
    best_energy = current_energy
    
    N = len(current)
    tabu_list = []
    
    for _ in range(max_iter):
        best_neighbor = None
        best_neighbor_energy = float('inf')
        best_move = -1
        
        # Evaluate all 1-flip neighbors
        for i in range(N):
            neighbor = current.copy()
            neighbor[i] *= -1
            e = compute_energy(neighbor)
            
            is_tabu = i in tabu_list
            beats_best = e < best_energy  # Aspiration criterion
            
            # Accept if not tabu OR if it beats global best (aspiration)
            if (not is_tabu) or beats_best:
                if e < best_neighbor_energy:
                    best_neighbor_energy = e
                    best_neighbor = neighbor
                    best_move = i
        
        if best_neighbor is None:
            break
            
        current = best_neighbor
        current_energy = best_neighbor_energy
        
        # Update global best
        if current_energy < best_energy:
            best_energy = current_energy
            best = current.copy()
        
        # Update tabu list (FIFO queue)
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
    return best, best_energy

# -----------------------------------------------------------------------------
# Population Initialization
# -----------------------------------------------------------------------------
def initialize_population_random(N, pop_size):
    return [np.random.choice([-1, 1], size=N) for _ in range(pop_size)]

def initialize_population_from_samples(samples, pop_size, N):
    population = []
    for sample in samples[:pop_size]:
        if isinstance(sample, str):
            # Convert bitstring to +1/-1 (0 -> +1, 1 -> -1)
            seq = np.array([1 if b == '0' else -1 for b in sample])
        else:
            seq = np.array(sample)
        population.append(seq)
    
    # If not enough samples, add random sequences
    while len(population) < pop_size:
        seq = np.random.choice([-1, 1], size=N)
        population.append(seq)
    
    return population

# -----------------------------------------------------------------------------
# Memetic Tabu Search (MTS) - Paper's Algorithm
# -----------------------------------------------------------------------------
def memetic_tabu_search(N, pop_size=20, generations=50, p_mut=0.1, E_target=0,
                        tabu_iterations=50, tabu_tenure=None,
                        initial_population=None, verbose=True):
    """
    Memetic Tabu Search as described in the paper.
    
    Algorithm:
    1. Initialize population with k random bitstrings
    2. s_star = best solution in population
    3. While E(s_star) > E_target:
       a. MakeChild: Sample directly OR Combine two parents
       b. Mutate child with probability p_mut per bit
       c. Run Tabu Search on child
       d. Update s_star if child is better
       e. Replace RANDOM population member if child is better
    
    Args:
        N: Sequence length
        pop_size: Population size (k in paper)
        generations: Maximum generations
        p_mut: Mutation probability per bit
        E_target: Target energy (stop if reached)
        tabu_iterations: Max iterations for local tabu search
        tabu_tenure: Tabu tenure (default: N//4)
        initial_population: Optional pre-initialized population
        verbose: Print progress
    
    Returns:
        s_star: Best sequence found
        E_star: Best energy found
        population: Final population
        history: List of best energies per generation
    """
    if tabu_tenure is None:
        tabu_tenure = max(N // 4, 3)
    
    # Initialize population with k random bitstrings
    if initial_population is None:
        population = initialize_population_random(N, pop_size)
    else:
        population = [np.array(p) for p in initial_population]
    
    # Compute initial energies
    energies = [compute_energy(ind) for ind in population]
    
    # s_star = best solution in population
    best_idx = np.argmin(energies)
    s_star = population[best_idx].copy()
    E_star = energies[best_idx]
    
    history = []
    
    # Main loop: while E(s_star) > E_target
    for gen in range(generations):
        if E_star <= E_target:
            if verbose:
                print(f"Target energy {E_target} reached at generation {gen}!")
            break
        
        # Step 1: MakeChild - two options
        if random.random() < 0.5:
            # Option A: Sample a bitstring directly from population
            child = random.choice(population).copy()
        else:
            # Option B: Combine two parent bitstrings from population
            idx1, idx2 = random.sample(range(pop_size), 2)
            child = combine(population[idx1], population[idx2])
        
        # Step 2: Mutate child
        child = mutate(child, p_mut)
        
        # Step 3: Tabu Search on child
        child_result, child_energy = tabu_search(
            child, max_iter=tabu_iterations, tabu_tenure=tabu_tenure
        )
        
        # Step 4: Update best solution (s_star)
        if child_energy < E_star:
            E_star = child_energy
            s_star = child_result.copy()
            if verbose:
                print(f"Generation {gen}: New best energy = {E_star}")
        
        # Step 5: Update population - replace RANDOM member if child is better
        random_idx = random.randint(0, pop_size - 1)
        if child_energy < energies[random_idx]:
            population[random_idx] = child_result
            energies[random_idx] = child_energy
        
        history.append(E_star)
    
    if verbose:
        print(f"\nMTS Complete: Best energy = {E_star}")
    
    return s_star, E_star, population, history

# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------
def visualize_mts_results(history, population, N, title_prefix=""):
    """Visualize MTS optimization results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Energy convergence
    ax1 = axes[0]
    ax1.plot(range(len(history)), history, 'b-', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Energy')
    ax1.set_title(f'{title_prefix}Energy Convergence (N={N})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final population energy distribution
    ax2 = axes[1]
    final_energies = [compute_energy(seq) for seq in population]
    ax2.hist(final_energies, bins=max(5, len(set(final_energies))), 
             edgecolor='black', alpha=0.7)
    ax2.axvline(x=min(final_energies), color='r', linestyle='--', 
                label=f'Best: {min(final_energies)}')
    ax2.set_xlabel('Energy')
    ax2.set_ylabel('Count')
    ax2.set_title(f'{title_prefix}Final Population Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best sequence autocorrelation
    ax3 = axes[2]
    best_idx = np.argmin(final_energies)
    best_seq = population[best_idx]
    autocorr = compute_autocorrelation(best_seq)
    ax3.bar(range(len(autocorr)), autocorr, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Lag k')
    ax3.set_ylabel('Autocorrelation C_k')
    ax3.set_title(f'{title_prefix}Best Sequence Autocorrelation')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()


# ============================================================================
# EXERCISE 3: CUDA-Q Kernels for 2-qubit and 4-qubit Operators
# ============================================================================
# Based on Appendix B of the paper: "Efficient construction of DCQO circuits"
#
# From the paper (Eq. B3):
# U(0,T) = ∏_{k=1}^{n_trot} [2-body terms] × [4-body terms]
#
# 2-body terms: R_YZ(4θh_i^x) R_ZY(4θh_{i+k}^x)
# 4-body terms: R_YZZZ(8θh_i^x) R_ZYZZ(8θh_{i+t}^x) R_ZZYZ(8θh_{i+k}^x) R_ZZZY(8θh_{i+k+t}^x)
#
# Figure 3: 2-qubit block requires 2 R_ZZ gates + 4 single-qubit gates
# Figure 4: 4-qubit block requires 10 R_ZZ gates + 28 single-qubit gates

import matplotlib.pyplot as plt

# =============================================================================
# R_ZZ Gate - Fundamental building block
# =============================================================================
@cudaq.kernel
def rzz_gate(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    """
    Implements R_ZZ(θ) = exp(-i·θ/2·Z⊗Z)
    
    Standard decomposition:
    ──●──────────●──
      │          │
    ──X──Rz(θ)──X──
    """
    x.ctrl(q0, q1)      # CNOT
    rz(theta, q1)       # RZ on target
    x.ctrl(q0, q1)      # CNOT

# =============================================================================
# 2-Qubit Operators: R_YZ and R_ZY (from Paper Figure 3)
# =============================================================================
# The paper shows that R_YZ(θ)R_ZY(θ) combined uses 2 R_ZZ gates
# Individual gates use basis transformation: Y = Rx(π/2)† Z Rx(π/2)

@cudaq.kernel
def r_yz_gate(reg: cudaq.qview, i: int, j: int, theta: float):
    """
    Implements R_YZ(θ) = exp(-i·θ/2·Y_i⊗Z_j)
    
    From paper Figure 3 - First half of 2-qubit block:
    q_i: ──Rx(π/2)──●──────────●──Rx(-π/2)──
                    │          │
    q_j: ───────────X──Rz(θ)───X────────────
    
    Key insight: Transform Y→Z basis with Rx(±π/2), then use ZZ interaction
    """
    rx(np.pi / 2.0, reg[i])     # Y → Z basis transformation
    x.ctrl(reg[i], reg[j])       # CNOT (creates ZZ parity)
    rz(theta, reg[i])            # Apply rotation
    x.ctrl(reg[i], reg[j])       # CNOT (undo parity)
    rx(-np.pi / 2.0, reg[i])    # Z → Y basis transformation

@cudaq.kernel
def r_zy_gate(reg: cudaq.qview, i: int, j: int, theta: float):
    """
    Implements R_ZY(θ) = exp(-i·θ/2·Z_i⊗Y_j)
    
    From paper Figure 3 - Second half of 2-qubit block:
    q_i: ───────────●──────────●────────────
                    │          │
    q_j: ──Rx(π/2)──X──Rz(θ)───X──Rx(-π/2)──
    """
    rx(np.pi / 2.0, reg[j])     # Y → Z basis transformation
    x.ctrl(reg[i], reg[j])       # CNOT
    rz(theta, reg[j])            # Apply rotation  
    x.ctrl(reg[i], reg[j])       # CNOT
    rx(-np.pi / 2.0, reg[j])    # Z → Y basis transformation

# =============================================================================
# 4-Qubit Operators: R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY (from Paper Figure 4)
# =============================================================================
# The paper shows the 4-qubit block uses 10 R_ZZ gates total
# Each individual gate uses CNOT cascade to accumulate parity

@cudaq.kernel
def r_yzzz_gate(reg: cudaq.qview, i: int, it: int, ik: int, ikt: int, theta: float):
    """
    Implements R_YZZZ(θ) = exp(-i·θ/2·Y_i⊗Z_{i+t}⊗Z_{i+k}⊗Z_{i+k+t})
    
    From paper Figure 4:
    - Rx(π/2) on qubit with Y (qubit i)
    - CNOT cascade to accumulate parity onto final qubit
    - Rz(θ) on final qubit
    - Reverse CNOT cascade
    - Rx(-π/2) on qubit i
    
    q_i:   ──Rx(π/2)──●─────────────────────────●──Rx(-π/2)──
                      │                         │
    q_it:  ───────────X────●───────────────●────X────────────
                           │               │
    q_ik:  ────────────────X────●─────●────X─────────────────
                                │     │
    q_ikt: ─────────────────────X─Rz──X──────────────────────
    """
    # Transform Y_i → Z_i
    rx(np.pi / 2.0, reg[i])
    
    # CNOT cascade: accumulate parity on qubit ikt
    x.ctrl(reg[i], reg[it])
    x.ctrl(reg[it], reg[ik])
    x.ctrl(reg[ik], reg[ikt])
    
    # Apply rotation on accumulated parity
    rz(theta, reg[ikt])
    
    # Reverse CNOT cascade
    x.ctrl(reg[ik], reg[ikt])
    x.ctrl(reg[it], reg[ik])
    x.ctrl(reg[i], reg[it])
    
    # Transform back Z_i → Y_i
    rx(-np.pi / 2.0, reg[i])

@cudaq.kernel
def r_zyzz_gate(reg: cudaq.qview, i: int, it: int, ik: int, ikt: int, theta: float):
    """
    Implements R_ZYZZ(θ) = exp(-i·θ/2·Z_i⊗Y_{i+t}⊗Z_{i+k}⊗Z_{i+k+t})
    
    Same structure as R_YZZZ but Rx gates on qubit it (the Y position)
    """
    rx(np.pi / 2.0, reg[it])
    
    x.ctrl(reg[i], reg[it])
    x.ctrl(reg[it], reg[ik])
    x.ctrl(reg[ik], reg[ikt])
    
    rz(theta, reg[ikt])
    
    x.ctrl(reg[ik], reg[ikt])
    x.ctrl(reg[it], reg[ik])
    x.ctrl(reg[i], reg[it])
    
    rx(-np.pi / 2.0, reg[it])

@cudaq.kernel
def r_zzyz_gate(reg: cudaq.qview, i: int, it: int, ik: int, ikt: int, theta: float):
    """
    Implements R_ZZYZ(θ) = exp(-i·θ/2·Z_i⊗Z_{i+t}⊗Y_{i+k}⊗Z_{i+k+t})
    
    Rx gates on qubit ik (the Y position)
    """
    rx(np.pi / 2.0, reg[ik])
    
    x.ctrl(reg[i], reg[it])
    x.ctrl(reg[it], reg[ik])
    x.ctrl(reg[ik], reg[ikt])
    
    rz(theta, reg[ikt])
    
    x.ctrl(reg[ik], reg[ikt])
    x.ctrl(reg[it], reg[ik])
    x.ctrl(reg[i], reg[it])
    
    rx(-np.pi / 2.0, reg[ik])

@cudaq.kernel
def r_zzzy_gate(reg: cudaq.qview, i: int, it: int, ik: int, ikt: int, theta: float):
    """
    Implements R_ZZZY(θ) = exp(-i·θ/2·Z_i⊗Z_{i+t}⊗Z_{i+k}⊗Y_{i+k+t})
    
    Rx gates on qubit ikt (the Y position)
    """
    rx(np.pi / 2.0, reg[ikt])
    
    x.ctrl(reg[i], reg[it])
    x.ctrl(reg[it], reg[ik])
    x.ctrl(reg[ik], reg[ikt])
    
    rz(theta, reg[ikt])
    
    x.ctrl(reg[ik], reg[ikt])
    x.ctrl(reg[it], reg[ik])
    x.ctrl(reg[i], reg[it])
    
    rx(-np.pi / 2.0, reg[ikt])

# ============================================================================
# EXERCISE 4: Generate Interaction Sets G2 and G4
# ============================================================================

def get_interactions(N):
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists of ints.
    
    From Equation 15 (B3 in appendix):
    
    2-body terms (G2): For i=1 to N-2, k=1 to floor((N-i)/2)
        Indices: [i, i+k]
        
    4-body terms (G4): For i=1 to N-3, t=1 to floor((N-i-1)/2), k=t+1 to N-i-t
        Indices: [i, i+t, i+k, i+k+t]
    
    Note: Paper uses 1-based indexing, we convert to 0-based.
    
    Args:
        N (int): Sequence length.
        
    Returns:
        G2: List of lists containing two body term indices [[i, j], ...]
        G4: List of lists containing four body term indices [[i, t, k, kt], ...]
    """
    G2 = []
    G4 = []
    
    # ---------------------------------------------------------------------
    # Two-body terms G2
    # From equation: prod_{i=1}^{N-2} prod_{k=1}^{floor((N-i)/2)}
    # Paper uses 1-based, so i goes from 1 to N-2 (inclusive)
    # In 0-based: i goes from 0 to N-3 (inclusive)
    # ---------------------------------------------------------------------
    for i in range(N - 2):  # i = 0, 1, ..., N-3 (corresponds to paper's i=1 to N-2)
        # Paper: k from 1 to floor((N-i)/2) where i is 1-based
        # For 0-based i, the 1-based i_paper = i + 1
        # So k goes from 1 to floor((N - (i+1))/2) = floor((N-i-1)/2)
        max_k = (N - i - 1) // 2
        for k in range(1, max_k + 1):
            # Indices: i (0-based) and i+k (0-based)
            G2.append([i, i + k])
    
    # ---------------------------------------------------------------------
    # Four-body terms G4  
    # From equation: prod_{i=1}^{N-3} prod_{t=1}^{floor((N-i-1)/2)} prod_{k=t+1}^{N-i-t}
    # Paper uses 1-based indexing
    # In 0-based: i goes from 0 to N-4 (inclusive)
    # ---------------------------------------------------------------------
    for i in range(N - 3):  # i = 0, 1, ..., N-4 (corresponds to paper's i=1 to N-3)
        # For 0-based i, paper's i_paper = i + 1
        # t goes from 1 to floor((N - i_paper - 1)/2) = floor((N - i - 2)/2)
        max_t = (N - i - 2) // 2
        for t in range(1, max_t + 1):
            # k goes from t+1 to N - i_paper - t = N - (i+1) - t = N - i - 1 - t
            max_k = N - i - 1 - t
            for k in range(t + 1, max_k + 1):
                # Four indices (0-based):
                # i, i+t, i+k, i+k+t
                idx_i = i
                idx_it = i + t
                idx_ik = i + k
                idx_ikt = i + k + t
                G4.append([idx_i, idx_it, idx_ik, idx_ikt])
                
    return G2, G4


# ============================================================================
# EXERCISE 5: Full Trotterized Counteradiabatic Circuit
# ============================================================================

@cudaq.kernel
def trotterized_circuit(N: int, G2: list[list[int]], G4: list[list[int]], 
                        steps: int, dt: float, T: float, thetas: list[float]):
    """
    Full Trotterized counteradiabatic circuit for LABS optimization.
    
    Implements Equation 15 (B3):
    U(0,T) = prod_{n=1}^{n_trot} [ 2-body terms ] x [ 4-body terms ]
    
    Args:
        N: Number of qubits (sequence length)
        G2: List of 2-body interaction indices [[i, j], ...]
        G4: List of 4-body interaction indices [[i, t, k, kt], ...]
        steps: Number of Trotter steps
        dt: Time step size
        T: Total evolution time
        thetas: Pre-computed theta values for each Trotter step
    """
    reg = cudaq.qvector(N)
    
    # Initialize in |+⟩^N (ground state of H_i = sum_i sigma_x_i)
    h(reg)
    
    # Trotter loop over steps
    for step in range(steps):
        theta = thetas[step]
        
        # =====================================================================
        # 2-body terms: R_YZ and R_ZY
        # For each pair [qi, qj] in G2:
        #   R_YZ(4*theta*h_i^x) where h_i^x = 1
        #   R_ZY(4*theta*h_j^x) where h_j^x = 1
        # =====================================================================
        for pair_idx in range(len(G2)):
            # Use unique names to avoid CUDA-Q captured variable conflicts
            qubit_a = G2[pair_idx][0]
            qubit_b = G2[pair_idx][1]
            
            angle_2body = 4.0 * theta  # Since h^x = 1
            
            # R_YZ gate: exp(-i * angle/2 * Y_a ⊗ Z_b)
            rx(np.pi / 2.0, reg[qubit_a])
            x.ctrl(reg[qubit_a], reg[qubit_b])
            rz(angle_2body, reg[qubit_a])
            x.ctrl(reg[qubit_a], reg[qubit_b])
            rx(-np.pi / 2.0, reg[qubit_a])
            
            # R_ZY gate: exp(-i * angle/2 * Z_a ⊗ Y_b)
            rx(np.pi / 2.0, reg[qubit_b])
            x.ctrl(reg[qubit_a], reg[qubit_b])
            rz(angle_2body, reg[qubit_b])
            x.ctrl(reg[qubit_a], reg[qubit_b])
            rx(-np.pi / 2.0, reg[qubit_b])
        
        # =====================================================================
        # 4-body terms: R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY
        # For each quad [q0, q1, q2, q3] in G4:
        #   R_YZZZ(8*theta*h_q0^x)
        #   R_ZYZZ(8*theta*h_q1^x)
        #   R_ZZYZ(8*theta*h_q2^x)
        #   R_ZZZY(8*theta*h_q3^x)
        # =====================================================================
        for quad_idx in range(len(G4)):
            # Use unique names to avoid CUDA-Q captured variable conflicts
            q0 = G4[quad_idx][0]
            q1 = G4[quad_idx][1]
            q2 = G4[quad_idx][2]
            q3 = G4[quad_idx][3]
            
            angle_4body = 8.0 * theta  # Since h^x = 1
            
            # -----------------------------------------------------------------
            # R_YZZZ: Y on q0, Z on others
            # -----------------------------------------------------------------
            rx(np.pi / 2.0, reg[q0])
            x.ctrl(reg[q0], reg[q1])
            x.ctrl(reg[q1], reg[q2])
            x.ctrl(reg[q2], reg[q3])
            rz(angle_4body, reg[q3])
            x.ctrl(reg[q2], reg[q3])
            x.ctrl(reg[q1], reg[q2])
            x.ctrl(reg[q0], reg[q1])
            rx(-np.pi / 2.0, reg[q0])
            
            # -----------------------------------------------------------------
            # R_ZYZZ: Y on q1, Z on others
            # -----------------------------------------------------------------
            rx(np.pi / 2.0, reg[q1])
            x.ctrl(reg[q0], reg[q1])
            x.ctrl(reg[q1], reg[q2])
            x.ctrl(reg[q2], reg[q3])
            rz(angle_4body, reg[q3])
            x.ctrl(reg[q2], reg[q3])
            x.ctrl(reg[q1], reg[q2])
            x.ctrl(reg[q0], reg[q1])
            rx(-np.pi / 2.0, reg[q1])
            
            # -----------------------------------------------------------------
            # R_ZZYZ: Y on q2, Z on others
            # -----------------------------------------------------------------
            rx(np.pi / 2.0, reg[q2])
            x.ctrl(reg[q0], reg[q1])
            x.ctrl(reg[q1], reg[q2])
            x.ctrl(reg[q2], reg[q3])
            rz(angle_4body, reg[q3])
            x.ctrl(reg[q2], reg[q3])
            x.ctrl(reg[q1], reg[q2])
            x.ctrl(reg[q0], reg[q1])
            rx(-np.pi / 2.0, reg[q2])
            
            # -----------------------------------------------------------------
            # R_ZZZY: Y on q3, Z on others
            # -----------------------------------------------------------------
            rx(np.pi / 2.0, reg[q3])
            x.ctrl(reg[q0], reg[q1])
            x.ctrl(reg[q1], reg[q2])
            x.ctrl(reg[q2], reg[q3])
            rz(angle_4body, reg[q3])
            x.ctrl(reg[q2], reg[q3])
            x.ctrl(reg[q1], reg[q2])
            x.ctrl(reg[q0], reg[q1])
            rx(-np.pi / 2.0, reg[q3])


# ============================================================================
# EXERCISE 6: Quantum-Enhanced Memetic Tabu Search (QE-MTS)
# ============================================================================
# Compare MTS with random initialization vs. quantum-seeded initialization

if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 6: Quantum-Enhanced MTS Comparison")
    print("=" * 60)

    # ============================================================================
    # Configuration
    # ============================================================================
    N_qemts = 5     # Problem size (use smaller N for CPU simulation)
    pop_size = 10   # Population size
    max_generations = 40
    n_shots_quantum = 2000

    print(f"\nConfiguration:")
    print(f"  N = {N_qemts}")
    print(f"  Population size = {pop_size}")
    print(f"  MTS generations = {max_generations}")
    print(f"  Quantum shots = {n_shots_quantum}")

    # ============================================================================
    # Step 1: Run quantum circuit to generate seed population
    # ============================================================================
    print("\n" + "-" * 40)
    print("Step 1: Generating quantum seed population")
    print("-" * 40)

    T_qe = 1.0
    n_steps_qe = 2
    dt_qe = T_qe / n_steps_qe

    G2_qe, G4_qe = get_interactions(N_qemts)
    print(f"  |G2| = {len(G2_qe)}, |G4| = {len(G4_qe)}")

    thetas_qe = []
    for step in range(1, n_steps_qe + 1):
        t = step * dt_qe
        theta_val = utils.compute_theta(t, dt_qe, T_qe, N_qemts, G2_qe, G4_qe)
        thetas_qe.append(theta_val)

    quantum_result = cudaq.sample(
        trotterized_circuit,
        N_qemts, G2_qe, G4_qe, n_steps_qe, dt_qe, T_qe, thetas_qe,
        shots_count=n_shots_quantum
    )

    quantum_samples = list(quantum_result)
    print(f"  Obtained {len(quantum_samples)} unique bitstrings from quantum sampling")

    quantum_population = []
    quantum_energies = []
    for bitstring in quantum_samples[:pop_size]:
        seq = np.array([1 if b == '0' else -1 for b in bitstring])
        quantum_population.append(seq)
        quantum_energies.append(compute_energy(seq))

    print(f"  Initial quantum population energies: min={min(quantum_energies)}, "
          f"mean={np.mean(quantum_energies):.1f}, max={max(quantum_energies)}")

    # ============================================================================
    # Step 2: Run MTS with quantum-seeded population
    # ============================================================================
    print("\n" + "-" * 40)
    print("Step 2: Running Quantum-Enhanced MTS")
    print("-" * 40)

    qe_best_seq, qe_best_energy, qe_final_pop, qe_history = memetic_tabu_search(
        N=N_qemts,
        pop_size=pop_size,
        generations=max_generations,
        p_mut=0.2,
        tabu_iterations=50,
        tabu_tenure=5,
        initial_population=quantum_population,
        verbose=True
    )

    # ============================================================================
    # Step 3: Run MTS with random initialization (baseline)
    # ============================================================================
    print("\n" + "-" * 40)
    print("Step 3: Running Classical MTS (Random Init)")
    print("-" * 40)

    random_best_seq, random_best_energy, random_final_pop, random_history = memetic_tabu_search(
        N=N_qemts,
        pop_size=pop_size,
        generations=max_generations,
        p_mut=0.2,
        tabu_iterations=50,
        tabu_tenure=5,
        initial_population=None,
        verbose=True
    )

    # ============================================================================
    # Step 4: Compare Results
    # ============================================================================
    print("\n" + "=" * 60)
    print("COMPARISON: Quantum-Enhanced vs Classical MTS")
    print("=" * 60)

    print(f"\nQuantum-Enhanced MTS:")
    print(f"  Best energy: {qe_best_energy}")
    print(f"  Best sequence: {qe_best_seq}")

    print(f"\nClassical MTS (Random Init):")
    print(f"  Best energy: {random_best_energy}")
    print(f"  Best sequence: {random_best_seq}")

    known_optimal = {
        5: 2, 7: 2, 9: 2, 11: 2, 13: 4, 15: 6, 17: 6, 19: 10, 21: 10
    }
    if N_qemts in known_optimal:
        print(f"\nKnown optimal for N={N_qemts}: E = {known_optimal[N_qemts]}")
        print(f"  QE-MTS gap from optimal: {qe_best_energy - known_optimal[N_qemts]}")
        print(f"  Classical gap from optimal: {random_best_energy - known_optimal[N_qemts]}")

    # ============================================================================
    # Step 5: Visualization
    # ============================================================================
    print("\n" + "-" * 40)
    print("Generating comparison plots...")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    qe_generations = list(range(len(qe_history)))
    random_generations = list(range(len(random_history)))

    ax1 = axes[0, 0]
    ax1.plot(qe_generations, qe_history, 'b-', label='Best', linewidth=2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Energy')
    ax1.set_title('QE-MTS Energy Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    qe_final_energies = [compute_energy(seq) for seq in qe_final_pop]
    ax2.hist(qe_final_energies, bins=10, edgecolor='black', alpha=0.7, color='blue')
    ax2.axvline(x=qe_best_energy, color='r', linestyle='--', label=f'Best: {qe_best_energy}')
    ax2.set_xlabel('Energy')
    ax2.set_ylabel('Count')
    ax2.set_title('QE-MTS Final Population')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    autocorr_qe = compute_autocorrelation(qe_best_seq)
    ax3.bar(range(len(autocorr_qe)), autocorr_qe, alpha=0.7, color='blue')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Lag k')
    ax3.set_ylabel('C_k')
    ax3.set_title('QE-MTS Best Autocorrelation')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    ax4.plot(random_generations, random_history, 'r-', label='Best', linewidth=2)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Energy')
    ax4.set_title('Classical MTS Energy Convergence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    random_final_energies = [compute_energy(seq) for seq in random_final_pop]
    ax5.hist(random_final_energies, bins=10, edgecolor='black', alpha=0.7, color='red')
    ax5.axvline(x=random_best_energy, color='b', linestyle='--', label=f'Best: {random_best_energy}')
    ax5.set_xlabel('Energy')
    ax5.set_ylabel('Count')
    ax5.set_title('Classical MTS Final Population')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    autocorr_rand = compute_autocorrelation(random_best_seq)
    ax6.bar(range(len(autocorr_rand)), autocorr_rand, alpha=0.7, color='red')
    ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax6.set_xlabel('Lag k')
    ax6.set_ylabel('C_k')
    ax6.set_title('Classical MTS Best Autocorrelation')
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'Quantum-Enhanced vs Classical MTS (N={N_qemts})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ============================================================================
    # Summary Statistics
    # ============================================================================
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\n{'Metric':<35} {'QE-MTS':>12} {'Classical':>12}")
    print("-" * 60)
    print(f"{'Initial population mean energy':<35} {np.mean(quantum_energies):>12.2f} {np.mean([compute_energy(s) for s in initialize_population_random(N_qemts, pop_size)]):>12.2f}")
    print(f"{'Final best energy':<35} {qe_best_energy:>12} {random_best_energy:>12}")
    print(f"{'Final population mean energy':<35} {np.mean(qe_final_energies):>12.2f} {np.mean(random_final_energies):>12.2f}")

    qe_best_gen = qe_history.index(qe_best_energy) if qe_best_energy in qe_history else len(qe_history)-1
    random_best_gen = random_history.index(random_best_energy) if random_best_energy in random_history else len(random_history)-1
    print(f"{'Convergence (generation to best)':<35} {qe_best_gen:>12} {random_best_gen:>12}")