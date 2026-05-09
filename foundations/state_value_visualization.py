"""
State-Value Function Visualization on a Simple Grid

We compute and visualize V(s) — the expected total reward an agent
collects starting from state s, following a given policy.

Environment: 4x4 GridWorld
- Start: top-left (0,0)
- Goal: bottom-right (3,3)  → reward +1, episode ends
- Walls: none
- Holes: (1,1), (1,3), (2,3), (3,0)  → reward -1, episode ends
- All other steps: reward 0

We evaluate two policies:
  1. Uniform random (all 4 directions equally likely)
  2. Greedy-toward-goal (biased toward right/down)

Value computation via iterative policy evaluation (Bellman expectation).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')


# ---------- Environment Definition ----------

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4   # 0=Up, 1=Down, 2=Left, 3=Right

GOAL_STATE = 15   # (3,3)
HOLE_STATES = {5, 7, 11, 12}   # (1,1),(1,3),(2,3),(3,0)
TERMINAL_STATES = {GOAL_STATE} | HOLE_STATES

ACTION_DELTAS = {
    0: (-1,  0),   # Up
    1: ( 1,  0),   # Down
    2: ( 0, -1),   # Left
    3: ( 0,  1),   # Right
}

GAMMA = 0.99   # discount factor


def state_to_pos(s):
    return s // GRID_SIZE, s % GRID_SIZE


def pos_to_state(r, c):
    return r * GRID_SIZE + c


def step(state, action):
    """Return (next_state, reward). Deterministic transitions."""
    if state in TERMINAL_STATES:
        return state, 0.0

    r, c = state_to_pos(state)
    dr, dc = ACTION_DELTAS[action]
    nr = np.clip(r + dr, 0, GRID_SIZE - 1)
    nc = np.clip(c + dc, 0, GRID_SIZE - 1)
    next_state = pos_to_state(nr, nc)

    if next_state == GOAL_STATE:
        return next_state, 1.0
    if next_state in HOLE_STATES:
        return next_state, -1.0
    return next_state, 0.0


def build_transition_matrix(policy_probs):
    """
    policy_probs: (N_STATES, N_ACTIONS) array
    Returns P[s, s'] = sum_a pi(a|s) * I[next(s,a)=s']
    and R[s] = sum_a pi(a|s) * r(s,a)
    """
    P = np.zeros((N_STATES, N_STATES))
    R = np.zeros(N_STATES)
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            ns, r = step(s, a)
            P[s, ns] += policy_probs[s, a]
            R[s] += policy_probs[s, a] * r
    return P, R


def policy_evaluation(policy_probs, gamma=GAMMA, theta=1e-8):
    """Iterative policy evaluation — solve V = R + gamma * P * V."""
    P, R = build_transition_matrix(policy_probs)
    V = np.zeros(N_STATES)
    for _ in range(10_000):
        V_new = R + gamma * P @ V
        # Terminal states always have value 0 (no future reward from inside)
        for ts in TERMINAL_STATES:
            V_new[ts] = 0.0
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new
    return V


# ---------- Policies ----------

def uniform_random_policy():
    return np.full((N_STATES, N_ACTIONS), 0.25)


def biased_toward_goal_policy():
    """Prefer Down and Right (toward goal at bottom-right)."""
    probs = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        r, c = state_to_pos(s)
        if s in TERMINAL_STATES:
            probs[s] = 0.25
            continue
        # Weight: down > right > up > left based on distance to goal
        w = np.array([0.1, 0.4, 0.1, 0.4])   # Up, Down, Left, Right
        # Adjust if at edge
        if r == 0: w[0] = 0.0   # can't go further Up
        if r == GRID_SIZE - 1: w[1] = 0.0
        if c == 0: w[2] = 0.0
        if c == GRID_SIZE - 1: w[3] = 0.0
        probs[s] = w / w.sum()
    return probs


# ---------- Visualization ----------

CELL_LABELS = {0: 'S', GOAL_STATE: 'G'}
for hs in HOLE_STATES:
    CELL_LABELS[hs] = 'H'


def plot_value_function(V, title, out_path, policy_probs=None):
    grid = V.reshape(GRID_SIZE, GRID_SIZE)

    fig, ax = plt.subplots(figsize=(7, 7))

    vmax = max(abs(V.min()), abs(V.max())) + 0.01
    im = ax.imshow(grid, cmap='RdYlGn', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="State Value V(s)")

    for s in range(N_STATES):
        r, c = state_to_pos(s)
        label = CELL_LABELS.get(s, '')
        val_str = f"{V[s]:.3f}"
        cell_text = f"{label}\n{val_str}" if label else val_str

        ax.text(c, r, cell_text, ha='center', va='center',
                fontsize=10 if label else 9,
                fontweight='bold' if label else 'normal',
                color='black')

        # Draw policy arrows if provided
        if policy_probs is not None and s not in TERMINAL_STATES:
            best_action = np.argmax(policy_probs[s])
            dr, dc = ACTION_DELTAS[best_action]
            ax.annotate('', xy=(c + dc * 0.35, r + dr * 0.35),
                        xytext=(c, r),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Grid lines
    for i in range(GRID_SIZE + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.8)
        ax.axvline(i - 0.5, color='gray', linewidth=0.8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)

    legend_patches = [
        mpatches.Patch(color='#d5e8d4', label='S = Start'),
        mpatches.Patch(color='#f8cecc', label='H = Hole (reward -1)'),
        mpatches.Patch(color='#d5e8d4', label='G = Goal (reward +1)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Value function plot saved to {out_path}")


def main():
    print("=== State-Value Function Visualization ===")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}  |  Goal: state {GOAL_STATE}  |  Holes: {sorted(HOLE_STATES)}")
    print(f"Discount factor γ = {GAMMA}")
    print()

    # Policy 1: Uniform Random
    rand_policy = uniform_random_policy()
    V_rand = policy_evaluation(rand_policy)
    print("Uniform Random Policy — State Values:")
    print(V_rand.reshape(GRID_SIZE, GRID_SIZE).round(3))
    print()

    # Policy 2: Biased toward goal
    goal_policy = biased_toward_goal_policy()
    V_goal = policy_evaluation(goal_policy)
    print("Biased-Toward-Goal Policy — State Values:")
    print(V_goal.reshape(GRID_SIZE, GRID_SIZE).round(3))
    print()

    print("Observation: States closer to the goal (bottom-right) have higher values.")
    print("States adjacent to holes are penalized. The biased policy achieves")
    print("substantially higher values near the start than the random policy.")

    plot_value_function(
        V_rand,
        title=f"State Values: Uniform Random Policy (γ={GAMMA})",
        out_path="outputs/value_function_random_policy.png",
        policy_probs=rand_policy,
    )
    plot_value_function(
        V_goal,
        title=f"State Values: Biased-Toward-Goal Policy (γ={GAMMA})",
        out_path="outputs/value_function_biased_policy.png",
        policy_probs=goal_policy,
    )


if __name__ == "__main__":
    main()
